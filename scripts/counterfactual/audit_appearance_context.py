"""Audit full context geometry/rankings and independently recheck each winner.

All groups and missing witnesses stay in the denominators. Literal centered
pixel comparisons recheck the unique top1 alternatives of all four rankings
and the zero control. Non-winning path pixels are not all recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .audit_appearance_paths import require, close, pearson
from .probe_native_region_motion import resolve_media, load_masks
from .static_jitter import digest


def expected_groups(pair, masks, shape):
    """Literal radial inequalities, without the probe's support constructor."""
    yy, xx = np.indices(shape)
    groups = {}
    for point_id, point in enumerate(pair["source_points"]):
        distance2 = (xx - point["xy"][0]) ** 2 + (yy - point["xy"][1]) ** 2
        for ref in point["regional_supports"]:
            region = np.ones(shape, bool) if ref["whole_frame_control"] else masks[ref["region"]]
            radius2 = (point["size"] * ref["factor"]) ** 2
            supports = (region & (distance2 <= radius2), region & (distance2 > radius2) & (distance2 <= 4 * radius2),
                        region & (distance2 > 4 * radius2) & (distance2 <= 16 * radius2))
            keys = tuple(hashlib.sha256(np.packbits(m).tobytes()).hexdigest() for m in supports)
            require(keys[0] == ref["support_sha256"], "bound source support changed")
            group = groups.setdefault(keys, {"masks": supports, "references": []})
            group["references"].append({"source_point": point_id, **ref})
    return groups


def literal_evidence(source, target, mask, displacement):
    y, x = np.nonzero(mask)
    tx, ty = x + displacement[0], y + displacement[1]
    visible = (tx >= 0) & (tx < mask.shape[1]) & (ty >= 0) & (ty < mask.shape[0])
    overlap = float(visible.mean()) if len(x) else 0.
    if len(x) < 3 or visible.sum() < 3 or overlap < .8:
        return {**{name: np.nan for name in ("ncc", "zero_ncc", "mse", "zero_mse")}, "overlap": overlap if len(x) >= 3 else 0.}
    a, b, zero = source[y[visible], x[visible]], target[ty[visible], tx[visible]], target[y[visible], x[visible]]
    return {"ncc": pearson(a, b), "zero_ncc": pearson(a, zero), "mse": float(np.mean((a - b) ** 2)),
            "zero_mse": float(np.mean((a - zero) ** 2)), "overlap": overlap}


def verify_group(arrays, row, prior, geometry, images, group_id, offset):
    require(row["path_start"] == offset and row["score"] is None and row["references"] == geometry["references"], "group references/offset/null mismatch")
    require(row["support_pixels"] == [int(m.sum()) for m in geometry["masks"]], "witness area mismatch")
    candidates = [h["displacement_pixels"] for h in prior["hypotheses"]]
    require(row["parent_hypotheses"] == len(candidates), "parent hypothesis count mismatch")
    zero = candidates.index([0, 0]) if [0, 0] in candidates else len(candidates)
    if [0, 0] not in candidates:
        candidates.append([0, 0])
    n = len(candidates); sl = slice(offset, offset + n)
    require(row["candidate_count"] == n and row["zero_control_index"] == zero, "zero control or candidate count changed")
    require(np.array_equal(arrays["displacements"][sl], candidates) and np.all(arrays["group"][sl] == group_id), "candidate membership mismatch")
    close(arrays["core_ncc"][offset:offset + len(prior["hypotheses"])], [h["correlation"] for h in prior["hypotheses"]], "source NCC mismatch")
    selected = {zero}; variants = []; availability = {}
    for name, roles in (("core", ["core"]), ("core_inner", ["core", "inner"]), ("core_outer", ["core", "outer"]),
                        ("core_both", ["core", "inner", "outer"])):
        values = np.minimum.reduce([arrays[role + "_ncc"][sl] for role in roles])
        observable = [i for i in range(n) if np.isfinite(values[i])]
        ranked = sorted(observable, key=lambda i: (-values[i], i))[:3]
        variants.append({"variant": name, "observable_hypotheses": len(observable), "ranked_indices": ranked,
                         "ranked_ncc": values[ranked].tolist(), "score": None})
        availability[name] = bool(ranked)
        if ranked:
            selected.add(ranked[0])
    require(row["rankings"] == variants, "ranking or missing evidence differs")
    for index in sorted(selected):
        for role, mask in zip(("core", "inner", "outer"), geometry["masks"]):
            checked = literal_evidence(images[0], images[1], mask, candidates[index])
            for name, value in checked.items():
                close(value, arrays[role + "_" + name][offset + index], "literal pixel mismatch: " + role + "_" + name)
    return n, len(selected), availability


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "appearance-run", "execution-root", "appearance-execution-root", "manifest", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    source, parent, execution, parent_execution, manifest, regions, output = map(Path, (args.source_run, args.appearance_run, args.execution_root,
                                                                                     args.appearance_execution_root, args.manifest, args.region_run, args.output))
    require(not output.exists(), "fresh analysis required")
    ledger = {r["candidate_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    identities, runtimes, bindings, stats, controls, signatures = [], [], {}, [], [], {}
    cv2.setNumThreads(1)
    for shard in sorted(source.glob("shard-*")):
        identity = json.loads((shard / "provenance.json").read_text())
        runtime = json.loads((shard / "runtime.json").read_text())
        parent_shard = parent / shard.name
        origin = json.loads((parent_shard / "provenance.json").read_text())
        parent_state = json.loads((parent_shard / "runtime.json").read_text())
        for run, metadata, state in ((shard, identity, runtime), (parent_shard, origin, parent_state)):
            require(state["status"] == "finished" and not state["failed"] and state["completed"] == state["expected"], "incomplete or failed shard")
            require(digest(run / "diagnostics.jsonl") == metadata["diagnostics_sha256"] and digest(manifest) == metadata["manifest_sha256"], "changed records/manifest")
            require(digest(regions / "provenance.json") == metadata["region_provenance_sha256"], "changed SAM identity")
        require(identity["source_provenance_sha256"] == digest(parent_shard / "provenance.json") and identity["source_diagnostics_sha256"] == origin["diagnostics_sha256"], "changed appearance evidence")
        require(identity["sharding"] == origin["sharding"] and identity["starts"] == origin["starts"] and identity["lags"] == origin["lags"] == [1], "cohort/phases mismatch")
        for metadata, root, script in ((identity, execution, "probe_appearance_context.py"), (origin, parent_execution, "probe_local_appearance.py")):
            for name, sha in {**metadata["code_files"], **metadata["helper_sha256"], "scripts/counterfactual/" + script: metadata["script_sha256"]}.items():
                require(digest(root / name) == sha, "changed execution snapshot: " + name)
        records = list(map(json.loads, (shard / "diagnostics.jsonl").read_text().splitlines()))
        parents = {r["candidate_id"]: r for r in map(json.loads, (parent_shard / "diagnostics.jsonl").read_text().splitlines())}
        require([r["candidate_id"] for r in records] == identity["sharding"]["assigned"] and len(records) == runtime["completed"], "missing assigned record")
        files = set()
        for record in records:
            key = record["candidate_id"]
            frames, times, _ = decode_video(resolve_media(ledger[key], args.video_root), TrajectoryConfig(sample_fps=8, max_side=512))
            require(identity["input_sha256"][key] == ledger[key]["sha256"] and list(frames.shape) == ledger[key]["decoded_shape"] and np.array_equal(times, ledger[key]["pts"]), "native input mismatch")
            require(record["status"] == "diagnostic_only" and record["score"] is None and np.array_equal(times, record["sampling"]["timestamps"]), "sampling or null contract mismatch")
            require([(p["start"], p["lag"]) for p in record["pairs"]] == [(i, 1) for i in range(len(frames) - 1)], "missing adjacent phase")
            masks = load_masks(regions / "evidence" / (key + ".npz"), identity["region_cache_sha256"][key], frames, times)
            signature = hashlib.sha256(json.dumps(record["pairs"], sort_keys=True).encode())
            summary = {"candidate_id": key, "pairs": len(record["pairs"]), "groups": 0, "comparisons": 0, "pixel_rechecked_candidates": 0,
                       "groups_with_ranking": {v: 0 for v in ("core", "core_inner", "core_outer", "core_both")}}
            for row, original in zip(record["pairs"], parents[key]["pairs"]):
                start = row["start"]
                require(row["score"] is None and row["status"] == "diagnostic_only" and row["source_queries"] == len(original["source_points"]) and row["seconds"] == times[start + 1] - times[start], "pair contract mismatch")
                file = shard / "evidence" / f"{key}_start{start:02d}.npz"
                files.add(file.name)
                require(digest(file) == identity["evidence_sha256"][file.name], "changed arrays")
                with np.load(file, allow_pickle=False) as cache:
                    arrays = {k: cache[k] for k in cache.files}
                require(np.array_equal(arrays["timestamps"], times[start:start + 2]), "array timeline mismatch")
                expected = expected_groups(original, masks[start], frames.shape[1:3])
                require([(r["core_key"], r["inner_key"], r["outer_key"]) for r in row["groups"]] == sorted(expected), "missing or duplicated group")
                offset = 0; rechecked = 0
                images = frames[start:start + 2].astype(float) / 255.
                for index, (keys, group) in enumerate(sorted(expected.items())):
                    count, pixels, availability = verify_group(arrays, row["groups"][index], original["supports"][keys[0]], group, images, index, offset)
                    offset += count; rechecked += pixels
                    for name, value in availability.items():
                        summary["groups_with_ranking"][name] += int(value)
                require(offset == row["candidate_comparisons"] == len(arrays["group"]), "comparison count mismatch")
                summary["groups"] += len(expected); summary["comparisons"] += offset; summary["pixel_rechecked_candidates"] += rechecked
                for name, value in sorted(arrays.items()):
                    signature.update(name.encode()); signature.update(str((value.shape, value.dtype.str)).encode()); signature.update(value.tobytes())
                print(json.dumps({"candidate_id": key, "start": start, "groups": len(expected), "pixel_rechecked_candidates": rechecked}), flush=True)
            if record["family"] == "original":
                signatures[record["base_id"]] = signature.hexdigest()
            elif record["family"] == "encoding_control":
                require(signatures.get(record["base_id"]) == signature.hexdigest(), "encoding control fields/arrays differ")
                controls.append({"candidate_id": key, "all_fields_and_arrays_exact": True})
            stats.append(summary)
        require(files == set(identity["evidence_sha256"]) == {f.name for f in (shard / "evidence").glob("*.npz")}, "evidence inventory mismatch")
        require(runtime["pairs_completed"] == sum(len(r["pairs"]) for r in records), "runtime pair count mismatch")
        identities.append(identity); runtimes.append(runtime); bindings[shard.name] = digest(shard / "provenance.json")
    require(bool(identities), "no shards")
    reference = identities[0]
    require(sorted(s["candidate_id"] for s in stats) == sorted(reference["sharding"]["selected_cohort"]), "incomplete/duplicated cohort")
    require(sorted(i["sharding"]["shard"] for i in identities) == list(range(reference["sharding"]["shards"])), "shard coverage mismatch")
    for identity in identities:
        for name in ("config", "input_sha256", "review_sha256", "code_files", "script_sha256", "helper_sha256", "region_cache_sha256"):
            require(identity[name] == reference[name], "inconsistent shard: " + name)
    output.mkdir(parents=True)
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "videos": stats, "runtime": runtimes,
              "source_provenance_sha256": bindings, "encoding_controls": controls, "script_sha256": digest(Path(__file__)),
              "audit_helper_sha256": digest(Path(__file__).with_name("audit_appearance_paths.py")),
              "verified": "all references/disjoint masks/candidate membership/rankings; unique top1s of all variants plus zero control independently rechecked from RGB",
              "not_verified": "all non-winning candidate pixels, physical identity, invariance, true periodic motion, final Repair"}
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
