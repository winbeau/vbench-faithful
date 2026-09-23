"""Audit path enumeration and rankings; independently recheck top1 RGB evidence.

All saved paths are checked algebraically. Pixel recomputation covers every
top1 per declared support/query group, not every path or global search surface.
Neither layer certifies physical identity or a successful Dynamic repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.local_appearance import feature_support
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import load_masks, resolve_media
from .static_jitter import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=1e-11, atol=2e-11, equal_nan=True), message)


def pearson(a, b):
    """Literal centered RGB samples, independent of streaming moment code."""
    if len(a) < 3:
        return np.nan
    aa, bb = a - a.mean(axis=0), b - b.mean(axis=0)
    ea, eb = np.sum(aa ** 2), np.sum(bb ** 2)
    floor = 128 * np.finfo(float).eps * max(1., np.sum(a ** 2), np.sum(b ** 2))
    return float(np.clip(np.sum(aa * bb) / np.sqrt(ea * eb), -1, 1)) if min(ea, eb) > floor else np.nan


def sample(frame, support, offset):
    y, x = np.nonzero(support)
    x, y = x + offset[0], y + offset[1]
    valid = (x >= 0) & (x <= frame.shape[1] - 1) & (y >= 0) & (y <= frame.shape[0] - 1)
    # Limit OpenCV's dimension per call, without dropping any pixels.
    chunks = [cv2.remap(frame, x[i:i + 8192].astype(np.float32)[None], y[i:i + 8192].astype(np.float32)[None],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0] for i in range(0, len(x), 8192)]
    return (np.concatenate(chunks) if chunks else np.empty((0, 3))), valid


def pair_check(frames, support, offset):
    a = frames[0][support]
    b, visible = sample(frames[1], support, offset)
    overlap = float(visible.mean()) if len(visible) else 0.
    return (pearson(a[visible], b[visible]) if overlap >= .8 else np.nan), overlap


def common_check(frames, support, first, total):
    a = frames[0][support]
    b, vb = sample(frames[1], support, first)
    c, vc = sample(frames[2], support, total)
    visible = vb & vc
    fraction = float(visible.mean()) if len(visible) else 0.
    values = [pearson(x[visible], y[visible]) if fraction >= .8 else np.nan for x, y in ((a, b), (a, c), (b, c))]
    return values, fraction


def support_mask(pair, support_key, masks, shape):
    for point in pair["source_points"]:
        for ref in point["regional_supports"]:
            if ref["support_sha256"] != support_key:
                continue
            region = np.ones(shape, bool) if ref["whole_frame_control"] else masks[ref["region"]]
            mask = feature_support(region, point["xy"], point["size"], ref["factor"])
            require(hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() == support_key, "source geometry hash mismatch")
            return mask
    raise ValueError("unreferenced support key")


def verify_rankings(arrays, row, mode):
    values = ([arrays[k] for k in ("actual_first_ncc", "cached_second_ncc", "reference_ncc")] if mode == "detected"
              else [arrays[k] for k in ("common_first_ncc", "common_second_ncc", "common_reference_ncc")])
    merit = np.minimum.reduce(values)
    close(arrays["bottleneck_ncc"], merit, "bottleneck mismatch")
    require(len(merit) == row["candidate_paths"] and int(np.isfinite(merit).sum()) == row["observable_reference_paths"], "path totals mismatch")
    keys = (np.c_[arrays["source_point"], arrays["source_support"]] if mode == "detected" else arrays["source_support"][:, None])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    if mode == "detected":
        require([(r["source_point"], r["source_support"]) for r in row["rankings"]] == list(map(tuple, unique)), "ranking groups mismatch")
    else:
        require([r["source_support"] for r in row["rankings"]] == list(range(len(arrays["source_support_keys"]))), "reference support group mismatch")
    selected = []
    groups = {tuple(key): np.flatnonzero(inverse == i) for i, key in enumerate(unique)}
    for ranking in row["rankings"]:
        group = (ranking["source_point"], ranking["source_support"]) if mode == "detected" else (ranking["source_support"],)
        ids = groups.get(group, np.empty(0, int))
        observed = [int(i) for i in ids if np.isfinite(merit[i])]
        best = sorted(observed, key=lambda i: (-merit[i], i))[:3]
        require(ranking["candidate_paths"] == len(ids) and ranking["ranked_path_indices"] == best, "ranking changed or paths omitted")
        count_name = "reference_observable_paths" if mode == "detected" else "observable_paths"
        require(ranking[count_name] == len(observed), "observed group count mismatch")
        if best:
            selected.append(best[0])
    return selected


def verify_detected(arrays, row, first, second):
    """Membership + uniqueness + complete independent degree count."""
    source, middle = first["source_points"], second["source_points"]
    require(arrays["source_support_keys"].tolist() == sorted(first["supports"]), "first support inventory mismatch")
    require(arrays["middle_support_keys"].tolist() == sorted(second["supports"]), "middle support inventory mismatch")
    tables = []
    for pair, points, role in ((first, source, "source"), (second, middle, "middle")):
        keys = arrays[f"{role}_support_keys"].tolist()
        index = {k: i for i, k in enumerate(keys)}
        allowed = np.zeros((len(points), len(keys)), bool)
        for i, point in enumerate(points):
            allowed[i, [index[r["support_sha256"]] for r in point["regional_supports"]]] = True
        hypotheses = [pair["supports"][k]["hypotheses"] for k in keys]
        counts = np.array([len(h) for h in hypotheses], int)
        offsets = np.r_[0, np.cumsum(counts)]
        flattened = [h for group in hypotheses for h in group]
        point, support = arrays[f"{role}_point"], arrays[f"{role}_support"]
        rank = arrays["first_rank" if role == "source" else "second_rank"]
        require(np.all((point >= 0) & (point < len(points))) and np.all((support >= 0) & (support < len(keys))), "path index out of range")
        require(allowed[point, support].all() and np.all((rank >= 0) & (rank < counts[support])), "path uses absent point support or hypothesis")
        selection = offsets[support] + rank
        displacement = np.array([h["displacement_pixels"] for h in flattened], float).reshape(-1, 2)
        ncc = np.array([h["correlation"] for h in flattened])
        close(arrays["cached_first_ncc" if role == "source" else "cached_second_ncc"], ncc[selection], "cached NCC changed")
        tables.append((allowed, counts, offsets, displacement, selection))
    xy1, xy2 = (np.array([p["xy"] for p in points], float).reshape(-1, 2) for points in (source, middle))
    i, j = arrays["source_point"], arrays["middle_point"]
    actual = xy2[j] - xy1[i]
    close(arrays["first_offset"], actual, "snapped offset mismatch")
    close(arrays["total_offset"], actual + tables[1][3][tables[1][4]], "composed offset mismatch")
    error = np.linalg.norm(actual - tables[0][3][tables[0][4]], axis=1)
    sizes1, sizes2 = (np.array([p["size"] for p in points]) for points in (source, middle))
    radius = np.maximum(1., .5 * np.sqrt(sizes1[i] * sizes2[j]))
    close(arrays["association_error"], error, "association error mismatch")
    close(arrays["association_radius"], radius, "association radius mismatch")
    require(np.all(error <= radius + 1e-12), "geometrically inadmissible path")
    tuples = np.c_[i, arrays["source_support"], arrays["first_rank"], j, arrays["middle_support"], arrays["second_rank"]]
    require(len(np.unique(tuples, axis=0)) == len(tuples), "duplicate hypothesis path")
    expected = 0
    associations = []
    outgoing = tables[1][0] @ tables[1][1]
    for q, point in enumerate(source):
        support_ids = np.flatnonzero(tables[0][0][q])
        displacements = np.concatenate([tables[0][3][tables[0][2][k]:tables[0][2][k + 1]] for k in support_ids]) if len(support_ids) else np.empty((0, 2))
        endpoints = np.asarray(point["xy"]) + displacements
        distances = np.linalg.norm(endpoints[:, None] - xy2[None], axis=2)
        compatible = distances <= np.maximum(1., .5 * np.sqrt(point["size"] * sizes2))[None]
        linked = int(compatible.any(axis=1).sum())
        expected += int(np.sum(compatible * outgoing[None]))
        associations.append({"source_point": q, "declared_first_hypotheses": len(displacements), "with_middle_detection": linked,
                             "without_middle_detection": len(displacements) - linked})
    require(expected == len(tuples) and associations == row["associations"], "incomplete admissible path enumeration")


def verify_reference(arrays, row, first):
    keys = arrays["source_support_keys"].tolist()
    require(keys == sorted(first["supports"]) and row["unique_supports"] == len(keys), "reference support inventory mismatch")
    offset = 0
    for sid, key in enumerate(keys):
        ranking = row["rankings"][sid]
        a, b = first["supports"][key]["hypotheses"], ranking["reference_hypotheses"]
        n = len(a) * len(b)
        sl = slice(offset, offset + n)
        require(ranking["path_start"] == offset and ranking["score"] is None, "path start/null contract mismatch")
        require(np.all(arrays["source_support"][sl] == sid), "support path membership mismatch")
        rank1, rank2 = np.repeat(np.arange(len(a)), len(b)), np.tile(np.arange(len(b)), len(a))
        require(np.array_equal(arrays["first_rank"][sl], rank1) and np.array_equal(arrays["reference_rank"][sl], rank2), "incomplete reference Cartesian paths")
        close(arrays["first_offset"][sl], np.array([h["displacement_pixels"] for h in a]).reshape(-1, 2)[rank1], "reference first offsets changed")
        close(arrays["total_offset"][sl], np.array([h["displacement_pixels"] for h in b]).reshape(-1, 2)[rank2], "reference final offsets changed")
        offset += n
    require(offset == row["candidate_paths"], "reference path count mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "appearance-run", "execution-root", "appearance-execution-root", "manifest", "region-run", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--video-root", action="append", default=[])
    args = parser.parse_args(argv)
    source, parent, execution, parent_execution, manifest, regions, output = map(Path, (args.source_run, args.appearance_run, args.execution_root,
                                                                                     args.appearance_execution_root, args.manifest, args.region_run, args.output))
    require(not output.exists(), "fresh analysis required")
    ledger = {r["candidate_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    cv2.setNumThreads(1)
    stats, controls, signatures, runtimes, bindings, identities, all_ids = [], [], {}, [], {}, [], []
    for shard in sorted(source.glob("shard-*")):
        identity = json.loads((shard / "provenance.json").read_text())
        runtime = json.loads((shard / "runtime.json").read_text())
        parent_shard = parent / shard.name
        origin = json.loads((parent_shard / "provenance.json").read_text())
        state = json.loads((parent_shard / "runtime.json").read_text())
        mode = identity.get("mode", "detected")
        require(mode in {"detected", "reference"}, "unknown path protocol")
        for run, metadata, status in ((shard, identity, runtime), (parent_shard, origin, state)):
            require(status["status"] == "finished" and status["failed"] == 0 and status["completed"] == status["expected"], "incomplete or failed shard")
            require(digest(run / "diagnostics.jsonl") == metadata["diagnostics_sha256"], "changed records")
            require(digest(manifest) == metadata["manifest_sha256"] and digest(regions / "provenance.json") == metadata["region_provenance_sha256"], "changed input/SAM binding")
        for metadata, root, script in ((identity, execution, "probe_appearance_paths.py"), (origin, parent_execution, "probe_local_appearance.py")):
            for name, sha in {**metadata["code_files"], **metadata["helper_sha256"], "scripts/counterfactual/" + script: metadata["script_sha256"]}.items():
                require(digest(root / name) == sha, "changed execution source: " + name)
        require(digest(parent_shard / "provenance.json") == identity["source_provenance_sha256"] and origin["diagnostics_sha256"] == identity["source_diagnostics_sha256"], "changed appearance parent")
        require(identity["sharding"] == origin["sharding"], "parent and child cohort differ")
        parents = {r["candidate_id"]: r for r in map(json.loads, (parent_shard / "diagnostics.jsonl").read_text().splitlines())}
        records = list(map(json.loads, (shard / "diagnostics.jsonl").read_text().splitlines()))
        require([r["candidate_id"] for r in records] == identity["sharding"]["assigned"] and len(records) == runtime["completed"], "assigned cohort mismatch")
        evidence_names = set()
        for row in records:
            key = row["candidate_id"]
            frames, times, _ = decode_video(resolve_media(ledger[key], args.video_root), TrajectoryConfig(sample_fps=8, max_side=512))
            require(identity["input_sha256"][key] == ledger[key]["sha256"] and list(frames.shape) == ledger[key]["decoded_shape"] and np.array_equal(times, ledger[key]["pts"]), "native source mismatch")
            masks = load_masks(regions / "evidence" / (key + ".npz"), identity["region_cache_sha256"][key], frames, times)
            require(row["status"] == "diagnostic_only" and row["score"] is None and np.array_equal(times, row["sampling"]["timestamps"]), "score or native timeline mismatch")
            require([t["start"] for t in row["triplets"]] == list(range(len(frames) - 2)), "missing time phase")
            native = frames.astype(float) / 255.
            signature = hashlib.sha256(json.dumps(row["triplets"], sort_keys=True).encode())
            checked, path_count, linked, declared = 0, 0, 0, 0
            for triplet in row["triplets"]:
                start = triplet["start"]
                require(triplet["status"] == "diagnostic_only" and triplet["score"] is None and triplet["frames"] == [start, start + 1, start + 2], "triplet contract changed")
                file = shard / "evidence" / f"{key}_start{start:02d}.npz"
                evidence_names.add(file.name)
                require(digest(file) == identity["evidence_sha256"][file.name], "changed path arrays")
                with np.load(file, allow_pickle=False) as cache:
                    arrays = {k: cache[k] for k in cache.files}
                require(np.array_equal(arrays["timestamps"], times[start:start + 3]), "array timeline mismatch")
                first, second = parents[key]["pairs"][start:start + 2]
                require(triplet["source_queries"] == len(first["source_points"]), "source query count mismatch")
                if mode == "detected":
                    verify_detected(arrays, triplet, first, second)
                    linked += sum(a["with_middle_detection"] for a in triplet["associations"])
                    declared += sum(a["declared_first_hypotheses"] for a in triplet["associations"])
                else:
                    verify_reference(arrays, triplet, first)
                selected = verify_rankings(arrays, triplet, mode)
                for index in selected:
                    support = support_mask(first, str(arrays["source_support_keys"][arrays["source_support"][index]]), masks[start], frames.shape[1:3])
                    if mode == "detected":
                        for offset_key, target, ncc_key, overlap_key in (("first_offset", start + 1, "actual_first_ncc", "actual_first_overlap"),
                                                                       ("total_offset", start + 2, "reference_ncc", "reference_overlap")):
                            ncc, overlap = pair_check(native[[start, target]], support, arrays[offset_key][index])
                            close([ncc, overlap], [arrays[ncc_key][index], arrays[overlap_key][index]], "top1 pixel recheck mismatch")
                        mid_support = support_mask(second, str(arrays["middle_support_keys"][arrays["middle_support"][index]]), masks[start + 1], frames.shape[1:3])
                        ncc, _ = pair_check(native[start + 1:start + 3], mid_support, arrays["total_offset"][index] - arrays["first_offset"][index])
                        close(ncc, arrays["cached_second_ncc"][index], "top1 cached middle pixels mismatch")
                    else:
                        ncc, overlap = common_check(native[start:start + 3], support, arrays["first_offset"][index], arrays["total_offset"][index])
                        close([*ncc, overlap], [arrays[k][index] for k in ("common_first_ncc", "common_reference_ncc", "common_second_ncc", "common_overlap")], "top1 common pixels mismatch")
                checked += len(selected); path_count += triplet["candidate_paths"]
                for name, value in sorted(arrays.items()):
                    signature.update(name.encode()); signature.update(str((value.shape, value.dtype.str)).encode()); signature.update(value.tobytes())
                print(json.dumps({"candidate_id": key, "start": start, "paths": triplet["candidate_paths"], "top1_pixel_rechecks": len(selected)}), flush=True)
            fingerprint = signature.hexdigest()
            if row["family"] == "original":
                signatures[row["base_id"]] = fingerprint
            elif row["family"] == "encoding_control":
                require(signatures.get(row["base_id"]) == fingerprint, "encoding control differs in fields or arrays")
                controls.append({"candidate_id": key, "all_fields_and_arrays_exact": True})
            stats.append({"candidate_id": key, "triplets": len(row["triplets"]), "paths_checked": path_count, "top1_pixel_rechecks": checked,
                          "declared_first_hypotheses": declared if mode == "detected" else None, "with_middle_detection": linked if mode == "detected" else None})
            all_ids.append(key)
        require(evidence_names == set(identity["evidence_sha256"]) == {p.name for p in (shard / "evidence").glob("*.npz")}, "extra/missing evidence files")
        require(runtime["triplets_completed"] == sum(len(r["triplets"]) for r in records), "runtime triplet count mismatch")
        identities.append(identity); runtimes.append(runtime)
        bindings[shard.name] = digest(shard / "provenance.json")
    require(bool(identities), "no completed shards")
    reference = identities[0]
    require(sorted(all_ids) == sorted(reference["sharding"]["selected_cohort"]) and len(set(all_ids)) == len(all_ids), "full cohort missing/duplicated")
    require(sorted(i["sharding"]["shard"] for i in identities) == list(range(reference["sharding"]["shards"])), "shard coverage mismatch")
    for identity in identities:
        for name in ("config", "input_sha256", "review_sha256", "code_files", "script_sha256", "helper_sha256", "region_cache_sha256"):
            require(identity[name] == reference[name], "inconsistent shard " + name)
    output.mkdir(parents=True)
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "mode": reference.get("mode", "detected"),
              "script_sha256": digest(Path(__file__)), "source_provenance_sha256": bindings, "runtime": runtimes, "videos": stats, "encoding_controls": controls,
              "coverage": "all path indices/geometry/completeness/rankings; all top1 group pixel evidence independently recomputed",
              "not_verified": "all path pixels, global search optimality, physical correspondence, nuisance discrimination and final score"}
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
