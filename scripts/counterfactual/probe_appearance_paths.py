"""All-phase two-hop correspondence paths on one completed appearance shard.

The source's full hypothesis set is retained. Start-template checks supplement
both adjacent steps; two-step net displacement is never used as a motion score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.appearance_paths import direct_ncc, two_hop_candidates, rank_reference_paths, reference_template_paths
from dynamic_degree.local_appearance import feature_support
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import resolve_media, load_masks
from .score_static_jitter import source_hashes
from .static_jitter import digest


def source_supports(frames, masks, first):
    all_masks = np.concatenate((masks, np.ones((1, *frames.shape[1:3]), bool)))
    geometries = {}
    for point in first["source_points"]:
        for ref in point["regional_supports"]:
            key = ref["support_sha256"]
            if key in geometries:
                continue
            support = feature_support(all_masks[ref["region"]], point["xy"], point["size"], ref["factor"])
            if hashlib.sha256(np.packbits(support).tobytes()).hexdigest() != key:
                raise ValueError("source support geometry differs from bound appearance search")
            geometries[key] = support
    if set(geometries) != set(first["supports"]):
        raise ValueError("unreferenced or missing source support")
    return geometries


def inspect_reference_triplet(frames, masks, first, second, update=lambda _: None):
    if len(frames) != 3 or second["start"] != first["start"] + 1 or first["lag"] != 1 or second["lag"] != 1:
        raise ValueError("three consecutive native frames and adjacent source pairs required")
    geometries = source_supports(frames, masks, first)
    keys = sorted(geometries)
    rankings, lists, offset = [], {}, 0
    for support_id, key in enumerate(keys):
        result, arrays = reference_template_paths(frames, geometries[key], first["supports"][key]["hypotheses"])
        count = result["candidate_paths"]
        result.update(source_support=support_id, path_start=offset)
        result["ranked_path_indices"] = [offset + i for i in result["ranked_path_indices"]]
        rankings.append(result)
        arrays["source_support"] = np.full(count, support_id, int)
        for name, value in arrays.items():
            lists.setdefault(name, []).append(value)
        offset += count
        update({"support_groups_completed": support_id + 1, "support_groups_expected": len(keys), "candidate_paths": offset})
    arrays = {name: np.concatenate(values) for name, values in lists.items()}
    # Empty automatic-query sets are explicit missing evidence, never static.
    if not lists:
        arrays = {name: np.empty(0, int) for name in ("source_support", "first_rank", "reference_rank")}
        arrays.update({name: np.empty((0, 2), int) for name in ("first_offset", "total_offset")})
        arrays.update({name: np.empty(0) for name in ("common_first_ncc", "common_reference_ncc", "common_second_ncc", "common_overlap", "bottleneck_ncc")})
    arrays["source_support_keys"] = np.asarray(keys, dtype="U64")
    result = {"status": "diagnostic_only", "score": None, "start": first["start"], "frames": [first["start"] + i for i in range(3)],
              "source_queries": len(first["source_points"]), "unique_supports": len(keys), "candidate_paths": offset,
              "observable_reference_paths": int(np.isfinite(arrays["bottleneck_ncc"]).sum()), "rankings": rankings,
              "warning": "unchanged templates avoid re-detection loss, but correlated/repeated appearances are not identity certificates"}
    return result, arrays


def inspect_triplet(frames, masks, first, second, update=lambda _: None):
    if len(frames) != 3 or second["start"] != first["start"] + 1 or first["lag"] != 1 or second["lag"] != 1:
        raise ValueError("three consecutive native frames and adjacent source pairs required")
    arrays, associations = two_hop_candidates(first, second)
    keys = arrays["source_support_keys"].tolist()
    geometries = source_supports(frames, masks, first)
    count = len(arrays["source_point"])
    for name in ("actual_first_ncc", "actual_first_overlap", "reference_ncc", "reference_overlap"):
        arrays[name] = np.full(count, np.nan)
    used = np.unique(arrays["source_support"])
    for done, support_id in enumerate(used, 1):
        selected = np.flatnonzero(arrays["source_support"] == support_id)
        support = geometries[keys[support_id]]
        for offset_name, frame, ncc_name, overlap_name in (
                ("first_offset", frames[1], "actual_first_ncc", "actual_first_overlap"),
                ("total_offset", frames[2], "reference_ncc", "reference_overlap")):
            offsets, inverse = np.unique(arrays[offset_name][selected], axis=0, return_inverse=True)
            values, overlap = direct_ncc(frames[0], frame, support, offsets)
            arrays[ncc_name][selected] = values[inverse]
            arrays[overlap_name][selected] = overlap[inverse]
        update({"support_groups_completed": done, "support_groups_expected": len(used), "candidate_paths": count})
    rankings, merit = rank_reference_paths(arrays)
    arrays["bottleneck_ncc"] = merit
    result = {"status": "diagnostic_only", "score": None, "start": first["start"], "frames": [first["start"] + i for i in range(3)],
              "source_queries": len(first["source_points"]), "middle_queries": len(second["source_points"]),
              "candidate_paths": count, "observable_reference_paths": int(np.isfinite(merit).sum()),
              "associations": associations, "rankings": rankings,
              "warning": "linked detections and reference appearance are not physical identity; returning paths are not zero motion"}
    return result, arrays


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-shard", "source-execution-root", "manifest", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    p.add_argument("--mode", choices=["detected", "reference"], default="detected")
    args = p.parse_args(argv)
    source, execution, manifest, regions, output = map(Path, (args.source_shard, args.source_execution_root, args.manifest, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh path shard required; no implicit resume")
    identity = json.loads((source / "provenance.json").read_text())
    state = json.loads((source / "runtime.json").read_text())
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    ledger = {r["candidate_id"]: r for r in map(json.loads, manifest.read_text().splitlines())}
    if (identity.get("search") != "ambiguity" or identity["lags"] != [1]
            or state["status"] != "finished" or state["failed"] or state["completed"] != state["expected"]
            or len(rows) != state["completed"] or [r["candidate_id"] for r in rows] != identity["sharding"]["assigned"]
            or digest(source / "diagnostics.jsonl") != identity["diagnostics_sha256"]
            or digest(manifest) != identity["manifest_sha256"]
            or digest(regions / "provenance.json") != identity["region_provenance_sha256"]):
        raise ValueError("complete, bound adjacent-frame appearance and SAM evidence required")
    for name, sha in {**identity["code_files"], **identity["helper_sha256"],
                      "scripts/counterfactual/probe_local_appearance.py": identity["script_sha256"]}.items():
        if digest(execution / name) != sha:
            raise ValueError("source execution snapshot changed")
    paths = {r["candidate_id"]: resolve_media(ledger[r["candidate_id"]], args.video_root) for r in rows}
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    root = Path(__file__).resolve().parents[2]
    provenance = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                  "source_provenance_sha256": digest(source / "provenance.json"), "source_diagnostics_sha256": identity["diagnostics_sha256"],
                  "manifest_sha256": digest(manifest), "review_sha256": identity["review_sha256"],
                  "region_provenance_sha256": identity["region_provenance_sha256"], "region_cache_sha256": identity["region_cache_sha256"],
                  "input_sha256": identity["input_sha256"], "sharding": identity["sharding"],
                  "mode": args.mode,
                  "config": {"association_radius": "max(1px, .5*sqrt(source_size*middle_size))", "min_visible_overlap": .8,
                             "hypotheses": "all source/middle cached peaks", "ranking": "min(actual_first_NCC, cached_second_NCC, reference_NCC)",
                             "returning_paths": "both adjacent offsets kept; no net-displacement score", "support_duplicates": "deduplicated within queried location; original region references retained in source"},
                  "script_sha256": digest(Path(__file__)), "code_files": source_hashes(root),
                  "helper_sha256": {"scripts/counterfactual/probe_native_region_motion.py": digest(root / "scripts/counterfactual/probe_native_region_motion.py")},
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu"},
                  "evidence_sha256": {}}
    if args.mode == "reference":
        provenance["config"] = {"min_visible_overlap": .8, "hypotheses": "all 12 cached first peaks x 12 globally searched reference peaks",
                                "support": "identical source-index RGB samples, common visibility in all three frames",
                                "ranking": "minimum of all three common-support pairwise NCC values",
                                "returning_paths": "both adjacent offsets kept; no net-displacement score",
                                "re_detection": "not required; no source support replacement", "support_duplicates": "unique masks cached; source query references retained in parent"}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0, "failed": 0,
               "triplets_completed": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def update(fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    update({})
    with (output / "diagnostics.jsonl").open("x") as handle:
        for original in rows:
            key = original["candidate_id"]
            record = {k: original[k] for k in ("candidate_id", "base_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, triplets=[])
            try:
                frames, times, sampling = decode_video(paths[key], TrajectoryConfig(sample_fps=8, max_side=512))
                if (original["status"] != "diagnostic_only" or original["score"] is not None
                        or identity["input_sha256"][key] != ledger[key]["sha256"] or list(frames.shape) != ledger[key]["decoded_shape"]
                        or not np.array_equal(times, ledger[key]["pts"]) or not np.array_equal(times, original["sampling"]["timestamps"])
                        or identity["starts"] != list(range(len(frames) - 1))
                        or [(p["start"], p["lag"]) for p in original["pairs"]] != [(i, 1) for i in range(len(frames) - 1)]):
                    raise ValueError("all native phases and original video geometry required")
                masks = load_masks(regions / "evidence" / f"{key}.npz", identity["region_cache_sha256"][key], frames, times)
                record["sampling"] = sampling
                for start in range(len(frames) - 2):
                    update({"candidate_id": key, "start": start, "support_groups_completed": 0})
                    inspect = inspect_triplet if args.mode == "detected" else inspect_reference_triplet
                    result, arrays = inspect(frames[start:start + 3], masks[start], original["pairs"][start], original["pairs"][start + 1], update)
                    arrays["timestamps"] = times[start:start + 3]
                    path = output / "evidence" / f"{key}_start{start:02d}.npz"
                    np.savez_compressed(path, **arrays)
                    provenance["evidence_sha256"][path.name] = digest(path)
                    record["triplets"].append(result)
                    update({"triplets_completed": runtime["triplets_completed"] + 1})
                    print(json.dumps({"candidate_id": key, "start": start, "candidate_paths": result["candidate_paths"],
                                      "observable_reference_paths": result["observable_reference_paths"]}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            update({"completed": runtime["completed"] + 1})
    update({"status": "finished", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    provenance["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
