"""All-phase native line/junction identity diagnostics; no Dynamic score.

The evaluator sees RGB and timestamps only. Review/seed/source identities are
used by the runner for cohort accounting, never for correspondence selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.line_structure import line_features, reciprocal_lines, segment_junctions, junction_correspondences
from dynamic_degree.sparse_structure import identity_triangles
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import resolve_media
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest

CONFIG = {"protocol": "native_line_junction_dev_v1", "sample_fps": 8, "decode_max_side": 512,
          "lags": [1, 2, 3, 4], "detector": "OpenCV LSD_REFINE_STD defaults, all detections",
          "strip_along": 32, "strip_across": 9, "strip_half_length_factor": .75,
          "strip_half_width_factor": 3., "min_overlap": .8, "descriptor_ratio": .75,
          "junction_extension": "one measured LSD width at each endpoint", "score": None}


def inspect_video(frames, times, update=lambda _: None):
    if (frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3 or len(frames) < 3
            or np.shape(times) != (len(frames),) or not np.isfinite(times).all() or not (np.diff(times) > 0).all()):
        raise ValueError("native RGB and strictly increasing physical timestamps required")
    features = [line_features(frame) for frame in frames]
    junctions = [segment_junctions(f["lines"], f["widths"], frames.shape[1:3]) for f in features]
    result = {"status": "diagnostic_only", "score": None, "frames": [], "pairs": []}
    for feature, nodes in zip(features, junctions):
        result["frames"].append({**{k: feature[k].tolist() for k in ("lines", "widths", "precision")}, "junctions": nodes})
    for lag in CONFIG["lags"]:
        for start in range(len(frames) - lag):
            end = start + lag
            lines = reciprocal_lines(features[start], features[end], ratio=CONFIG["descriptor_ratio"])
            nodes = junction_correspondences(junctions[start], junctions[end], lines["matches"])
            result["pairs"].append({"start": start, "lag": lag, "seconds": float(times[end] - times[start]),
                                    "lines": lines, "junctions": nodes})
            update({"pairs_completed_video": len(result["pairs"]), "start": start, "lag": lag})
    for kind in ("lines", "junctions"):
        pairs = [{"start": p["start"], "lag": p["lag"], "matches": p[kind]["matches"]} for p in result["pairs"]]
        for pair, support in zip(result["pairs"], identity_triangles(pairs)):
            pair[kind]["identity_support"] = support["matches"]
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        p.error("invalid shard assignment")
    root = Path(__file__).resolve().parents[2]
    manifest, review, output = map(Path, (args.manifest, args.review, args.output))
    if output.exists():
        raise FileExistsError("fresh output required; no implicit restart")
    all_rows = select_reviewed_candidates(list(map(json.loads, manifest.read_text().splitlines())), review, manifest, root)
    if any(r["split"] != "dev" for r in all_rows):
        raise ValueError("development cohort only")
    rows = all_rows[args.shard::args.shards]
    if not rows:
        raise ValueError("empty shard")
    paths = {r["candidate_id"]: resolve_media(r, args.video_root) for r in rows}
    cv2.setNumThreads(1)
    identity = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "config": CONFIG,
                "manifest_sha256": digest(manifest), "review_sha256": digest(review),
                "input_sha256": {r["candidate_id"]: r["sha256"] for r in all_rows},
                "resolved_media": {k: str(v) for k, v in paths.items()}, "code_files": source_hashes(root),
                "script_sha256": digest(Path(__file__)),
                "helper_sha256": {"scripts/counterfactual/probe_native_region_motion.py": digest(root / "scripts/counterfactual/probe_native_region_motion.py")},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu",
                                "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "opencv_threads": cv2.getNumThreads()},
                "sharding": {"shard": args.shard, "shards": args.shards, "full_cohort": [r["candidate_id"] for r in all_rows],
                             "assigned": [r["candidate_id"] for r in rows]}}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0, "failed": 0,
               "pairs_completed": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()
    output.mkdir(parents=True)

    def update(fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    (output / "provenance.json").write_text(json.dumps(identity, indent=2)); update({})
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key = row["candidate_id"]
            record = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None)
            try:
                update({"candidate_id": key, "pairs_completed_video": 0})
                frames, times, sampling = decode_video(paths[key], TrajectoryConfig(sample_fps=CONFIG["sample_fps"], max_side=CONFIG["decode_max_side"]))
                if list(frames.shape) != row["decoded_shape"] or not np.array_equal(times, row["pts"]):
                    raise ValueError("all original frames, geometry and timestamps required")
                record.update(inspect_video(frames, times, update), sampling=sampling,
                              decoded_shape=list(frames.shape), media_duration_seconds=len(frames) / sampling["source_fps"])
                runtime["pairs_completed"] += len(record["pairs"])
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}"); runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            update({"completed": runtime["completed"] + 1})
            print(json.dumps({k: record[k] for k in ("candidate_id", "status")}), flush=True)
    update({"status": "finished", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    identity["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
