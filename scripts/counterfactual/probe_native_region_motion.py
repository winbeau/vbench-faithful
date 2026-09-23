"""All-phase native RGB/SAM/SIFT motion diagnostics on the bound dev cohort."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import scipy

from dynamic_degree.native_region_motion import (
    NativeMotionConfig, fit_native_regions, sift_features, sift_correspondences, temporal_match_closure,
)
from dynamic_degree.trajectory import TrajectoryConfig, decode_video
from vbench_audit_models.sam_regions import unpack_masks
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def resolve_media(row, roots):
    candidates = [Path(row["video"]), *(Path(root) / Path(row["video"]).name for root in roots)]
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if not found:
        raise FileNotFoundError(row["video"])
    if any(digest(p) != row["sha256"] for p in found):
        raise ValueError("relocated media differs from original bound bytes")
    return found[0]


def load_masks(path, expected_sha, frames, times):
    if digest(path) != expected_sha:
        raise ValueError("SAM cache changed")
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["timestamps"], times) or not np.array_equal(z["input_shape"], frames.shape):
            raise ValueError("native RGB/SAM timestamps or image geometry differ")
        return [unpack_masks(z[f"frame_{i}_masks_packed"], z[f"frame_{i}_image_shape"])
                for i in range(len(frames))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "region-run", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--video-root", action="append", default=[])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        parser.error("invalid shard assignment")
    root = Path(__file__).resolve().parents[2]
    regions, output = Path(args.region_run), Path(args.output)
    if output.exists():
        raise FileExistsError("fresh output required; no implicit restart or overwrite")
    manifest, review = Path(args.manifest), Path(args.review)
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], review, manifest, root)
    r_id = json.loads((regions / "provenance.json").read_text())
    r_time = json.loads((regions / "runtime.json").read_text())
    if (r_time["status"] != "finished" or r_time["failed"] or r_time["completed"] != len(rows)
            or r_time["expected"] != len(rows) or r_id["manifest_sha256"] != digest(manifest)
            or r_id["review_sha256"] != digest(review)
            or r_id["input_sha256"] != {r["candidate_id"]: r["sha256"] for r in rows}
            or set(r_id["cache_sha256"]) != set(r_id["input_sha256"])):
        raise ValueError("completed SAM evidence must bind exactly the reviewed cohort")
    config = json.loads(Path(args.config).read_text())
    protocols = {"native_rgb_region_crosscheck_dev_v1": "source_only",
                 "native_rgb_region_crosscheck_dev_v2": "associated_sam"}
    if config["protocol"] not in protocols or config["lags"] != [1, 2, 3, 4]:
        raise ValueError("native RGB full-phase protocol required")
    motion = NativeMotionConfig(**config["motion"])
    if motion.target_visibility != protocols[config["protocol"]]:
        raise ValueError("protocol and visibility model differ")
    all_ids = [r["candidate_id"] for r in rows]
    rows = rows[args.shard::args.shards]
    if not rows:
        raise ValueError("empty shard")
    paths = {r["candidate_id"]: resolve_media(r, args.video_root) for r in rows}
    cv2.setNumThreads(1)
    identity = {"status": "diagnostic_only", "formal_acceptance": "NOT EVALUATED", "config": config,
                "config_sha256": digest(Path(args.config)), "motion_config": asdict(motion),
                "manifest_sha256": digest(manifest), "review_sha256": digest(review),
                "feature_provenance_sha256": None, "feature_source": "native RGB; SIFT freshly computed, no DINO",
                "region_provenance_sha256": digest(regions / "provenance.json"), "region_source": str(regions),
                "input_sha256": r_id["input_sha256"], "region_cache_sha256": r_id["cache_sha256"],
                "resolved_media": {k: str(v) for k, v in paths.items()},
                "code_files": source_hashes(root), "script_sha256": digest(Path(__file__)),
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__, "opencv": cv2.__version__, "device": "cpu",
                                "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                                "opencv_threads": cv2.getNumThreads()},
                "sharding": {"shard": args.shard, "shards": args.shards, "full_cohort": all_ids,
                             "assigned": [r["candidate_id"] for r in rows]}}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0,
               "failed": 0, "pairs_completed": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    output.mkdir(parents=True)
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    started = time.monotonic()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key = row["candidate_id"]
            record = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, pairs=[])
            try:
                decode = TrajectoryConfig(sample_fps=r_id["config"]["sample_fps"], max_side=r_id["config"]["decode_max_side"])
                frames, times, sampling = decode_video(paths[key], decode)
                if list(frames.shape) != row["decoded_shape"] or not np.array_equal(times, row["pts"]):
                    raise ValueError("development check requires all native frames and original geometry")
                masks = load_masks(regions / "evidence" / f"{key}.npz", r_id["cache_sha256"][key], frames, times)
                features = [sift_features(frame) for frame in frames]
                record.update(sampling=sampling, sift_keypoints_per_frame=[len(f["xy"]) for f in features])
                for lag in config["lags"]:
                    for start in range(len(times) - lag):
                        target = start + lag
                        matches = sift_correspondences(features[start], features[target], motion)
                        source_masks = np.concatenate((masks[start], np.ones((1, *frames.shape[1:3]), bool)))
                        result = fit_native_regions(frames[start], frames[target], source_masks, masks[target], matches, motion)
                        for item in result:
                            item["whole_frame_control"] = item["region"] == len(source_masks) - 1
                        record["pairs"].append({"start": start, "lag": lag, "seconds": float(times[target] - times[start]),
                                                "regions": result, "sift_matches": matches})
                        runtime.update(pairs_completed=runtime["pairs_completed"] + 1, current_candidate=key,
                                       current_start=start, current_lag=lag, elapsed_seconds=time.monotonic() - started)
                        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
                        print(json.dumps({"candidate_id": key, "start": start, "lag": lag,
                                          "regions": len(result), "sift_matches": len(matches)}), flush=True)
                record["sparse_temporal_closure"] = temporal_match_closure(record["pairs"])
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            runtime["completed"] += 1
            (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    runtime.update(status="finished", elapsed_seconds=time.monotonic() - started,
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    return 1 if runtime["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
