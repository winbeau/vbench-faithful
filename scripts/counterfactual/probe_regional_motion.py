"""Cache-only, all-phase regional translation diagnostics; no final Repair."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.regional_motion import RegionMotionConfig, fit_regions, region_weights
from vbench_audit_models.sam_regions import unpack_masks
from .score_static_jitter import source_hashes
from .static_jitter import digest


def frame_weights(cache, frame, grid_shape):
    prefix = f"frame_{frame}_"
    masks = unpack_masks(cache[prefix + "masks_packed"], cache[prefix + "image_shape"])
    weights = region_weights(masks, grid_shape)
    # This includes background/camera evidence; never subtract a global motion.
    return np.concatenate((weights, np.ones((1, weights.shape[1]), np.float32)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-run", "region-run", "config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        parser.error("invalid shard assignment")
    root = Path(__file__).resolve().parents[2]
    features, regions, output = Path(args.feature_run), Path(args.region_run), Path(args.output)
    f_id = json.loads((features / "provenance.json").read_text())
    r_id = json.loads((regions / "provenance.json").read_text())
    for source in (features, regions):
        runtime = json.loads((source / "runtime.json").read_text())
        if runtime["status"] != "finished" or runtime["failed"] or runtime["completed"] != runtime["expected"]:
            raise ValueError("complete, failure-free bound source runs required")
    if (r_id["feature_provenance_sha256"] != digest(features / "provenance.json")
            or r_id["feature_cache_sha256"] != f_id["cache_sha256"]
            or r_id["input_sha256"] != f_id["input_sha256"]
            or set(r_id["cache_sha256"]) != set(f_id["cache_sha256"])):
        raise ValueError("feature and region cohort identities differ")
    config = json.loads(Path(args.config).read_text())
    motion = RegionMotionConfig(**config["motion"])
    if config["facet"] != "key9" or config["lags"] != [1, 2, 3, 4]:
        raise ValueError("fixed structure facet and all-phase lag protocol required")
    if output.exists():
        raise FileExistsError("fresh output required; no implicit restart/overwrite")
    output.mkdir(parents=True)
    records = [json.loads(s) for s in (regions / "diagnostics.jsonl").read_text().splitlines()]
    if len(records) != len(r_id["cache_sha256"]) or len({r["candidate_id"] for r in records}) != len(records):
        raise ValueError("duplicate/missing region records")
    all_ids = [r["candidate_id"] for r in records]
    records = records[args.shard::args.shards]
    if not records:
        raise ValueError("empty shard")
    identity = {"status": "diagnostic_only", "formal_acceptance": "NOT EVALUATED", "config": config,
                "config_sha256": digest(Path(args.config)), "motion_config": asdict(motion),
                "feature_source": str(features), "region_source": str(regions),
                "feature_provenance_sha256": digest(features / "provenance.json"),
                "region_provenance_sha256": digest(regions / "provenance.json"),
                "feature_cache_sha256": f_id["cache_sha256"], "region_cache_sha256": r_id["cache_sha256"],
                "input_sha256": r_id["input_sha256"], "code_files": source_hashes(root),
                "script_sha256": digest(Path(__file__)), "environment": {"python": platform.python_version(),
                "numpy": np.__version__, "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
                "sharding": {"shard": args.shard, "shards": args.shards, "full_cohort": all_ids,
                             "assigned": [r["candidate_id"] for r in records]}}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(records), "completed": 0,
               "failed": 0, "pairs_completed": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    started = time.monotonic()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in records:
            key = row["candidate_id"]
            record = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, pairs=[])
            try:
                feature_path, region_path = features / "evidence" / f"{key}.npz", regions / "evidence" / f"{key}.npz"
                if digest(feature_path) != f_id["cache_sha256"][key] or digest(region_path) != r_id["cache_sha256"][key]:
                    raise ValueError("bound descriptor/region cache changed")
                with np.load(feature_path, allow_pickle=False) as f, np.load(region_path, allow_pickle=False) as m:
                    times, shape = f["timestamps"], f["input_shape"]
                    if not np.array_equal(times, m["timestamps"]) or not np.array_equal(shape, m["input_shape"]):
                        raise ValueError("descriptor/region timeline or geometry mismatch")
                    h, w = map(int, f["grid_shape"])
                    feature = f[config["facet"]].astype(np.float32).reshape(len(times), h, w, -1)
                    weights = [frame_weights(m, frame, (h, w)) for frame in range(len(times))]
                    step = np.array([shape[2] / w, shape[1] / h])
                    for lag in config["lags"]:
                        for start in range(len(times) - lag):
                            result = fit_regions(feature[start], feature[start + lag], weights[start], motion)
                            seconds = float(times[start + lag] - times[start])
                            for item in result:
                                item["whole_frame_control"] = item["region"] == len(weights[start]) - 1
                                d = item["full"]["displacement_grid"]
                                item["displacement_pixels"] = (np.array(d) * step).tolist() if d is not None else None
                                item["speed_shortside_per_second"] = float(np.linalg.norm(np.array(d) * step) / min(shape[1:3]) / seconds) if d is not None else None
                                fold = [x["displacement_grid"] for x in item["folds"]]
                                item["fold_disagreement_pixels"] = float(np.linalg.norm((np.array(fold[0]) - fold[1]) * step)) if all(x is not None for x in fold) else None
                            record["pairs"].append({"start": start, "lag": lag, "seconds": seconds, "regions": result})
                            runtime.update(pairs_completed=runtime["pairs_completed"] + 1, current_candidate=key,
                                           current_start=start, current_lag=lag)
                            (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
                            print(json.dumps({"candidate_id": key, "start": start, "lag": lag, "regions": len(result)}), flush=True)
                    record.update(sampling=row["sampling"], descriptor_step_pixels=step.tolist())
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
