"""All automatic points/regions in declared DEV frame pairs: local RGB search.

No final score. Point selection is not based on motion or known correctness;
identical source supports are only deduplicated for computation, not dropped.
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

from dynamic_degree.local_appearance import feature_support, relocate_support, appearance_ambiguity
from dynamic_degree.sparse_structure import extract_features
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import resolve_media, load_masks
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def feature_groups(features):
    # Only duplicate orientations with identical position AND size are merged.
    # Distinct feature scales stay explicit even at an identical position.
    groups = {}
    for key, (xy, scale) in enumerate(zip(features["xy"], features["scale"])):
        identifier = (*map(float, xy), float(scale))
        groups.setdefault(identifier, []).append(key)
    return [{"xy": list(k[:2]), "size": k[2], "source_keys": v} for k, v in groups.items()]


def probe_pair(source, target, masks, features, update=lambda _: None, *, search="legacy", factors=(1, 2, 4)):
    if search not in {"legacy", "ambiguity"}:
        raise ValueError("unknown local appearance search")
    if not factors or len(set(factors)) != len(factors) or not set(factors) <= {1, 2, 4}:
        raise ValueError("distinct declared support factors from 1/2/4 required")
    points = feature_groups(features)
    masks = np.concatenate((masks, np.ones((1, *source.shape[:2]), bool)))
    supports = {}
    for index, point in enumerate(points):
        x, y = np.rint(point["xy"]).astype(int)
        if not (0 <= x < source.shape[1] and 0 <= y < source.shape[0]):
            raise ValueError("detected source location outside native geometry")
        point["regional_supports"] = []
        for region in np.flatnonzero(masks[:, y, x]):
            for factor in factors:
                support = feature_support(masks[region], point["xy"], point["size"], factor)
                key = hashlib.sha256(np.packbits(support).tobytes()).hexdigest()
                if key not in supports:
                    supports[key] = (relocate_support(source, target, support) if search == "legacy"
                                     else appearance_ambiguity(source, target, support))
                point["regional_supports"].append({"region": int(region), "whole_frame_control": bool(region == len(masks) - 1),
                                                   "factor": factor, "support_sha256": key})
        update({"points_completed": index + 1, "points_expected": len(points), "unique_supports": len(supports)})
    return {"source_points": points, "supports": supports, "regions": len(masks) - 1,
            "whole_frame_control_region": len(masks) - 1}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--starts", nargs="+", type=int, required=True)
    p.add_argument("--lags", nargs="+", type=int, default=[1])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--video-root", action="append", default=[])
    p.add_argument("--search", choices=["legacy", "ambiguity"], default="legacy")
    p.add_argument("--factors", nargs="+", type=int, choices=[1, 2, 4], default=[1, 2, 4])
    args = p.parse_args(argv)
    if (len(set(args.candidates)) != len(args.candidates) or len(set(args.starts)) != len(args.starts)
            or len(set(args.lags)) != len(args.lags) or min(args.starts) < 0 or min(args.lags) < 1
            or len(set(args.factors)) != len(args.factors) or not 0 <= args.shard < args.shards):
        p.error("distinct candidates/phases and valid shard required")
    root = Path(__file__).resolve().parents[2]
    manifest, review, regions, output = map(Path, (args.manifest, args.review, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh output required; do not overwrite/restart a probe")
    cohort = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], review, manifest, root)
    by_id = {r["candidate_id"]: r for r in cohort}
    if not set(args.candidates) <= set(by_id):
        raise ValueError("requested candidates outside reviewed DEV cohort")
    identity = json.loads((regions / "provenance.json").read_text())
    state = json.loads((regions / "runtime.json").read_text())
    if (identity["manifest_sha256"] != digest(manifest) or identity["review_sha256"] != digest(review)
            or identity["input_sha256"] != {r["candidate_id"]: r["sha256"] for r in cohort}
            or set(identity["cache_sha256"]) != set(identity["input_sha256"])
            or state["status"] != "finished" or state["failed"] or state["completed"] != len(cohort)
            or state["expected"] != len(cohort)):
        raise ValueError("complete SAM evidence must bind exactly the reviewed native videos")
    selected = [r for r in cohort if r["candidate_id"] in args.candidates]
    assigned = selected[args.shard::args.shards]
    if not assigned:
        raise ValueError("empty shard")
    paths = {r["candidate_id"]: resolve_media(r, args.video_root) for r in assigned}
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    provenance = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                  "scope": "only the declared native frame pairs; not full-video scores or an invariance pass",
                  "manifest_sha256": digest(manifest), "review_sha256": digest(review),
                  "region_provenance_sha256": digest(regions / "provenance.json"), "input_sha256": identity["input_sha256"],
                  "region_cache_sha256": identity["cache_sha256"], "starts": args.starts, "lags": args.lags,
                  "support_factors": args.factors, "hypotheses": 3 if args.search == "legacy" else 12, "min_visible_overlap": .8,
                  "search": args.search,
                  "reverse_check": ("all three forward hypotheses; all three independently searched reverse hypotheses retained"
                                    if args.search == "legacy" else "NOT RUN for distinct peaks; legacy results not transferred"),
                  "self_intersection_fractions": None if args.search == "legacy" else [.25, .5, .75],
                  "sharding": {"shard": args.shard, "shards": args.shards, "selected_cohort": [r["candidate_id"] for r in selected],
                               "assigned": [r["candidate_id"] for r in assigned]},
                  "script_sha256": digest(Path(__file__)), "code_files": source_hashes(root),
                  "helper_sha256": {"scripts/counterfactual/probe_native_region_motion.py":
                                    digest(root / "scripts/counterfactual/probe_native_region_motion.py")},
                  "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                                  "device": "cpu", "opencv_threads": cv2.getNumThreads()},
                  "resolved_media": {k: str(v) for k, v in paths.items()}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(assigned), "completed": 0,
               "failed": 0, "pairs_completed": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def update(fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    update({})
    with (output / "diagnostics.jsonl").open("x") as out:
        for row in assigned:
            key = row["candidate_id"]
            record = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, pairs=[])
            try:
                frames, times, sampling = decode_video(paths[key], TrajectoryConfig(sample_fps=8, max_side=512))
                if list(frames.shape) != row["decoded_shape"] or not np.array_equal(times, row["pts"]):
                    raise ValueError("native RGB geometry/time axis changed")
                masks = load_masks(regions / "evidence" / f"{key}.npz", identity["cache_sha256"][key], frames, times)
                record["sampling"] = sampling
                for start in args.starts:
                    features = extract_features(frames[start], "sift")
                    for lag in args.lags:
                        if start + lag >= len(frames):
                            raise ValueError("requested frame pair beyond video; do not silently shorten the probe")
                        update({"candidate_id": key, "start": start, "lag": lag, "points_completed": 0})
                        pair = probe_pair(frames[start], frames[start + lag], masks[start], features, update,
                                          search=args.search, factors=args.factors)
                        pair.update(start=start, lag=lag, seconds=float(times[start + lag] - times[start]))
                        record["pairs"].append(pair)
                        update({"pairs_completed": runtime["pairs_completed"] + 1})
                        print(json.dumps({"candidate_id": key, "start": start, "lag": lag,
                                          "points": len(pair["source_points"]), "unique_supports": len(pair["supports"])}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            out.write(json.dumps(record, allow_nan=False) + "\n"); out.flush()
            update({"completed": runtime["completed"] + 1})
    update({"status": "finished", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    provenance["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
