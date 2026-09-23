"""All-phase spatial/temporal correspondence ablation, never final scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.native_region_motion import rank_with_common_sift
from dynamic_degree.sparse_structure import (
    METHODS, extract_features, reciprocal_matches, source_keys_in_mask, identity_triangles,
)
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import load_masks, resolve_media
from .probe_regional_geometry import validate_source
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def rank_pair(source_pair, masks, target_masks, matches, temporal_support):
    keep = {m["source_key"] for m in temporal_support if m["supported_without_conflict"]}
    verified = [m for m in matches if m["source_key"] in keep]
    records = []
    for region in source_pair["regions"]:
        mask = np.ones(masks.shape[1:], bool) if region["whole_frame_control"] else masks[region["region"]]
        raw = {**region, "sift_match_source_keys": source_keys_in_mask(matches, mask)}
        closed = {**region, "sift_match_source_keys": source_keys_in_mask(verified, mask)}
        records.append({k: region[k] for k in ("region", "area_pixels", "whole_frame_control")})
        records[-1].update(
            raw_ranking=rank_with_common_sift(raw, target_masks, matches),
            temporal_ranking=rank_with_common_sift(closed, target_masks, verified),
        )
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "native-run", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = p.parse_args(argv)
    if len(set(args.methods)) != len(args.methods):
        p.error("duplicate witness methods")
    root = Path(__file__).resolve().parents[2]
    manifest, review, native, regions, output = map(Path, (
        args.manifest, args.review, args.native_run, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh output required, no implicit restart")
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], review, manifest, root)
    identity = validate_source(native)
    region_id = json.loads((regions / "provenance.json").read_text())
    keys = [r["candidate_id"] for r in rows]
    if (identity["common_identity"]["full_cohort"] != keys
            or identity["common_identity"]["region_provenance_sha256"] != digest(regions / "provenance.json")
            or region_id["manifest_sha256"] != digest(manifest) or region_id["review_sha256"] != digest(review)
            or region_id["input_sha256"] != {r["candidate_id"]: r["sha256"] for r in rows}
            or identity["common_identity"]["input_sha256"] != region_id["input_sha256"]):
        raise ValueError("bound native/SAM/review identities differ")
    paths = {r["candidate_id"]: resolve_media(r, args.video_root) for r in rows}
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    provenance = {
        "status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
        "methods": args.methods, "ratio": .75, "lags": [1, 2, 3, 4],
        "temporal_rule": "at least one exact third-frame identity and no observed contradictory third frame",
        "scope": "all phases/regions; no physical motion certificate, score or velocity filter",
        "native_provenance_sha256": digest(native / "provenance.json"),
        "native_diagnostics_sha256": identity["diagnostics_sha256"],
        "region_provenance_sha256": digest(regions / "provenance.json"),
        "manifest_sha256": digest(manifest), "review_sha256": digest(review),
        "region_cache_sha256": region_id["cache_sha256"], "input_sha256": region_id["input_sha256"],
        "resolved_media": {k: str(v) for k, v in paths.items()},
        "script_sha256": digest(Path(__file__)), "code_files": source_hashes(root),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "opencv": cv2.__version__, "device": "cpu", "opencv_threads": cv2.getNumThreads()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0,
               "failed": 0, "method_pairs_completed": 0,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def checkpoint():
        runtime["elapsed_seconds"] = time.monotonic() - started
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    checkpoint()
    with (native / "diagnostics.jsonl").open() as source, (output / "diagnostics.jsonl").open("x") as out:
        for expected in rows:
            line = source.readline()
            if not line:
                raise ValueError("native run missing an expected video")
            old = json.loads(line)
            key = expected["candidate_id"]
            if old["candidate_id"] != key:
                raise ValueError("native cohort order changed")
            record = {k: old[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, methods={})
            runtime["current_candidate"] = key
            try:
                if old["status"] != "diagnostic_only" or old["score"] is not None:
                    raise ValueError("source failure remains missing")
                frames, times, sampling = decode_video(paths[key], TrajectoryConfig(sample_fps=8, max_side=512))
                if (list(frames.shape) != expected["decoded_shape"] or not np.array_equal(times, expected["pts"])
                        or not np.array_equal(times, old["sampling"]["timestamps"])):
                    raise ValueError("all native frames and original time axis required")
                phase_keys = [(lag, start) for lag in (1, 2, 3, 4) for start in range(len(frames) - lag)]
                if [(pair["lag"], pair["start"]) for pair in old["pairs"]] != phase_keys:
                    raise ValueError("incomplete native phases")
                masks = load_masks(regions / "evidence" / f"{key}.npz", region_id["cache_sha256"][key], frames, times)
                record["sampling"] = sampling
                for method in args.methods:
                    runtime["current_method"] = method
                    features = [extract_features(frame, method) for frame in frames]
                    pairs = []
                    for lag, start in phase_keys:
                        matches = reciprocal_matches(features[start], features[start + lag])
                        pairs.append({"start": start, "lag": lag, "matches": matches,
                                      "seconds": float(times[start + lag] - times[start])})
                    if method == "sift":
                        for new, previous in zip(pairs, old["pairs"]):
                            common = [{k: m[k] for k in ("source_key", "target_key", "source_xy", "target_xy", "distance", "ratio")}
                                      for m in new["matches"]]
                            if common != previous["sift_matches"]:
                                raise ValueError("fresh SIFT does not reproduce bound native baseline")
                    support = identity_triangles(pairs)
                    for pair, triangle, previous in zip(pairs, support, old["pairs"]):
                        pair["identity_support"] = triangle["matches"]
                        pair["regions"] = rank_pair(previous, masks[pair["start"]],
                            masks[pair["start"] + pair["lag"]], pair["matches"], triangle["matches"])
                        runtime["method_pairs_completed"] += 1
                        checkpoint()
                    record["methods"][method] = {"keypoints_per_frame": [len(f["xy"]) for f in features], "pairs": pairs}
                    print(json.dumps({"candidate_id": key, "method": method, "pairs": len(pairs)}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            out.write(json.dumps(record, allow_nan=False) + "\n"); out.flush()
            runtime["completed"] += 1
            checkpoint()
        if source.readline():
            raise ValueError("unexpected additional native videos")
    runtime.update(status="finished", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    checkpoint()
    provenance["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
