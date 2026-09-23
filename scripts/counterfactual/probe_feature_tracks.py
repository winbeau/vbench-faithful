"""Seed an explicit local tracker at every distinct SIFT location in a DEV frame.

This is a post-hoc mechanism pilot, not a full score or acceptance run. Queries
are independent per video, never selected by a known motion or paired clean
mask. All native frames remain input, before and after the query frame.
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

from dynamic_degree.sparse_structure import extract_features, distinct_queries
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import resolve_media, load_masks
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def prepare_queries(frame, query_frame, detector="sift", grid_size=12):
    """Image-dependent SIFT or a fixed, complete cell-center grid."""
    if detector == "sift":
        features = extract_features(frame, "sift")
        queries, groups = distinct_queries(features, query_frame)
        return queries, groups, len(features["xy"])
    if detector != "grid" or type(grid_size) is not int or grid_size < 2:
        raise ValueError("known detector and grid size >=2 required")
    height, width = frame.shape[:2]
    if grid_size > min(height, width):
        raise ValueError("grid must not exceed native image resolution")
    x = (np.arange(grid_size) + .5) * width / grid_size - .5
    y = (np.arange(grid_size) + .5) * height / grid_size - .5
    xx, yy = np.meshgrid(x, y)
    queries = np.column_stack((np.full(xx.size, query_frame), xx.ravel(), yy.ravel())).astype(np.float32)
    return queries, [[i] for i in range(len(queries))], len(queries)


def tracker_reference_identity(reference, family, root):
    if family == "cotracker2":
        return {"tracker_kind": family, "tracker_code": reference["tracker_code"],
                "tracker_weight_sha256": reference["tracker_weight_sha256"],
                "tracker_reference_role": "previous_cotracker2_execution"}
    if family != "cotracker3-offline" or reference.get("protocol") != "cotracker3-comparison-preparation-v1":
        raise ValueError("known tracker family and explicit CoTracker3 preparation identity required")
    model = reference["model"]
    relative = Path(model["source_manifest"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "configs":
        raise ValueError("tracker source identity must be a repository config")
    path = root / relative
    if digest(path) != model["source_manifest_sha256"]:
        raise ValueError("tracker source manifest changed")
    source = json.loads(path.read_text())
    if (source.get("schema") != "local-tracker-source-manifest-v1"
            or source["source_files_count"] != len(source["source_files"])
            or source["source_root"] != model["existing_source_root"]
            or model["predictor_args"] != {"v2": False, "offline": True, "window_len": 60}):
        raise ValueError("CoTracker3 source/architecture identity differs")
    return {"tracker_kind": family, "tracker_code": source["source_files"],
            "tracker_weight_sha256": model["expected_sha256"], "tracker_weight_size_bytes": model["expected_size_bytes"],
            "tracker_reference_role": "planned_local_assets_not_previous_model_inference",
            "tracker_source_manifest_sha256": model["source_manifest_sha256"]}


def make_tracker(request, root, weight, device):
    kind = request.get("tracker_kind", "cotracker2")  # preserves existing bound v2 requests
    if kind == "cotracker2":
        from vbench_audit_models.point_tracker import CoTracker2Model
        return CoTracker2Model(Path(root), Path(weight), device)
    if kind == "cotracker3-offline":
        from vbench_audit_models.cotracker3 import CoTracker3OfflineModel
        return CoTracker3OfflineModel(Path(root), Path(weight), device,
                                     expected_source=request["tracker_code"], expected_sha256=request["tracker_weight_sha256"],
                                     expected_size_bytes=request["tracker_weight_size_bytes"])
    raise ValueError("unknown tracker family; no fallback to another model")


def prepare(args):
    root = Path(__file__).resolve().parents[2]
    manifest, review = Path(args.manifest), Path(args.review)
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], review, manifest, root)
    if not set(args.candidates) <= {r["candidate_id"] for r in rows} or len(set(args.candidates)) != len(args.candidates):
        raise ValueError("distinct reviewed DEV identities required")
    reference = json.loads(Path(args.tracker_reference).read_text())
    detector, grid_size = getattr(args, "query_detector", "sift"), getattr(args, "grid_size", 12)
    request = {"status": "diagnostic_only", "score": None,
               "scope": "posthoc feature-seeded tracking pilot; not final scores or all-video coverage",
               "manifest_sha256": digest(manifest), "review_sha256": digest(review),
               "reference_provenance_sha256": digest(Path(args.tracker_reference)),
               "query_frame": args.query_frame, "detector": detector, "deduplication": "exact spatial position; all orientations retained in key_groups" if detector == "sift" else "fixed cell centers, row-major; no content selection",
               "script_sha256": digest(Path(__file__)), "code_files": source_hashes(root), "videos": []}
    request["helper_sha256"] = {"scripts/counterfactual/probe_native_region_motion.py": digest(root / "scripts/counterfactual/probe_native_region_motion.py")}
    if detector == "grid":
        request["grid_size"] = grid_size
    request.update(tracker_reference_identity(reference, args.tracker_kind, root))
    if args.tracker_kind == "cotracker3-offline":
        cohort = reference["cohort"]
        if cohort["manifest_sha256"] != digest(manifest) or cohort["review_sha256"] != digest(review):
            raise ValueError("CoTracker3 preparation binds a different reviewed cohort")
    if args.region_run:
        region_id = json.loads((Path(args.region_run) / "provenance.json").read_text())
        if (region_id["manifest_sha256"] != digest(manifest) or region_id["review_sha256"] != digest(review)
                or region_id["input_sha256"] != {r["candidate_id"]: r["sha256"] for r in rows}):
            raise ValueError("SAM evidence must bind the same reviewed cohort")
        request.update(region_provenance_sha256=digest(Path(args.region_run) / "provenance.json"),
                       region_cache_sha256=region_id["cache_sha256"], region_context=1.5,
                       region_scope="every source-frame SAM region, no manual ROI or target-mask transfer")
    cv2.setNumThreads(1)
    for row in rows:
        if row["candidate_id"] not in args.candidates:
            continue
        frames, times, sampling = decode_video(resolve_media(row, args.video_root), TrajectoryConfig(sample_fps=8, max_side=512))
        if list(frames.shape) != row["decoded_shape"] or not np.array_equal(times, row["pts"]):
            raise ValueError("all native frames and geometry required")
        if not 0 <= args.query_frame < len(frames):
            raise ValueError("query frame outside video")
        queries, groups, detected = prepare_queries(frames[args.query_frame], args.query_frame, detector, grid_size)
        request["videos"].append({k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed", "video", "sha256")})
        request["videos"][-1].update(input_shape=list(frames.shape), timestamps=times.tolist(), sampling=sampling,
                                    frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                                    queries_txy=queries.tolist(), key_groups=groups, detected_keypoints=detected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(request, f, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "queries": {r["candidate_id"]: len(r["queries_txy"]) for r in request["videos"]}}))
    return 0


def infer(args):
    import torch

    request_path, output = Path(args.request), Path(args.output)
    if output.exists():
        raise FileExistsError("fresh feature-track output required")
    request = json.loads(request_path.read_text())
    kind = request.get("tracker_kind", "cotracker2")
    if kind not in {"cotracker2", "cotracker3-offline"}:
        raise ValueError("unknown tracker family")
    if kind == "cotracker3-offline":
        local_root = Path(__file__).resolve().parents[2]
        helper = "scripts/counterfactual/probe_native_region_motion.py"
        if request.get("helper_sha256", {}).get(helper) != digest(local_root / helper):
            raise ValueError("bound native media helper changed")
    if request["detector"] not in {"sift", "grid"}:
        raise ValueError("unrecognized query preparation")
    if bool(args.region_run) != ("region_provenance_sha256" in request):
        raise ValueError("region conditioning must match the prepared request")
    if args.region_run and digest(Path(args.region_run) / "provenance.json") != request["region_provenance_sha256"]:
        raise ValueError("original same-video SAM evidence changed")
    code = {str(p.relative_to(args.tracker_root)): digest(p) for p in sorted(Path(args.tracker_root).glob("cotracker/**/*.py"))}
    if code != request["tracker_code"] or digest(Path(args.tracker_weight)) != request["tracker_weight_sha256"]:
        raise ValueError("tracker source and weight must match the explicit request")
    if kind == "cotracker3-offline" and Path(args.tracker_weight).stat().st_size != request["tracker_weight_size_bytes"]:
        raise ValueError("CoTracker3 checkpoint size differs from the request")
    cv2.setNumThreads(1); torch.set_num_threads(4); torch.manual_seed(42)
    decoded, region_masks = {}, {}
    for row in request["videos"]:
        video = resolve_media(row, args.video_root)
        frames, times, _ = decode_video(video, TrajectoryConfig(sample_fps=8, max_side=512))
        if (list(frames.shape) != row["input_shape"] or not np.array_equal(times, row["timestamps"])
                or hashlib.sha256(frames.tobytes()).hexdigest() != row["frames_sha256"]):
            raise ValueError("native pixels or time axis changed")
        queries, groups, detected = prepare_queries(frames[request["query_frame"]], request["query_frame"], request["detector"], request.get("grid_size", 12))
        if not np.array_equal(queries, row["queries_txy"]) or groups != row["key_groups"]:
            raise ValueError("query detector differs across execution environments")
        if not 0 < len(queries) <= args.max_queries:
            raise ValueError("query count outside explicit memory guard; never subsample silently")
        decoded[row["candidate_id"]] = frames, times, queries
        if args.region_run:
            cache = Path(args.region_run) / "evidence" / f'{row["candidate_id"]}.npz'
            region_masks[row["candidate_id"]] = load_masks(cache, request["region_cache_sha256"][row["candidate_id"]],
                                                           frames, times)[request["query_frame"]]
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        free, _ = torch.cuda.mem_get_info()
        if free < 40 * 1024**3:
            raise RuntimeError("at least 40 GiB free required before loading the model")
        torch.cuda.set_per_process_memory_fraction(.20)
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    runtime = {"status": "running", "pid": os.getpid(), "completed": 0, "expected": len(request["videos"]), "failed": 0,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    start = time.monotonic()
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    provenance = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                  "request_sha256": digest(request_path), "request": request, "tracker_root": str(Path(args.tracker_root).resolve()),
                  "script_sha256": digest(Path(__file__)), "code_files": source_hashes(Path(__file__).resolve().parents[2]),
                  "helper_sha256": {"scripts/counterfactual/probe_native_region_motion.py": digest(Path(__file__).resolve().parents[2] / "scripts/counterfactual/probe_native_region_motion.py")},
                  "model": ("CoTracker2 v2=True window_len=8" if kind == "cotracker2" else "CoTracker3 v2=False offline=True window_len=60")
                           + "; upstream support grid 6x6; no query batching within a model call",
                  "conditioning": "all_source_sam_regions" if args.region_run else "whole_native_frame",
                  "caution": "query-frame coordinates/visibility forced by upstream, not a reliability certificate; no independent reverse check",
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "opencv": cv2.__version__,
                                  "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                  "cache_sha256": {}}
    try:
        model = make_tracker(request, args.tracker_root, args.tracker_weight, args.device)
        if kind == "cotracker3-offline":
            provenance["verified_tracker_assets"] = model.identity
        with (output / "diagnostics.jsonl").open("x") as out:
            for row in request["videos"]:
                key = row["candidate_id"]
                frames, times, queries = decoded[key]
                result = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
                result.update(status="diagnostic_only", score=None, queries=len(queries), sampling=row["sampling"])
                try:
                    if args.region_run:
                        from dynamic_degree.region_tracks import track_regions
                        regions, prediction = track_regions(model, frames, queries, region_masks[key], request["region_context"])
                        result["regions"] = regions
                        if any(r["status"] == "failed" for r in regions):
                            result.update(status="failed", error="one or more regional tracking calls failed; partial evidence retained")
                            runtime["failed"] += 1
                    else:
                        prediction = model.track_queries(frames, queries)
                        result["model_visible_fraction"] = float(np.mean(prediction["visible"]))
                    path = output / "evidence" / f"{key}.npz"
                    np.savez_compressed(path, queries_txy=queries, timestamps=times, shape=np.array(frames.shape), **prediction)
                    provenance["cache_sha256"][key] = digest(path)
                except Exception as exc:
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    runtime["failed"] += 1
                out.write(json.dumps(result, allow_nan=False) + "\n"); out.flush()
                runtime.update(completed=runtime["completed"] + 1, current_candidate=key, elapsed_seconds=time.monotonic() - start)
                (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
                brief = {k: result[k] for k in ("candidate_id", "status", "score", "queries")}
                if "regions" in result:
                    brief["regions"] = len(result["regions"])
                    brief["region_states"] = {s: sum(r["status"] == s for r in result["regions"])
                                              for s in ("diagnostic_only", "insufficient", "failed")}
                print(json.dumps(brief), flush=True)
        runtime["status"] = "finished"
    except Exception as exc:
        runtime.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runtime.update(elapsed_seconds=time.monotonic() - start, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if args.device.startswith("cuda"):
            runtime["max_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for name in ("manifest", "review", "tracker-reference", "output"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--candidates", nargs="+", required=True)
    prep.add_argument("--query-frame", type=int, required=True)
    prep.add_argument("--tracker-kind", choices=["cotracker2", "cotracker3-offline"], default="cotracker2")
    prep.add_argument("--query-detector", choices=["sift", "grid"], default="sift")
    prep.add_argument("--grid-size", type=int, default=12, help="grid arm only; all cell centers retained")
    prep.add_argument("--video-root", action="append", default=[])
    prep.add_argument("--region-run", help="condition tracking on every source-frame SAM region")
    run = sub.add_parser("infer")
    for name in ("request", "tracker-root", "tracker-weight", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--video-root", action="append", default=[])
    run.add_argument("--region-run", help="the exact SAM cache bound during prepare")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--max-queries", type=int, default=2048)
    args = p.parse_args(argv)
    if args.action == "prepare":
        if args.query_frame < 0:
            p.error("nonnegative query frame required")
        return prepare(args)
    if args.max_queries < 1:
        p.error("positive memory query limit required")
    return infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
