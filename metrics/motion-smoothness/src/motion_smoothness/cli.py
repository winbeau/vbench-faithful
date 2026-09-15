from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.cli import build_parser as build_core_parser
from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.errors import AuditError
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.paths import output_base
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.schemas import RunSummary, VideoResult

from .schemas import MotionSmoothnessConfig
from .metric import evaluate_backend_sharded


def build_parser() -> argparse.ArgumentParser:
    parser = build_core_parser("motion-smoothness", "VBench Motion Smoothness evaluator")
    parser.add_argument("--model-weight")
    parser.add_argument("--tail-quantile", type=float, default=0.90)
    parser.add_argument("--tail-weight", type=float, default=0.25)
    parser.add_argument("--magnitude-weight", type=float, default=0.7)
    parser.add_argument("--direction-weight", type=float, default=0.3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        config = MotionSmoothnessConfig(
            tail_quantile=args.tail_quantile,
            tail_weight=args.tail_weight,
            magnitude_weight=args.magnitude_weight,
            direction_weight=args.direction_weight,
        )
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    try:
        output_root = output_base(args.output)
        current_run = run_id()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    exit_code = 0
    for backend in backends:
        default_weight = "VBENCH_AUDIT_AMT_WEIGHT" if backend == "vbench" else "VBENCH_AUDIT_RAFT_WEIGHT"
        default_path = Path.home() / ".cache/vbench" / ("amt_model/amt-s.pth" if backend == "vbench" else "raft_model/models/raft-things.pth")
        weight = Path(args.model_weight or os.environ.get(default_weight, str(default_path)))
        try:
            raw_results = evaluate_backend_sharded(backend, videos, gpu_ids, weight, config, args.seed)
            interrupted = False
        except KeyboardInterrupt:
            raw_results = [{"video": str(video), "backend": backend, "score": None, "status": "interrupted", "error": "KeyboardInterrupt: evaluation interrupted", "failure_reason": "interrupted"} for video in videos]
            interrupted = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw_results = [{"video": str(video), "backend": backend, "score": None, "status": "failed", "error": error, "failure_reason": error} for video in videos]
            interrupted = False
        results = [VideoResult(video=str(item["video"]), status=str(item["status"]), score=item.get("score"), error=item.get("error"), metric={key: value for key, value in item.items() if key not in {"video", "status", "score", "error"}}) for item in raw_results]
        scores = [float(item.score) for item in results if item.score is not None and item.status == "succeeded"]
        status = "interrupted" if interrupted else ("complete" if len(scores) == len(videos) else ("partial" if scores else "failed"))
        destination = output_root / "motion-smoothness" / backend / current_run
        write_results(destination, results, RunSummary("motion-smoothness", backend, status, len(results), len(scores), len(results)-len(scores), len(scores), sum(scores) / len(scores) if scores else None, "official-amt" if backend == "vbench" else "continuity-mean-tail"), {"metric": "motion-smoothness", "backend": backend, "videos": [str(video) for video in videos], "metadata": str(metadata_path) if metadata_path else None, "device": device_info, "seed": args.seed, "schedule": last_schedule()})
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            exit_code = 1
        if interrupted:
            return 130
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
