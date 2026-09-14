from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from vbench_audit_core.cli import build_parser as build_core_parser
from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.errors import AuditError, InputError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.paths import output_base
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.vbench import (
    DEFAULT_WEIGHT,
    inspect_upstream,
    metadata_to_official_entries,
)
from .conditions import parse_target_objects
from .metric import aggregate_frame_totals, evaluate_backend_sharded
from .schemas import MultipleObjectsConfig


def build_parser() -> argparse.ArgumentParser:
    parser = build_core_parser("multiple-objects", "VBench Multiple Objects evaluator")
    parser.add_argument("--model-weight")
    parser.add_argument("--repair-candidate-threshold", type=float, default=0.0)
    parser.add_argument("--softmin-beta", type=float, default=10.0)
    parser.add_argument("--aggregation", choices=("softmin", "hard_min"), default="softmin")
    return parser


def _to_video_result(item: dict[str, Any]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(
        video=str(item["video"]), status=str(item["status"]), score=item.get("score"),
        error=item.get("error"),
        metric={key: value for key, value in item.items() if key not in reserved},
    )


def _aggregate(results: list[dict[str, Any]]) -> tuple[float | None, int]:
    totals: list[tuple[float, int]] = []
    for result in results:
        if result.get("status") != "succeeded":
            continue
        diagnostics = result.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("successful result lacks diagnostics")
        totals.append((float(diagnostics["frame_score_sum"]), int(diagnostics["frame_count"])))
    return aggregate_frame_totals(totals), sum(count for _, count in totals)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        if metadata_path is None:
            raise InputError("multiple-objects 需要 metadata.json，且未自动找到")
        metadata = load_metadata(metadata_path, videos)
        missing = [video.name for video in videos if video.name not in metadata]
        if missing:
            raise InputError(f"metadata 缺少视频映射: {', '.join(missing)}")
        for video in videos:
            try:
                parse_target_objects(metadata[video.name])
            except ValueError as exc:
                raise InputError(f"{video.name}: {exc}") from exc
        if args.vbench or args.both:
            metadata_to_official_entries(videos, metadata)
        state = inspect_upstream()
        upstream_state = state.__dict__ if hasattr(state, "__dict__") else state
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        config = MultipleObjectsConfig(
            repair_candidate_threshold=args.repair_candidate_threshold,
            softmin_beta=args.softmin_beta,
            aggregation_mode=args.aggregation,
        )
        configured_weight = args.model_weight or os.environ.get("VBENCH_AUDIT_GRIT_WEIGHT")
        model_weight = Path(configured_weight).expanduser() if configured_weight else DEFAULT_WEIGHT
        if not model_weight.is_file():
            raise InputError(
                f"GRiT 权重不存在: {model_weight}; 使用 --model-weight 或 "
                "VBENCH_AUDIT_GRIT_WEIGHT 指定已校验的官方权重"
            )
        _set_seed(args.seed)
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    output_root = output_base(args.output)
    current_run = run_id()
    overall_code = 0
    started = time.monotonic()
    for backend in backends:
        backend_started = time.monotonic()
        interrupted = False
        try:
            results = evaluate_backend_sharded(backend, videos, metadata, gpu_ids, model_weight, config, args.seed)
        except KeyboardInterrupt:
            interrupted = True
            results = [
                {
                    "video": str(video), "prompt": str(metadata[video.name].get("prompt", "")),
                    "backend": backend, "score": None, "status": "interrupted",
                    "error": "KeyboardInterrupt: evaluation interrupted", "diagnostics": None,
                }
                for video in videos
            ]
        succeeded = sum(result["status"] == "succeeded" for result in results)
        failed = len(results) - succeeded
        status = "interrupted" if interrupted else (
            "complete" if not failed else ("partial" if succeeded else "failed")
        )
        aggregate, frame_count = _aggregate(results)
        summary = RunSummary(
            "multiple-objects", backend, status, len(results), succeeded, failed,
            frame_count, aggregate,
            "vbench1-hard-frame-conjunction-mean" if backend == "vbench"
            else f"confidence-{config.aggregation_mode}-frame-mean-v1",
        )
        destination = output_root / "multiple-objects" / backend / current_run
        write_results(
            destination, [_to_video_result(item) for item in results], summary,
            {
                "experiment": "multiple-objects evaluation",
                "purpose": "official parity or confidence-preserving set-completeness audit",
                "command": sys.argv, "parsed_args": vars(args), "seed": args.seed,
                "device": device_info, "metadata": str(metadata_path),
                "model_weight": str(model_weight), "upstream": upstream_state,
                "status": status, "backend_seconds": time.monotonic() - backend_started,
                "elapsed_seconds": time.monotonic() - started, "output": str(destination), "schedule": last_schedule(),
            },
        )
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            overall_code = 1
        if interrupted:
            return 130
    return overall_code


def main(argv: list[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
