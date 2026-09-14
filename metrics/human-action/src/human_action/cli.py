from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from vbench_audit_core.cli import build_parser as build_core_parser
from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.errors import AuditError, InputError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.paths import output_base
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.audit import parse_action_query
from .backends.vbench import verify_upstream
from .diagnostics import diagnostics_level
from .metric import (
    environment_record,
    evaluate_backend_sharded,
    load_categories,
    release_cuda_resources,
    set_seed,
    upstream_path,
    weight_path,
)


def _to_video_result(item: dict[str, Any]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(
        video=str(item["video"]),
        status=str(item["status"]),
        score=item.get("score"),
        error=item.get("error"),
        metric={key: value for key, value in item.items() if key not in reserved},
    )


def _aggregate(results: list[dict[str, Any]]) -> tuple[float | None, int]:
    scores = [
        float(result["score"])
        for result in results
        if result.get("status") == "succeeded" and result.get("score") is not None
    ]
    return (sum(scores) / len(scores), len(scores)) if scores else (None, 0)


def _interrupted_results(
    backend: str,
    videos: list[Path],
    metadata: dict[str, dict[str, Any]],
    categories: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "video": str(video),
            "prompt": (parse_action_query(metadata[video.name], categories).prompt if video.name in metadata else ""),
            "target_action": (parse_action_query(metadata[video.name], categories).target_action if video.name in metadata else None),
            "backend": backend,
            "score": None,
            "status": "interrupted",
            "failure_reason": "interrupted",
            "error": "KeyboardInterrupt: evaluation interrupted",
            "diagnostics": None,
        }
        for video in videos
    ]


def build_parser() -> argparse.ArgumentParser:
    return build_core_parser("human-action", "VBench human action evaluator")


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        if args.audit or args.both:
            if metadata_path is None:
                raise InputError("human-action audit 需要 metadata.json，且未自动找到")
            missing = [video.name for video in videos if video.name not in metadata]
            if missing:
                raise InputError(f"metadata 缺少视频映射: {', '.join(missing)}")
        verify_upstream(upstream_path())
        categories = load_categories(upstream_path())
        for video in videos:
            if args.audit or args.both:
                try:
                    parse_action_query(metadata[video.name], categories)
                except ValueError as exc:
                    raise InputError(f"{video.name}: {exc}") from exc
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        set_seed(args.seed)
        model_weight = weight_path()
        if not model_weight.is_file():
            raise InputError(
                f"UMT 权重不存在: {model_weight}; 设置 VBENCH_AUDIT_UMT_WEIGHT 指向已校验权重"
            )
        level = diagnostics_level(len(videos))
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    output_root = output_base(args.output)
    current_run = run_id()
    started = time.monotonic()
    base_environment = environment_record(videos, metadata_path, model_weight)
    overall_code = 0

    for backend in backends:
        backend_started = time.monotonic()
        interrupted = False
        try:
            results = evaluate_backend_sharded(
                backend,
                videos,
                metadata,
                gpu_ids,
                model_weight,
                categories,
                level,
                args.seed,
            )
        except KeyboardInterrupt:
            results = _interrupted_results(backend, videos, metadata, categories)
            interrupted = True
        finally:
            release_cuda_resources()

        succeeded = sum(result["status"] == "succeeded" for result in results)
        failed = len(results) - succeeded
        status = "interrupted" if interrupted else (
            "complete" if not failed else ("partial" if succeeded else "failed")
        )
        aggregate, valid_samples = _aggregate(results)
        summary = RunSummary(
            metric="human-action",
            backend=backend,
            status=status,
            total=len(results),
            succeeded=succeeded,
            failed=failed,
            valid_samples=valid_samples,
            aggregate=aggregate,
            formula_version=(
                "vbench1-filename-target-top5-threshold-video-boolean-mean"
                if backend == "vbench"
                else "duration-weighted-temporal-mean-target-probability-benchmark-scalar"
            ),
        )
        destination = output_root / "human-action" / backend / current_run
        run_info = {
            "experiment": "human-action evaluation",
            "purpose": "official parity or explicit-target temporal evidence audit",
            "command": sys.argv,
            "parsed_args": vars(args),
            "seed": args.seed,
            "device": device_info,
            "diagnostics_level": level.value,
            "status": status,
            "backend_seconds": time.monotonic() - backend_started,
            "elapsed_seconds": time.monotonic() - started,
            "output": str(destination),
            "schedule": last_schedule(),
            **base_environment,
        }
        write_results(
            destination,
            [_to_video_result(result) for result in results],
            summary,
            run_info,
        )
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            overall_code = 1
        if interrupted:
            return 130
    return overall_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
