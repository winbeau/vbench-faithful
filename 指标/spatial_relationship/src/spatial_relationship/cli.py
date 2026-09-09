from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from vbench_audit_core.cli import build_parser
from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.errors import AuditError, InputError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.vbench import verify_upstream
from .diagnostics import diagnostics_level
from .metric import environment_record, evaluate_backend_sharded, parse_query, set_seed, upstream_path, weight_path


def _to_video_result(item: dict[str, object]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(
        video=str(item["video"]),
        status=str(item["status"]),
        score=item.get("score"),
        error=item.get("error"),
        metric={key: value for key, value in item.items() if key not in reserved},
    )


def _aggregate(results: list[dict[str, object]]) -> tuple[float | None, int]:
    frame_scores: list[float] = []
    for result in results:
        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("frame_scores"), list):
            frame_scores.extend(float(score) for score in diagnostics["frame_scores"])
        elif isinstance(diagnostics, dict) and isinstance(diagnostics.get("frame_results"), list):
            frame_scores.extend(float(score) for score in diagnostics["frame_results"])
    if frame_scores:
        return sum(frame_scores) / len(frame_scores), len(frame_scores)
    scores = [float(item["score"]) for item in results if item.get("score") is not None and item.get("status") == "succeeded"]
    return (sum(scores) / len(scores), len(scores)) if scores else (None, 0)


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        if metadata_path is None:
            raise InputError("spatial-relationship 需要 metadata.json，且未自动找到")
        metadata = load_metadata(metadata_path, videos)
        missing = [video.name for video in videos if video.name not in metadata]
        if missing:
            raise InputError(f"metadata 缺少视频映射: {', '.join(missing)}")
        for video in videos:
            try:
                parse_query(metadata[video.name])
            except ValueError as exc:
                raise InputError(f"{video.name}: {exc}") from exc
        verify_upstream(upstream_path())
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        set_seed(args.seed)
        model_weight = weight_path()
        if not model_weight.is_file():
            raise InputError(
                f"GRiT 权重不存在: {model_weight}; 设置 VBENCH_AUDIT_GRIT_WEIGHT 指向已校验的官方权重"
            )
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    output_base = Path(args.output).expanduser() if args.output else Path.cwd() / "output"
    current_run = run_id()
    level = diagnostics_level(len(videos))
    overall_code = 0
    started = time.monotonic()
    base_environment = environment_record(videos, metadata_path, model_weight)
    for backend in backends:
        backend_started = time.monotonic()
        try:
            results = evaluate_backend_sharded(backend, videos, metadata, gpu_ids, model_weight, level)
        except KeyboardInterrupt:
            results = []
            for video in videos:
                query = parse_query(metadata[video.name])
                results.append(
                    {
                        "video": str(video), "prompt": str(metadata[video.name].get("prompt", "")),
                        "subject": query.subject, "relation": query.relation, "object": query.object,
                        "backend": backend, "score": None, "status": "interrupted",
                        "failure_reason": "interrupted", "error": "KeyboardInterrupt: evaluation interrupted",
                        "diagnostics": None,
                    }
                )
            destination = output_base / "spatial-relationship" / backend / current_run
            summary = RunSummary(
                metric="spatial-relationship", backend=backend, status="interrupted",
                total=len(results), succeeded=0, failed=len(results), valid_samples=0,
                aggregate=None, formula_version="vbench-1.0-frame-mean" if backend == "vbench" else "ordered-role-identity-first-v1",
            )
            run_info = {
                "experiment": "spatial-relationship evaluation",
                "purpose": "official parity or ordered-role audit",
                "command": sys.argv,
                "parsed_args": vars(args),
                "seed": args.seed,
                "device": device_info,
                "diagnostics_level": level.value,
                "status": "interrupted",
                "backend_seconds": time.monotonic() - backend_started,
                "elapsed_seconds": time.monotonic() - started,
                "output": str(destination),
                **base_environment,
            }
            write_results(destination, [_to_video_result(item) for item in results], summary, run_info)
            print(json.dumps({"backend": backend, "status": "interrupted", "output": str(destination)}, ensure_ascii=False))
            return 130
        succeeded = sum(item["status"] == "succeeded" for item in results)
        failed = len(results) - succeeded
        status = "complete" if not failed else ("partial" if succeeded else "failed")
        aggregate, valid_samples = _aggregate(results)
        summary = RunSummary(
            metric="spatial-relationship", backend=backend, status=status,
            total=len(results), succeeded=succeeded, failed=failed,
            valid_samples=valid_samples, aggregate=aggregate,
            formula_version="vbench-1.0-frame-mean" if backend == "vbench" else "ordered-role-identity-first-v1",
        )
        destination = output_base / "spatial-relationship" / backend / current_run
        run_info = {
            "experiment": "spatial-relationship evaluation",
            "purpose": "official parity or ordered-role audit",
            "command": sys.argv,
            "parsed_args": vars(args),
            "seed": args.seed,
            "device": device_info,
            "diagnostics_level": level.value,
            "backend_seconds": time.monotonic() - backend_started,
            "elapsed_seconds": time.monotonic() - started,
            "output": str(destination),
            **base_environment,
        }
        write_results(destination, [_to_video_result(item) for item in results], summary, run_info)
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            overall_code = 1
    return overall_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("spatial-relationship", "VBench Spatial Relationship evaluator")
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
