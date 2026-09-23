from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
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

from .diagnostics import diagnostics_level
from .metric import (
    environment_record,
    evaluate_backend_sharded,
    metadata_for_videos,
    prompt_and_override,
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
    backend: str, videos: list[Path], metadata: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "video": str(video),
            "prompt": prompt_and_override(metadata[video.name])[0],
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
    parser = build_core_parser("dynamic-degree", "VBench Dynamic Degree evaluator")
    parser.add_argument("--audit-variant", choices=["legacy", "trajectory", "local-trajectory"], default="legacy")
    parser.add_argument("--trajectory-config", help="explicit JSON configuration for the opt-in trajectory candidate")
    parser.add_argument("--tracker-root", help="local source root containing cotracker/predictor.py")
    parser.add_argument("--tracker-weight", help="existing local CoTracker2 checkpoint; never downloaded")
    return parser


def execute(args: argparse.Namespace) -> int:
    try:
        from .backends.vbench import verify_upstream
        from .prompt_target import parse_motion_target

        videos = enumerate_videos(args.video, args.video_dir)
        trajectory_options = None
        candidate = args.audit_variant in {"trajectory", "local-trajectory"} and (args.audit or args.both)
        if candidate:
            from .trajectory import TrajectoryConfig
            if args.audit_variant == "local-trajectory":
                from .local_trajectory import LocalTrajectoryConfig as TrajectoryConfig
            if not all((args.trajectory_config, args.tracker_root, args.tracker_weight)):
                raise InputError("trajectory requires --trajectory-config, --tracker-root and --tracker-weight")
            if not (Path(args.tracker_root) / "cotracker/predictor.py").is_file() or not Path(args.tracker_weight).is_file():
                raise InputError("local tracker source or checkpoint is missing")
            trajectory_options = {"config": asdict(TrajectoryConfig.read(Path(args.trajectory_config))),
                                  "source_root": str(Path(args.tracker_root).resolve()),
                                  "checkpoint": str(Path(args.tracker_weight).resolve())}
            if args.audit_variant == "local-trajectory":
                trajectory_options["variant"] = args.audit_variant
        # Neither official Dynamic nor the trajectory candidate needs text targets.
        metadata_path = None if candidate else find_metadata(args.video, args.video_dir, args.metadata)
        loaded_metadata = load_metadata(metadata_path, videos)
        if metadata_path is not None and (args.audit or args.both):
            missing = [video.name for video in videos if video.name not in loaded_metadata]
            if missing:
                raise InputError(f"metadata 缺少视频映射: {', '.join(missing)}")
        metadata = metadata_for_videos(videos, loaded_metadata)
        if (args.audit or args.both) and not candidate:
            for video in videos:
                prompt, override = prompt_and_override(metadata[video.name])
                parse_motion_target(prompt, override)
        if not candidate or args.both:
            verify_upstream(upstream_path())
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        set_seed(args.seed)
        model_weight = Path(args.tracker_weight) if candidate and not args.both else weight_path()
        if not model_weight.is_file():
            raise InputError(
                f"RAFT Things 权重不存在: {model_weight}; 设置 VBENCH_AUDIT_RAFT_WEIGHT 指向已校验权重"
            )
        level = diagnostics_level(len(videos))
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    output_root = output_base(args.output)
    current_run = run_id()
    started = time.monotonic()
    if candidate:
        from vbench_audit_core.inputs import sha256_file
        from vbench_audit_core.provenance import collect_provenance
        import torch
        workspace = Path(__file__).resolve().parents[4]
        lock = workspace / "uv.lock"
        base_environment = {
            **collect_provenance(videos, None, Path(args.trajectory_config),
                                 upstream_status="checked_for_official" if args.both else "not_used_by_trajectory"),
            "python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
            "trajectory_options": trajectory_options,
            "tracker_weight_sha256": sha256_file(Path(args.tracker_weight)),
            "tracker_source_sha256": {str(p.relative_to(args.tracker_root)): sha256_file(p)
                                      for p in sorted(Path(args.tracker_root).glob("cotracker/**/*.py"))},
            "trajectory_config_sha256": sha256_file(Path(args.trajectory_config)),
            "metadata_used": False,
        }
        if args.both:
            base_environment["official_environment"] = environment_record(videos, None, model_weight)
    else:
        base_environment = environment_record(videos, metadata_path, model_weight)
    overall_code = 0

    for backend in backends:
        backend_started = time.monotonic()
        interrupted = False
        try:
            call_args = (backend, videos, metadata, gpu_ids, model_weight, level, args.seed)
            results = (evaluate_backend_sharded(*call_args, trajectory_options=trajectory_options)
                       if candidate and backend == "audit" else evaluate_backend_sharded(*call_args))
        except KeyboardInterrupt:
            results = _interrupted_results(backend, videos, metadata)
            interrupted = True
        finally:
            release_cuda_resources()

        succeeded = sum(result["status"] == "succeeded" for result in results)
        failed = len(results) - succeeded
        if interrupted:
            status = "interrupted"
        else:
            status = "complete" if not failed else ("partial" if succeeded else "failed")
        aggregate, valid_samples = _aggregate(results)
        formula = (
            "vbench1-mean-video-booleans"
            if backend == "vbench"
            else "structured-intensity-and-temporal-coverage-no-default-scalarization"
        )
        if candidate and backend == "audit":
            formula = "trajectory-candidate-v1-short-side-lengths-per-second"
            if args.audit_variant == "local-trajectory":
                formula = "local-trajectory-reliability-v1-observed-motion-lower-bound"
        summary = RunSummary(
            metric="dynamic-degree",
            backend=backend,
            status=status,
            total=len(results),
            succeeded=succeeded,
            failed=failed,
            valid_samples=valid_samples,
            aggregate=aggregate,
            formula_version=formula,
        )
        destination = output_root / "dynamic-degree" / backend / current_run
        run_info = {
            "experiment": "dynamic-degree evaluation",
            "purpose": "verified video-only point trajectories" if candidate and backend == "audit"
                       else "official parity or source/time/persistence-aware audit",
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
