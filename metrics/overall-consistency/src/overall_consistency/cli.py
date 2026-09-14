from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from vbench_audit_core.devices import check_cuda, parse_gpu, round_robin_shards
from vbench_audit_core.coordinator import run_spawn_coordinator
from vbench_audit_core.errors import AuditError, InputError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata, sha256_file
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.paths import output_base
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.audit import RepairEvaluator
from .backends.vbench import OfficialEvaluator, checkpoint_path, import_official_module, upstream_path, verify_upstream
from .diagnostics import repair_diagnostics
from .metric import dataset_mean
from .models import LockedViCLIPEncoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    from vbench_audit_core.cli import build_parser as build_core_parser
    parser = build_core_parser("overall-consistency", "VBench Overall Consistency official and semantic-condition repair evaluator")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    return parser


def selected_modes(args: argparse.Namespace) -> list[str]:
    if args.both:
        return ["official", "repair"]
    if args.vbench:
        return ["official"]
    if args.audit:
        return ["repair"]
    return ["official"] if args.vbench else ["repair"]


def _result(video: Path, metadata_item: Mapping[str, Any], mode: str, evaluator: Any) -> dict[str, Any]:
    prompt = str(metadata_item["prompt"])
    try:
        if mode == "official":
            score = evaluator.evaluate_video(video, metadata_item)
            diagnostics = {
                "video_path": str(video),
                "prompt": prompt,
                "official_global_score": score,
                "official_entrypoint": "vbench.overall_consistency.overall_consistency",
            }
        else:
            repaired = evaluator.evaluate_video(video, metadata_item)
            score = repaired.repair_score
            diagnostics = repair_diagnostics(repaired)
        return {"video": str(video), "prompt": prompt, "backend": mode, "score": score, "status": "succeeded", "failure_reason": None, "error": None, "diagnostics": diagnostics}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {"video": str(video), "prompt": prompt, "backend": mode, "score": None, "status": "failed", "failure_reason": error, "error": error, "diagnostics": None}


def evaluate_sharded(
    mode: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    gpu_ids: list[int],
    checkpoint: Path,
    alpha: float,
    lambda_: float,
    *,
    evaluator_factory: Callable[[str, str], Any] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if evaluator_factory is not None:
        collected: list[dict[str, Any]] = []
        for gpu_id, shard in round_robin_shards(videos, gpu_ids).items():
            evaluator = evaluator_factory(mode, f"cuda:{gpu_id}")
            collected.extend(_result(video, metadata[video.name], mode, evaluator) for video in shard)
        by_video = {item["video"]: item for item in collected}
        return [by_video[str(video)] for video in videos]
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    return run_spawn_coordinator(
        _worker, [str(video) for video in videos], gpu_ids,
        worker_args=(mode, metadata_dict, str(checkpoint), alpha, lambda_, seed),
        backend="vbench" if mode == "official" else "audit", label="overall-consistency",
    ).results


def _worker(
    result_path: str, gpu_id: int, videos: list[str], mode: str,
    metadata: dict[str, dict[str, Any]], checkpoint: str, alpha: float,
    lambda_: float, seed: int = 42,
) -> None:
    paths = [Path(video) for video in videos]
    try:
        set_seed(seed)
        import torch
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        if mode == "official":
            evaluator = OfficialEvaluator(device, Path(checkpoint))
        else:
            module, _ = import_official_module(upstream_path())
            encoder = LockedViCLIPEncoder(module, device, Path(checkpoint))
            evaluator = RepairEvaluator(encoder, alpha=alpha, lambda_=lambda_)
        results = [_result(video, metadata[video.name], mode, evaluator) for video in paths]
    except Exception as exc:
        backend = "vbench" if mode == "official" else "audit"
        error = f"{type(exc).__name__}: {exc}"
        results = _failed_results(mode, paths, metadata, error)
        for item in results:
            item["backend"] = backend
    else:
        backend = "vbench" if mode == "official" else "audit"
        for item in results:
            item["backend"] = backend
    Path(result_path).write_text(json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8")


def _to_video_result(item: dict[str, Any]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(video=str(item["video"]), status=str(item["status"]), score=item.get("score"), error=item.get("error"), metric={key: value for key, value in item.items() if key not in reserved})


def _failed_results(
    mode: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    error: str,
    status: str = "failed",
) -> list[dict[str, Any]]:
    return [
        {
            "video": str(video),
            "prompt": metadata[video.name]["prompt"],
            "backend": mode,
            "score": None,
            "status": status,
            "failure_reason": error,
            "error": error,
            "diagnostics": None,
        }
        for video in videos
    ]


def _environment(videos: list[Path], metadata_path: Path, checkpoint: Path) -> dict[str, Any]:
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_sha = None
    return {
        "input_sha256": {video.name: sha256_file(video) for video in videos},
        "metadata_sha256": sha256_file(metadata_path),
        "model_weight": str(checkpoint),
        "model_weight_sha256": sha256_file(checkpoint),
        "upstream_path": str(upstream_path()),
        "code_sha": code_sha,
    }


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        if metadata_path is None:
            raise InputError("overall-consistency requires metadata.json")
        metadata = load_metadata(metadata_path, videos)
        missing = [video.name for video in videos if video.name not in metadata]
        if missing:
            raise InputError(f"metadata missing videos: {', '.join(missing)}")
        for video in videos:
            prompt = metadata[video.name].get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise InputError(f"{video.name}: prompt must be a non-empty string")
        # Validate formula parameters before CUDA/model work.
        RepairEvaluator(type("Noop", (), {})(), alpha=args.alpha, lambda_=args.lambda_)
        verify_upstream(upstream_path())
        checkpoint = checkpoint_path()
        if not checkpoint.is_file():
            raise InputError(f"ViCLIP checkpoint not found: {checkpoint}; set VBENCH_AUDIT_VICLIP_WEIGHT")
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        set_seed(args.seed)
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_root = output_base(args.output)
    current_run = run_id()
    modes = selected_modes(args)
    environment = _environment(videos, metadata_path, checkpoint)
    started = time.monotonic()
    exit_code = 0
    for mode in modes:
        interrupted = False
        try:
            results = evaluate_sharded(mode, videos, metadata, gpu_ids, checkpoint, args.alpha, args.lambda_, seed=args.seed)
        except KeyboardInterrupt:
            results = _failed_results(
                mode, videos, metadata, "KeyboardInterrupt: evaluation interrupted", status="interrupted"
            )
            interrupted = True
        except Exception as exc:
            results = _failed_results(mode, videos, metadata, f"{type(exc).__name__}: {exc}")
        succeeded = sum(item["status"] == "succeeded" for item in results)
        failed = len(results) - succeeded
        scores = [float(item["score"]) for item in results if item["status"] == "succeeded" and item["score"] is not None]
        status = "interrupted" if interrupted else ("complete" if not failed else ("partial" if succeeded else "failed"))
        backend_name = "vbench" if mode == "official" else "audit"
        for item in results:
            item["backend"] = backend_name
        summary = RunSummary(metric="overall-consistency", backend=backend_name, status=status, total=len(results), succeeded=succeeded, failed=failed, valid_samples=len(scores), aggregate=dataset_mean(scores) if scores else None, formula_version="vbench1-viclip-global-cosine" if mode == "official" else "viclip-global-condition-mean-min-v1")
        destination = output_root / "overall-consistency" / backend_name / current_run
        run_info = {"experiment": "overall-consistency evaluation", "purpose": "official parity or semantic non-compensation repair", "mode": mode, "variant": mode, "backend": backend_name, "command": sys.argv, "parsed_args": vars(args), "alpha": args.alpha, "lambda": args.lambda_, "device": device_info, "elapsed_seconds": time.monotonic() - started, "output": str(destination), "schedule": last_schedule(), **environment}
        write_results(destination, [_to_video_result(item) for item in results], summary, run_info)
        print(json.dumps({"backend": backend_name, "variant": mode, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            exit_code = 1
        if interrupted:
            return 130
    return exit_code


def main(argv: list[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
