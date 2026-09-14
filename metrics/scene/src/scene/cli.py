from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.cli import build_parser as build_core_parser
from vbench_audit_core.errors import AuditError, InputError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.paths import output_base
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.vbench import extract_scene_label, verify_upstream
from .metric import environment_record, evaluate_backend_sharded, set_seed, upstream_path
from .models import build_scorer


def build_parser() -> argparse.ArgumentParser:
    parser = build_core_parser("scene", "VBench 1.0 Scene evaluator and environment-grounded ablations")
    parser.add_argument("--audit-variant", choices=("global", "environment_grounded"), default="environment_grounded")
    parser.add_argument("--scorer", choices=("openclip",), default="openclip")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    return parser


def _mode_list(args: argparse.Namespace) -> list[str]:
    if args.both:
        return ["official", args.audit_variant]
    if args.audit:
        return [args.audit_variant]
    return ["official"]


def _to_video_result(item: dict[str, Any]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(video=str(item["video"]), status=str(item["status"]), score=item.get("score"), error=item.get("error"), metric={key: value for key, value in item.items() if key not in reserved})


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        if metadata_path is None:
            raise InputError("scene requires metadata.json")
        metadata = load_metadata(metadata_path, videos)
        missing = [video.name for video in videos if video.name not in metadata]
        if missing:
            raise InputError(f"metadata missing videos: {', '.join(missing)}")
        for video in videos:
            extract_scene_label(metadata[video.name])
        gpu_ids = parse_gpu(args.gpu)
        modes = _mode_list(args)
        device_info = check_cuda(gpu_ids)
        set_seed(args.seed)
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_root = output_base(args.output)
    current_run = run_id()
    started = time.monotonic()
    overall_code = 0
    for mode in modes:
        try:
            if mode == "official":
                verify_upstream(upstream_path())
                results = evaluate_backend_sharded(mode, videos, metadata, gpu_ids, seed=args.seed)
            else:
                results = evaluate_backend_sharded(
                    mode,
                    videos,
                    metadata,
                    gpu_ids,
                    scorer_config={"scorer": args.scorer, "model_name": args.model_name, "pretrained": args.pretrained},
                    seed=args.seed,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results = [{"video": str(video), "prompt": metadata[video.name].get("prompt", ""), "scene": extract_scene_label(metadata[video.name]), "backend": mode, "score": None, "status": "failed", "failure_reason": error, "error": error, "diagnostics": None} for video in videos]
        backend_name = "vbench" if mode == "official" else "audit"
        for item in results:
            item["backend"] = backend_name
        succeeded = sum(item.get("status") == "succeeded" for item in results)
        failed = len(results) - succeeded
        scores = [float(item["score"]) for item in results if item.get("status") == "succeeded" and item.get("score") is not None]
        status = "complete" if not failed else ("partial" if succeeded else "failed")
        destination = output_root / "scene" / backend_name / current_run
        summary = RunSummary(metric="scene", backend=backend_name, status=status, total=len(results), succeeded=succeeded, failed=failed, valid_samples=len(scores), aggregate=sum(scores) / len(scores) if scores else None, formula_version=("vbench1-tag2text-lexical-frame-rate" if mode == "official" else ("global-image-text-score" if mode == "global" else "global-times-regional-mean-v1")))
        run_info = {"experiment": "scene evaluation", "purpose": "official parity or environment-grounded ablation", "mode": mode, "variant": mode, "backend": backend_name, "command": sys.argv, "parsed_args": vars(args), "seed": args.seed, "device": device_info, "metadata": str(metadata_path), "elapsed_seconds": time.monotonic() - started, "output": str(destination), "schedule": last_schedule(), **environment_record(videos, metadata_path, args.model_name)}
        write_results(destination, [_to_video_result(item) for item in results], summary, run_info)
        print(json.dumps({"backend": backend_name, "variant": mode, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            overall_code = 1
    return overall_code


def main(argv: list[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
