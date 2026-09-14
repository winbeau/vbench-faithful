from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .devices import check_cuda, parse_gpu
from .errors import AuditError, InputError
from .inputs import enumerate_videos, find_metadata, load_metadata
from .outputs import run_id, write_results
from .paths import output_base as resolve_output_base
from .runner import run_audit_stub, run_vbench_backend
from .schemas import RunSummary, VideoResult


def build_parser(metric: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=metric, description=description)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--vbench", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--both", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--video-dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpu", nargs="?", const="0")
    parser.add_argument("--metadata")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _code_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def execute(metric: str, args: argparse.Namespace, vbench_backend=None, required_metadata: bool = False) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        if required_metadata:
            missing = [video.name for video in videos if video.name not in metadata]
            if missing:
                raise InputError(f"该维度需要元数据，但以下视频缺少映射: {', '.join(missing)}")
        gpu_ids = parse_gpu(args.gpu)
        needs_vbench = args.vbench or args.both
        device_info = check_cuda(gpu_ids) if needs_vbench else {"requested_gpu_ids": gpu_ids, "checked": False}
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    output_base = resolve_output_base(args.output)
    current_run = run_id()
    overall_code = 0
    for backend in backends:
        if backend == "audit":
            results = run_audit_stub([str(video) for video in videos])
            status = "not_implemented"
            aggregate = None
        elif vbench_backend is None:
            results = [VideoResult(video=str(video), status="not_implemented", error="官方 backend 尚未接入") for video in videos]
            status = "not_implemented"
            aggregate = None
            overall_code = 1
        else:
            results = run_vbench_backend(vbench_backend, [str(video) for video in videos])
            scores = [item.score for item in results if item.status == "succeeded" and item.score is not None]
            aggregate = sum(scores) / len(scores) if scores else None
            status = "complete" if len(scores) == len(videos) else ("partial" if scores else "failed")
            if status != "complete":
                overall_code = 1
        if backend == "audit":
            overall_code = 1
        summary = RunSummary(metric, backend, status, len(videos), sum(r.status == "succeeded" for r in results), sum(r.status == "failed" for r in results), sum(r.status == "succeeded" for r in results), aggregate, "unavailable")
        destination = output_base / metric / backend / current_run
        run_info = {
            "metric": metric, "backend": backend, "command": sys.argv,
            "seed": args.seed, "videos": [str(video) for video in videos],
            "metadata": str(metadata_path) if metadata_path else None,
            "metadata_entries": len(metadata), "code_sha": _code_sha(),
            "device": device_info, "status": status,
        }
        write_results(destination, results, summary, run_info)
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}, ensure_ascii=False))
    return overall_code
