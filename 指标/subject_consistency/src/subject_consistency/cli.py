from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from vbench_audit_core.cli import build_parser
from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.errors import AuditError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.vbench import verify_upstream
from .metric import build_dino_config, environment_record, evaluate_backend_sharded, repaired_dataset_score, set_seed, upstream_path


def build_subject_parser() -> argparse.ArgumentParser:
    parser = build_parser("subject-consistency", "VBench Subject Consistency evaluator")
    parser.add_argument("--dino-repo", default=None)
    parser.add_argument("--dino-weight", default=None)
    return parser


def _to_video_result(item: dict[str, object]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(
        video=str(item["video"]), status=str(item["status"]), score=item.get("score"), error=item.get("error"),
        metric={key: value for key, value in item.items() if key not in reserved},
    )


def _official_dataset_score(results: list[dict[str, object]], gpu_count: int) -> tuple[float | None, int]:
    succeeded = [item for item in results if item.get("status") == "succeeded" and item.get("score") is not None]
    if not succeeded:
        return None, 0
    if gpu_count > 1:
        return sum(float(item["score"]) for item in succeeded) / len(succeeded), len(succeeded)
    transition_count = sum(int(item["diagnostics"]["num_frames"]) - 1 for item in succeeded)
    weighted = sum(float(item["score"]) * (int(item["diagnostics"]["num_frames"]) - 1) for item in succeeded)
    return weighted / transition_count, transition_count


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        verify_upstream(upstream_path())
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        dino_config = build_dino_config(args.dino_repo, args.dino_weight)
        set_seed(args.seed)
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    output_base = Path(args.output).expanduser() if args.output else Path.cwd() / "output"
    current_run = run_id()
    started = time.monotonic()
    base_environment = environment_record(videos, metadata_path, dino_config)
    overall_code = 0
    for backend in backends:
        results = evaluate_backend_sharded(backend, videos, metadata, gpu_ids, dino_config)
        succeeded = sum(item["status"] == "succeeded" for item in results)
        failed = len(results) - succeeded
        status = "complete" if not failed else ("partial" if succeeded else "failed")
        if backend == "vbench":
            aggregate, valid_samples = _official_dataset_score(results, len(gpu_ids))
            formula_version = "vbench1-single-transition-weighted-or-multigpu-video-mean"
        else:
            aggregate, valid_samples = repaired_dataset_score(results)
            formula_version = "local-adjacent-plus-global-all-pairs-v1"
        destination = output_base / "subject-consistency" / backend / current_run
        summary = RunSummary("subject-consistency", backend, status, len(results), succeeded, failed, valid_samples, aggregate, formula_version)
        run_info = {
            "experiment": "subject consistency evaluation", "purpose": "official parity or symmetric temporal aggregation",
            "backend": backend, "command": sys.argv, "parsed_args": vars(args), "seed": args.seed,
            "device": device_info, "metadata": str(metadata_path) if metadata_path else None,
            "elapsed_seconds": time.monotonic() - started, "output": str(destination), **base_environment,
        }
        write_results(destination, [_to_video_result(item) for item in results], summary, run_info)
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}, ensure_ascii=False))
        if status != "complete":
            overall_code = 1
    return overall_code


def main(argv: list[str] | None = None) -> int:
    return execute(build_subject_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
