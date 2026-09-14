from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.audit import evaluate_timed_frames
from .backends.vbench import UPSTREAM_PATH
from .models import RaftFlowEstimator, decode_timed_frames
from .schemas import MotionSmoothnessConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motion-smoothness", description="VBench Motion Smoothness evaluator")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--vbench", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--both", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--video-dir")
    parser.add_argument("--output")
    parser.add_argument("--metadata")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-weight")
    parser.add_argument("--tail-quantile", type=float, default=0.90)
    parser.add_argument("--tail-weight", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    videos = enumerate_videos(args.video, args.video_dir)
    metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
    metadata = load_metadata(metadata_path, videos)
    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    for backend in backends:
        if backend == "vbench":
            from .backends.vbench import DEFAULT_CONFIG, DEFAULT_WEIGHT, evaluate_official_reference
            raw, _ = evaluate_official_reference(videos, f"cuda:{args.gpu}", DEFAULT_CONFIG, Path(args.model_weight or DEFAULT_WEIGHT))
            dataset_score, video_results = raw
            results = [VideoResult(video=str(item["video_path"]), status="succeeded", score=float(item["video_results"]), metric={"backend": "vbench", "diagnostics": {"dataset_score": dataset_score}}) for item in video_results]
            payload = {"backend": "vbench", "result": raw, "metadata": str(metadata_path) if metadata_path else None}
        else:
            try:
                import torch
                device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            except ImportError:
                device = "cpu"
            weight = Path(args.model_weight or os.environ.get("VBENCH_AUDIT_RAFT_WEIGHT", str(Path.home() / ".cache/vbench/raft_model/models/raft-things.pth")))
            estimator = RaftFlowEstimator(device, weight, UPSTREAM_PATH)
            config = MotionSmoothnessConfig(tail_quantile=args.tail_quantile, tail_weight=args.tail_weight)
            outputs = []
            for video in videos:
                frames, _ = decode_timed_frames(video)
                result = evaluate_timed_frames(video, frames, estimator, config)
                outputs.append({"video": str(video), "video_score": result.score, "overall_score": result.score, "diagnostics": result.diagnostics})
            results = [VideoResult(video=item["video"], status="succeeded", score=item["video_score"], metric={"backend": "audit", "diagnostics": item["diagnostics"]}) for item in outputs]
            payload = {"backend": "audit", "results": outputs, "metadata": str(metadata_path) if metadata_path else None}
        if args.output:
            destination = Path(args.output).expanduser() / "motion-smoothness" / backend / run_id()
            scores = [item.score for item in results if item.score is not None]
            write_results(destination, results, RunSummary("motion-smoothness", backend, "complete", len(results), len(results), 0, len(scores), sum(scores) / len(scores) if scores else None, "official-amt" if backend == "vbench" else "continuity-mean-tail"), {"metric": "motion-smoothness", "backend": backend, "videos": [str(video) for video in videos]})
            payload["output"] = str(destination)
        print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
