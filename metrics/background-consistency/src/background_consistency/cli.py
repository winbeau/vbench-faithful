from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vbench_audit_core.cli import build_parser as build_core_parser
from vbench_audit_core.contracts import load_model_config, run_batch_contract, run_batch_sharded
from vbench_audit_core.devices import check_cuda, parse_gpu
from vbench_audit_core.errors import AuditError
from vbench_audit_core.inputs import enumerate_videos, find_metadata, load_metadata
from vbench_audit_core.outputs import run_id, write_results
from vbench_audit_core.paths import output_base
from vbench_audit_core.provenance import collect_provenance
from vbench_audit_core.schemas import RunSummary, VideoResult

from .metric import evaluate_batch

METRIC = "background-consistency"


def build_parser() -> argparse.ArgumentParser:
    parser = build_core_parser(METRIC, "VBench background consistency evaluator")
    parser.set_defaults(seed=0)
    parser.add_argument(
        "--audit-variant",
        choices=("diagnostic", "aggregation", "repair", "frame_official", "frame_all_pairs", "union_official", "union_all_pairs", "patch_frame_calibrated", "patch_frame_balanced"),
        default="repair",
        help="repair / patch_frame_calibrated: validated background patch pooling with gain 1.75; diagnostic: all-pairs only; union_official: historical gray-fill candidate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        model_config = load_model_config(args.model_config)
        gpu_ids = parse_gpu(args.gpu)
        # Missing asset configuration is a per-video failure with output,
        # detected before any model import or CUDA initialization.
        configured = bool(model_config.get("clip", {}).get("checkpoint"))
        device_info = check_cuda(gpu_ids) if configured else {"checked": False, "reason": "missing local CLIP configuration"}
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config = {"model": model_config, "runtime": {"audit_variant": args.audit_variant, "seed": args.seed}}
    current_run = run_id()
    code = 0
    for backend in (["vbench", "audit"] if args.both else ["vbench" if args.vbench else "audit"]):
        try:
            if configured:
                rows = run_batch_sharded(evaluate_batch, backend, videos, metadata, gpu_ids, config=config, label=METRIC)
            else:
                rows = run_batch_contract(evaluate_batch, backend, videos, metadata, config=config)
        except Exception as exc:
            rows = [VideoResult(str(v), "failed", error=f"{type(exc).__name__}: {exc}",
                                metric={"backend": backend, "variant": args.audit_variant}) for v in videos]
        good = [r for r in rows if r.status == "succeeded"]
        # Official global reduction is transition weighted on any GPU count.
        # Repair datasets use equal-video reduction, explicitly recorded.
        weights = [r.metric["num_frames"] - 1 if backend == "vbench" else 1 for r in good]
        aggregate = sum(r.score*w for r, w in zip(good, weights)) / sum(weights) if good else None
        status = "complete" if len(good) == len(rows) else ("partial" if good else "failed")
        summary = RunSummary(METRIC, backend, status, len(rows), len(good), len(rows)-len(good), len(good),
                             aggregate, "background-v1", [r.error for r in rows if r.error])
        destination = output_base(args.output) / METRIC / backend / current_run
        run = {"metric": METRIC, "backend": backend, "parsed_args": vars(args), "device": device_info,
               "status": status, "aggregation": "transition_weighted" if backend == "vbench" else "equal_video",
               "provenance": collect_provenance(videos, metadata_path, Path(args.model_config) if args.model_config else None)}
        write_results(destination, rows, summary, run)
        print(json.dumps({"backend": backend, "status": status, "output": str(destination)}))
        code = max(code, int(status != "complete"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
