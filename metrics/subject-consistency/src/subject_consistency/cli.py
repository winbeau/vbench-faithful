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
from vbench_audit_core.paths import output_base
from vbench_audit_core.coordinator import last_schedule
from vbench_audit_core.schemas import RunSummary, VideoResult

from .backends.vbench import verify_upstream
from .metric import (
    build_dino_config,
    environment_record,
    evaluate_backend_sharded,
    evaluate_masked_sharded,
    repaired_dataset_score,
    set_seed,
    upstream_path,
)

MASKED_VARIANTS = ("subject_masked", "subject_isolated", "subject_hybrid")
AUDIT_VARIANTS = ("temporal_all_pairs", *MASKED_VARIANTS)


def build_subject_parser() -> argparse.ArgumentParser:
    parser = build_parser("subject-consistency", "VBench Subject Consistency evaluator")
    parser.add_argument("--dino-repo", default=None)
    parser.add_argument("--dino-weight", default=None)
    parser.add_argument("--audit-variant", choices=AUDIT_VARIANTS, default="temporal_all_pairs")
    parser.add_argument("--subject-masks", default=None, help="directory of frozen <stem>.npz subject masks")
    parser.add_argument("--subject-detector-checkpoint", help="subject_hybrid: pinned local COCO Mask R-CNN weight")
    parser.add_argument("--subject-mobilesam-checkpoint", help="subject_hybrid: pinned local MobileSAM vit_t weight")
    parser.add_argument("--subject-phrase-field", default="subject_en")
    parser.add_argument("--subject-max-frames", type=int, default=None)
    parser.add_argument("--subject-instance-mode", choices=("union", "mean"), default="union")
    parser.add_argument("--subject-missing-policy", choices=("zero", "exclude", "carry"), default=None,
                        help="default: exclude for subject_hybrid, zero for other variants")
    parser.add_argument("--subject-view", choices=("full", "crop"), default="crop",
                        help="subject_isolated: erase background before encoding; optionally normalize the subject crop")
    return parser


def _to_video_result(item: dict[str, object]) -> VideoResult:
    reserved = {"video", "status", "score", "error"}
    return VideoResult(
        video=str(item["video"]), status=str(item["status"]), score=item.get("score"), error=item.get("error"),
        metric={key: value for key, value in item.items() if key not in reserved},
    )


def _official_dataset_score(results: list[dict[str, object]], gpu_count: int | None = None) -> tuple[float | None, int]:
    succeeded = [item for item in results if item.get("status") == "succeeded" and item.get("score") is not None]
    if not succeeded:
        return None, 0
    transition_count = sum(max(0, int(item["diagnostics"]["num_frames"]) - 1) for item in succeeded)
    weighted = sum(float(item["score"]) * max(0, int(item["diagnostics"]["num_frames"]) - 1) for item in succeeded)
    if transition_count == 0:
        return None, 0
    return weighted / transition_count, transition_count


def execute(args: argparse.Namespace) -> int:
    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        if args.audit_variant in ("subject_masked", "subject_isolated") and not args.subject_masks:
            raise InputError(f"--audit-variant {args.audit_variant} requires --subject-masks")
        if args.audit_variant == "subject_hybrid" and (args.audit or args.both):
            for name in ("subject_detector_checkpoint", "subject_mobilesam_checkpoint"):
                value = getattr(args, name)
                if not value or not Path(value).is_file():
                    raise InputError(f"subject_hybrid requires a local --{name.replace('_', '-')}")
            if args.subject_masks:
                raise InputError("subject_hybrid creates independent current-clip masks; --subject-masks is incompatible")
            if (args.subject_view != "crop" or args.subject_instance_mode != "union"
                    or args.subject_max_frames is not None or args.subject_missing_policy not in (None, "exclude")):
                raise InputError("subject_hybrid fixes crop, union, all frames and exclude to the measured v5 method")
        if args.subject_max_frames is not None and args.subject_max_frames < 2:
            raise InputError("--subject-max-frames must be at least two")
        verify_upstream(upstream_path())
        gpu_ids = parse_gpu(args.gpu)
        device_info = check_cuda(gpu_ids)
        dino_config = build_dino_config(args.dino_repo, args.dino_weight)
        set_seed(args.seed)
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    mask_config = {
        "root": args.subject_masks,
        "phrase_field": args.subject_phrase_field,
        "max_frames": args.subject_max_frames,
        "instance_mode": args.subject_instance_mode,
        "missing_policy": args.subject_missing_policy or ("exclude" if args.audit_variant == "subject_hybrid" else "zero"),
        "encoding_mode": f"preencode_{args.subject_view}" if args.audit_variant in ("subject_isolated", "subject_hybrid") else "post_pool",
    }
    if args.audit_variant == "subject_hybrid":
        mask_config.update(localizer="hybrid_v5", detector_checkpoint=args.subject_detector_checkpoint,
                           mobilesam_checkpoint=args.subject_mobilesam_checkpoint)
    output_root = output_base(args.output)
    current_run = run_id()
    started = time.monotonic()
    base_environment = environment_record(videos, metadata_path, dino_config)
    overall_code = 0
    for backend in backends:
        if backend == "audit" and args.audit_variant in MASKED_VARIANTS:
            results = evaluate_masked_sharded(videos, metadata, gpu_ids, dino_config, mask_config, args.seed)
        else:
            results = evaluate_backend_sharded(backend, videos, metadata, gpu_ids, dino_config, args.seed)
        succeeded = sum(item["status"] == "succeeded" for item in results)
        failed = len(results) - succeeded
        status = "complete" if not failed else ("partial" if succeeded else "failed")
        if backend == "vbench":
            aggregate, valid_samples = _official_dataset_score(results)
            formula_version = "vbench1-single-process-transition-weighted-v2"
        elif args.audit_variant in MASKED_VARIANTS:
            aggregate, valid_samples = repaired_dataset_score(results)
            formula_version = (f"subject-isolated-{args.subject_view}-all-pairs-v2" if args.audit_variant == "subject_isolated"
                               else "subject-localised-all-pairs-v1")
            if args.audit_variant == "subject_hybrid":
                formula_version = "subject-hybrid-v5-isolated-crop-all-pairs-exclude"
        else:
            aggregate, valid_samples = repaired_dataset_score(results)
            formula_version = "local-adjacent-plus-global-all-pairs-v1"
        destination = output_root / "subject-consistency" / backend / current_run
        summary = RunSummary("subject-consistency", backend, status, len(results), succeeded, failed, valid_samples, aggregate, formula_version)
        run_info = {
            "experiment": "subject consistency evaluation", "purpose": "official parity or symmetric temporal aggregation",
            "backend": backend, "audit_variant": args.audit_variant if backend == "audit" else None,
            "subject_masks_root": args.subject_masks if backend == "audit" and args.audit_variant in ("subject_masked", "subject_isolated") else None,
            "subject_mask_config": mask_config if backend == "audit" and args.audit_variant in MASKED_VARIANTS else None,
            "command": sys.argv, "parsed_args": vars(args), "seed": args.seed,
            "device": device_info, "metadata": str(metadata_path) if metadata_path else None,
            "elapsed_seconds": time.monotonic() - started, "output": str(destination), "schedule": last_schedule(), **base_environment,
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
