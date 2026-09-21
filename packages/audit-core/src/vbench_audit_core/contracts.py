"""Common input, batch, and output contracts for independent metric packages."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from .devices import check_cuda, parse_gpu
from .errors import AuditError, InputError
from .inputs import enumerate_videos, find_metadata, load_metadata
from .outputs import run_id, write_results
from .paths import output_base
from .provenance import collect_provenance
from .schemas import RunSummary, VideoResult
from .coordinator import run_spawn_coordinator, last_schedule


Metadata = Mapping[str, Mapping[str, Any]]


class BatchEvaluator(Protocol):
    """The one evaluator shape shared by all metric packages.

    ``videos`` are already enumerated by ``audit-core`` and ``metadata`` is
    keyed by basename.  A backend must return exactly one ``VideoResult`` per
    input video, in any order; ``run_batch_contract`` validates and restores
    the caller's order.
    """

    def __call__(
        self,
        backend: str,
        videos: Sequence[Path],
        metadata: Metadata,
        device: str | None,
        config: Mapping[str, Any],
    ) -> Sequence[VideoResult]: ...


def load_model_config(path: str | Path | None) -> dict[str, Any]:
    """Read an optional TOML config without importing any model package."""

    if path is None:
        return {}
    selected = Path(path).expanduser()
    try:
        raw = tomllib.loads(selected.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InputError(f"无法读取 model config: {selected}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InputError("model config 顶层必须是 TOML table")
    return dict(raw)


def _as_video_result(item: VideoResult | Mapping[str, Any]) -> VideoResult:
    if isinstance(item, VideoResult):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("batch evaluator must return VideoResult or mapping items")
    metric = item.get("metric")
    if metric is not None and not isinstance(metric, dict):
        raise TypeError("batch result metric must be a mapping")
    return VideoResult(
        video=str(item.get("video", "")),
        status=str(item.get("status", "")),
        score=item.get("score"),
        metric=dict(metric or {}),
        error=item.get("error"),
    )


def validate_batch_results(
    videos: Sequence[Path], results: Sequence[VideoResult | Mapping[str, Any]]
) -> list[VideoResult]:
    """Validate exact coverage and restore the input order for one batch."""

    expected = [str(video) for video in videos]
    normalized = [_as_video_result(item) for item in results]
    actual = [item.video for item in normalized]
    if len(actual) != len(expected) or len(set(actual)) != len(actual):
        raise ValueError("batch evaluator must return exactly one unique result per video")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"batch result coverage mismatch: missing={missing}, extra={extra}")
    for item in normalized:
        if not item.status:
            raise ValueError(f"batch result has an empty status: {item.video}")
    uids = [(item.metric or {}).get("video_uid") for item in normalized]
    present_uids = [uid for uid in uids if uid is not None]
    if len(present_uids) != len(set(present_uids)):
        raise ValueError("batch result video_uid duplicates; refusing overlapping shards or query views")
    by_video = {item.video: item for item in normalized}
    return [by_video[video] for video in expected]


def run_batch_contract(
    evaluator: BatchEvaluator,
    backend: str,
    videos: Sequence[Path],
    metadata: Metadata,
    *,
    device: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[VideoResult]:
    """Call a metric evaluator through the shared batch contract."""

    results = evaluator(backend, videos, metadata, device, dict(config or {}))
    return validate_batch_results(videos, results)


def _spawn_batch_worker(
    result_path: str,
    gpu_id: int,
    videos: list[str],
    backend: str,
    metadata: dict[str, dict[str, Any]],
    config: dict[str, Any],
    evaluator: BatchEvaluator,
) -> None:
    """Adapter from the shared evaluator to the core spawn coordinator.

    Each process receives one visible physical card and therefore uses the
    logical device name ``cuda:0``.  ``gpu_id`` remains in the coordinator
    schedule/provenance; it is not passed as a second logical CUDA index.
    """

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        selected = str(gpu_id)
    else:
        visible = [value.strip() for value in inherited.split(",") if value.strip()]
        if gpu_id >= len(visible):
            raise ValueError("worker GPU index outside inherited CUDA_VISIBLE_DEVICES")
        selected = visible[gpu_id]
    # The child has not loaded a CUDA model. Respect parent remapping/UUIDs,
    # expose exactly one physical device, and use logical cuda:0 below.
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    paths = [Path(video) for video in videos]
    results = run_batch_contract(
        evaluator,
        backend,
        paths,
        metadata,
        device="cuda:0",
        config=config,
    )
    for item in results:
        item.metric = dict(item.metric or {})
        item.metric["worker_device"] = {"parent_visible_index": gpu_id,
                                         "cuda_visible_devices": selected, "logical_device": "cuda:0"}
    Path(result_path).write_text(
        json.dumps({"results": [asdict(item) for item in results]}, ensure_ascii=False),
        encoding="utf-8",
    )


def run_batch_sharded(
    evaluator: BatchEvaluator,
    backend: str,
    videos: Sequence[Path],
    metadata: Metadata,
    gpu_ids: Sequence[int],
    *,
    config: Mapping[str, Any] | None = None,
    label: str = "metric",
) -> list[VideoResult]:
    """Run the same batch contract through the shared four-shard coordinator."""

    coordinated = run_spawn_coordinator(
        _spawn_batch_worker,
        [str(video) for video in videos],
        list(gpu_ids),
        worker_args=(backend, {key: dict(value) for key, value in metadata.items()}, dict(config or {}), evaluator),
        backend=backend,
        label=label,
    )
    return validate_batch_results(videos, coordinated.results)


def not_implemented_batch(
    metric: str,
    backend: str,
    videos: Sequence[Path],
    *,
    variant: str = "skeleton",
    reason: str | None = None,
) -> list[VideoResult]:
    """Create explicit null results for an unimplemented backend.

    This is intentionally a status, not a score.  Downstream reports can
    retain the complete input denominator while refusing to treat the rows as
    successful measurements.
    """

    message = reason or f"{metric} {backend} backend is not implemented"
    return [
        VideoResult(
            video=str(video),
            status="not_implemented",
            score=None,
            metric={
                "backend": backend,
                "variant": variant,
                "implementation_status": "not_implemented",
            },
            error=message,
        )
        for video in videos
    ]


def _summary(metric: str, backend: str, results: Sequence[VideoResult]) -> RunSummary:
    succeeded = sum(item.status == "succeeded" for item in results)
    failed = sum(item.status == "failed" for item in results)
    not_implemented = sum(item.status == "not_implemented" for item in results)
    scores = [float(item.score) for item in results if item.status == "succeeded" and item.score is not None]
    if not results or not_implemented == len(results):
        status = "not_implemented"
    elif succeeded == len(results):
        status = "complete"
    elif succeeded:
        status = "partial"
    else:
        status = "failed"
    errors = [item.error for item in results if item.error]
    return RunSummary(
        metric=metric,
        backend=backend,
        status=status,
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        valid_samples=len(scores),
        aggregate=sum(scores) / len(scores) if scores else None,
        formula_version="unimplemented" if not_implemented else "metric-defined",
        errors=errors,
        status_counts=dict(Counter(item.status for item in results)),
    )


def execute_metric(
    metric: str,
    args: Any,
    evaluator: BatchEvaluator,
    *,
    description: str | None = None,
    summarize: Callable[[str, Sequence[VideoResult]], RunSummary] | None = None,
) -> int:
    """Run a metric CLI using the shared core interfaces.

    The M1 evaluators are model-free skeletons.  ``requires_cuda`` may be set
    on a future evaluator when its implementation is ready; until then the
    command records requested GPU IDs without probing CUDA or loading weights.
    """

    try:
        videos = enumerate_videos(args.video, args.video_dir)
        metadata_path = find_metadata(args.video, args.video_dir, args.metadata)
        metadata = load_metadata(metadata_path, videos)
        model_config = load_model_config(getattr(args, "model_config", None))
        gpu_ids = parse_gpu(args.gpu)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = ["vbench", "audit"] if args.both else (["vbench"] if args.vbench else ["audit"])
    cuda_requirement = getattr(evaluator, "requires_cuda", False)
    requires_cuda = (cuda_requirement(model_config, backends) if callable(cuda_requirement)
                     else bool(cuda_requirement))
    if requires_cuda:
        try:
            device_info = check_cuda(gpu_ids)
        except AuditError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        device = f"cuda:{gpu_ids[0]}"
    else:
        device_info = {
            "requested_gpu_ids": gpu_ids,
            "checked": False,
            "reason": "backend skeleton; CUDA/model loading deferred",
        }
        device = None

    base = output_base(args.output)
    current_run = run_id()
    overall_code = 0
    runtime = {
        "audit_variant": getattr(args, "audit_variant", "diagnostic"),
        "seed": args.seed,
    }
    for backend in backends:
        config = {"model": model_config, "runtime": {**runtime,
                  "evidence_dir": str(base / metric / "evidence" / backend / current_run)}}
        try:
            if requires_cuda:
                results = run_batch_sharded(evaluator, backend, videos, metadata,
                                            gpu_ids, config=config, label=metric)
            else:
                results = run_batch_contract(evaluator, backend, videos, metadata,
                                             device=device, config=config)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results = [
                VideoResult(
                    video=str(video),
                    status="failed",
                    score=None,
                    metric={"backend": backend, "variant": runtime["audit_variant"]},
                    error=error,
                )
                for video in videos
            ]
        summary = summarize(backend, results) if summarize else _summary(metric, backend, results)
        destination = base / metric / backend / current_run
        run_info = {
            "metric": metric,
            "backend": backend,
            "command": sys.argv,
            "parsed_args": vars(args),
            "metadata": str(metadata_path) if metadata_path else None,
            "metadata_entries": len(metadata),
            "model_config": str(getattr(args, "model_config", None)) if getattr(args, "model_config", None) else None,
            "device": device_info,
            "variant": runtime["audit_variant"],
            "description": description,
            "provenance": collect_provenance(
                list(videos),
                metadata_path,
                Path(args.model_config).expanduser() if getattr(args, "model_config", None) else None,
            ),
            "status": summary.status,
            "scheduling": last_schedule() if requires_cuda else {"workers": []},
        }
        provenances = {(json.dumps((item.metric or {}).get("provenance"), sort_keys=True))
                       for item in results if (item.metric or {}).get("provenance")}
        if provenances:
            run_info["provenance"]["backends"] = [json.loads(value) for value in sorted(provenances)]
        write_results(destination, results, summary, run_info)
        print(json.dumps({"backend": backend, "status": summary.status, "output": str(destination)}, ensure_ascii=False))
        if summary.status != "complete":
            overall_code = 1
    return overall_code


__all__ = [
    "BatchEvaluator",
    "load_model_config",
    "not_implemented_batch",
    "run_batch_contract",
    "run_batch_sharded",
    "validate_batch_results",
    "execute_metric",
]
