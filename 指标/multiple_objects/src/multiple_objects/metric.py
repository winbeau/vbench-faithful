from __future__ import annotations

import json
import math
import multiprocessing
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import DetectionEvidence, MultipleObjectsConfig


def _finite_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_detections(detections: Iterable[Any]) -> tuple[DetectionEvidence, ...]:
    normalized: list[DetectionEvidence] = []
    for item in detections:
        if isinstance(item, DetectionEvidence):
            normalized.append(
                DetectionEvidence(
                    item.label, item.official_threshold_score, item.box,
                    item.final_instance_confidence,
                )
            )
            continue
        label: Any = None
        confidence: Any = None
        box = None
        if isinstance(item, Mapping):
            label = item.get("label", item.get("description", item.get("name")))
            confidence = item.get(
                "official_threshold_score", item.get("confidence", item.get("score"))
            )
            confidence_present = any(
                key in item for key in ("official_threshold_score", "confidence", "score")
            )
            box = item.get("box")
            final_confidence = item.get("final_instance_confidence")
        elif isinstance(item, (tuple, list)) and item:
            label = item[0]
            box = item[1] if len(item) > 1 else None
            confidence = item[2] if len(item) > 2 and not isinstance(item[2], (list, tuple, set)) else None
            final_confidence = None
        else:
            final_confidence = None
        if not isinstance(label, str) or not label.strip():
            continue
        parsed_confidence = _finite_confidence(confidence)
        if ((isinstance(item, Mapping) and confidence_present) or confidence is not None) and parsed_confidence is None:
            raise ValueError(f"invalid confidence for detection {label!r}")
        if box is not None and not isinstance(box, (tuple, list)):
            raise ValueError(f"invalid box for detection {label!r}")
        parsed_box = tuple(box) if isinstance(box, (tuple, list)) else None
        normalized.append(
            DetectionEvidence(label.strip(), parsed_confidence, parsed_box, final_confidence)
        )
    return tuple(normalized)


def softmin(values: Iterable[float], beta: float = 10.0) -> float:
    scores = np.asarray([float(value) for value in values], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("softmin requires at least one score")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("softmin beta must be finite and positive")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be in [0, 1]")
    scaled = -float(beta) * scores
    max_scaled = float(np.max(scaled))
    value = -(max_scaled + math.log(float(np.mean(np.exp(scaled - max_scaled))))) / float(beta)
    return float(np.clip(value, 0.0, 1.0))


def hard_min(values: Iterable[float]) -> float:
    scores = [float(value) for value in values]
    if not scores:
        raise ValueError("hard_min requires at least one score")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("scores must be in [0, 1]")
    return min(scores)


def frame_object_evidence(
    targets: Iterable[str], detections: Iterable[Any], config: MultipleObjectsConfig | None = None,
) -> tuple[dict[str, tuple[DetectionEvidence, ...]], dict[str, float], float, str | None, float | None]:
    cfg = config or MultipleObjectsConfig()
    target_list = tuple(targets)
    normalized = normalize_detections(detections)
    matched: dict[str, tuple[DetectionEvidence, ...]] = {}
    confidences: dict[str, float] = {}
    for target in target_list:
        # confidence is the exact GRiT ROI score used by the official threshold.
        candidates = tuple(
            item for item in normalized
            if item.label == target and item.official_threshold_score is not None
        )
        matched[target] = candidates
        confidences[target] = max(
            (float(item.official_threshold_score) for item in candidates), default=0.0
        )
    values = list(confidences.values())
    frame_score = hard_min(values) if cfg.aggregation_mode == "hard_min" else softmin(values, cfg.softmin_beta)
    weakest = min(confidences, key=confidences.get) if confidences else None
    return matched, confidences, frame_score, weakest, confidences.get(weakest) if weakest else None


def official_frame_decision(
    targets: Iterable[str],
    detections: Iterable[Any],
    threshold: float = 0.5,
    *,
    official_labels: Iterable[str] | None = None,
) -> bool:
    """Apply exact-label conjunction with upstream's strict threshold boundary."""
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("official threshold must be finite and in [0, 1]")
    if official_labels is not None:
        labels = set(official_labels)
    else:
        # This branch is a model-free proxy. Real inference supplies labels
        # from a separate pass because GRiT overwrites the pre-filter score.
        labels = {
            item.label
            for item in normalize_detections(detections)
            if item.official_threshold_score is None
            or item.official_threshold_score > threshold
        }
    return all(str(target) in labels for target in targets)


def aggregate_video(frame_scores: Iterable[float]) -> float:
    values = [float(value) for value in frame_scores]
    if not values:
        raise ValueError("video must contain at least one sampled frame")
    if not np.isfinite(values).all():
        raise ValueError("frame scores must be finite")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("frame scores must be in [0, 1]")
    return float(np.mean(values))


def aggregate_frame_totals(score_sum_and_counts: Iterable[tuple[float, int]]) -> float | None:
    """Combine worker numerators/denominators without reweighting videos."""
    numerator = 0.0
    denominator = 0
    for score_sum, frame_count in score_sum_and_counts:
        value = float(score_sum)
        if (
            not math.isfinite(value)
            or not isinstance(frame_count, int)
            or isinstance(frame_count, bool)
            or frame_count < 0
            or value < 0.0
            or value > frame_count
        ):
            raise ValueError("invalid frame aggregate")
        numerator += value
        denominator += int(frame_count)
    return numerator / denominator if denominator else None


def _failed_results(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    exc: Exception,
) -> list[dict[str, Any]]:
    error = f"{type(exc).__name__}: {exc}"
    return [
        {
            "video": str(video),
            "prompt": str(metadata[video.name].get("prompt", "")),
            "backend": backend,
            "score": None,
            "status": "failed",
            "error": error,
            "diagnostics": None,
        }
        for video in videos
    ]


def _evaluate_batch(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    config: MultipleObjectsConfig,
) -> list[dict[str, Any]]:
    if backend == "vbench":
        from .backends.vbench import evaluate_official_batch

        return evaluate_official_batch(videos, metadata, device, model_weight)
    if backend == "audit":
        from .backends.audit import evaluate_audit_batch

        return evaluate_audit_batch(videos, metadata, device, model_weight, config)
    raise ValueError(f"unknown backend: {backend}")


def _worker(
    result_path: str,
    backend: str,
    gpu_id: int,
    videos: list[str],
    metadata: dict[str, dict[str, Any]],
    model_weight: str,
    config: MultipleObjectsConfig,
) -> None:
    paths = [Path(video) for video in videos]
    try:
        import torch

        torch.cuda.set_device(gpu_id)
        results = _evaluate_batch(
            backend, paths, metadata, torch.device(f"cuda:{gpu_id}"), Path(model_weight), config
        )
    except Exception as exc:
        results = _failed_results(backend, paths, metadata, exc)
    Path(result_path).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


def evaluate_backend_sharded(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    gpu_ids: list[int],
    model_weight: Path,
    config: MultipleObjectsConfig,
) -> list[dict[str, Any]]:
    from vbench_audit_core.devices import round_robin_shards

    shards = round_robin_shards(videos, gpu_ids)
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    if len(shards) == 1:
        gpu_id, shard = next(iter(shards.items()))
        try:
            import torch

            torch.cuda.set_device(gpu_id)
            return _evaluate_batch(
                backend,
                list(shard),
                metadata_dict,
                torch.device(f"cuda:{gpu_id}"),
                model_weight,
                config,
            )
        except Exception as exc:
            return _failed_results(backend, list(shard), metadata_dict, exc)

    context = multiprocessing.get_context("spawn")
    collected: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"multiple-objects-{backend}-workers-") as worker_root:
        processes = []
        result_paths = []
        for gpu_id, shard in shards.items():
            result_path = Path(worker_root) / f"gpu-{gpu_id}.json"
            process = context.Process(
                target=_worker,
                args=(
                    str(result_path), backend, gpu_id, [str(video) for video in shard],
                    metadata_dict, str(model_weight), config,
                ),
            )
            process.start()
            processes.append(process)
            result_paths.append(result_path)
        try:
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"multiple-objects worker exited with code {process.exitcode}")
            for result_path in result_paths:
                if not result_path.is_file():
                    raise RuntimeError(f"worker did not write result: {result_path.name}")
                collected.extend(json.loads(result_path.read_text(encoding="utf-8")))
        except KeyboardInterrupt:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise

    expected = [str(video) for video in videos]
    by_video: dict[str, dict[str, Any]] = {}
    for result in collected:
        video = str(result["video"])
        if video in by_video:
            raise RuntimeError(f"duplicate worker result: {video}")
        by_video[video] = result
    missing = [video for video in expected if video not in by_video]
    if missing:
        raise RuntimeError(f"missing worker results: {missing}")
    return [by_video[video] for video in expected]
