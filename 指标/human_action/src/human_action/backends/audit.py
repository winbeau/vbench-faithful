from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..models import AUDIT_MAX_WINDOWS, OFFICIAL_NUM_FRAMES, OFFICIAL_THRESHOLD, middle_sample_indices, temporal_window_indices
from ..schemas import ActionQuery, AuditActionResult, ClassEvidence, TemporalWindowEvidence


def _normalize_action(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_action_query(
    metadata_item: Mapping[str, Any],
    categories: tuple[str, ...],
) -> ActionQuery:
    prompt_value = metadata_item.get("prompt", "")
    if not isinstance(prompt_value, str):
        raise ValueError("metadata prompt must be a string")
    prompt = prompt_value.strip()
    dimension = metadata_item.get("dimension_metadata")
    target = metadata_item.get("target_action")
    if isinstance(dimension, Mapping):
        while "human_action" in dimension and isinstance(dimension["human_action"], Mapping):
            dimension = dimension["human_action"]
        target = dimension.get("target_action", target)
    source = "metadata_target_action"
    if target is None:
        normalized_prompt = _normalize_action(prompt)
        prefix = "a person is "
        target = normalized_prompt[len(prefix):] if normalized_prompt.startswith(prefix) else normalized_prompt
        source = "prompt_exact_k400"
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Human Action requires explicit target_action or an exact Kinetics-400 prompt")
    normalized = _normalize_action(target)
    category_set = set(categories)
    if normalized not in category_set:
        raise ValueError(f"target action is not an exact Kinetics-400 category: {normalized}")
    return ActionQuery(prompt=prompt, target_action=normalized, target_source=source)


def class_evidence(
    probabilities: np.ndarray,
    categories: tuple[str, ...],
    target_action: str,
) -> ClassEvidence:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (400,) or not np.isfinite(values).all():
        raise ValueError("UMT probabilities must be a finite 400-element vector")
    inverse = {action: index for index, action in enumerate(categories)}
    if target_action not in inverse:
        raise ValueError(f"unknown Kinetics-400 target: {target_action}")
    ordered = np.argsort(-values, kind="stable")
    target_index = inverse[target_action]
    rank = int(np.flatnonzero(ordered == target_index)[0]) + 1
    top = ordered[:5]
    return ClassEvidence(
        target_action=target_action,
        target_index=target_index,
        target_probability=float(values[target_index]),
        target_rank=rank,
        top5_actions=tuple(categories[int(index)] for index in top),
        top5_probabilities=tuple(float(values[index]) for index in top),
    )


def aggregate_temporal_evidence(
    windows: tuple[TemporalWindowEvidence, ...],
    threshold: float = OFFICIAL_THRESHOLD,
) -> tuple[float, float]:
    total_weight = sum(window.weight for window in windows)
    if not windows or total_weight <= 0:
        raise ValueError("temporal aggregation requires positive-weight windows")
    mean_probability = sum(
        window.classification.target_probability * window.weight for window in windows
    ) / total_weight
    coverage = sum(
        window.weight
        for window in windows
        if window.classification.target_probability >= threshold
    ) / total_weight
    return float(mean_probability), float(coverage)


class AuditHumanActionEvaluator:
    def __init__(self, classifier: Any, *, max_windows: int = AUDIT_MAX_WINDOWS):
        self.classifier = classifier
        self.max_windows = max_windows

    def evaluate_video(self, video: Path, query: ActionQuery) -> AuditActionResult:
        frames, fps = self.classifier.decode_video(video)
        full_indices = middle_sample_indices(OFFICIAL_NUM_FRAMES, len(frames))
        full = class_evidence(
            self.classifier.predict_frame_indices(frames, full_indices),
            self.classifier.categories,
            query.target_action,
        )
        window_results = []
        for window_index, (start, stop, indices) in enumerate(
            temporal_window_indices(len(frames), max_windows=self.max_windows)
        ):
            evidence = class_evidence(
                self.classifier.predict_frame_indices(frames, indices),
                self.classifier.categories,
                query.target_action,
            )
            window_results.append(
                TemporalWindowEvidence(
                    window_index=window_index,
                    source_start=start,
                    source_stop=stop,
                    sampled_source_frame_indices=indices,
                    weight=float(stop - start),
                    classification=evidence,
                )
            )
        windows = tuple(window_results)
        mean_probability, coverage = aggregate_temporal_evidence(windows)
        return AuditActionResult(
            video=str(video),
            query=query,
            status="succeeded",
            failure_reason=None,
            score=None,
            full_clip=full,
            windows=windows,
            temporal_mean_probability=mean_probability,
            temporal_coverage=coverage,
            threshold=OFFICIAL_THRESHOLD,
            threshold_source="preserved_vbench1_acceptance_threshold_uncalibrated_for_window_coverage",
            source_frame_count=len(frames),
            source_fps=fps,
            scalarization="not_calibrated_structured_evidence_only",
        )
