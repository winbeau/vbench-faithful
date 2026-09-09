from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..diagnostics import FrameDiagnostic


VIEW_BOXES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("upper-left", (0.0, 0.0, 0.5, 0.5)),
    ("upper-right", (0.5, 0.0, 1.0, 0.5)),
    ("lower-left", (0.0, 0.5, 0.5, 1.0)),
    ("lower-right", (0.5, 0.5, 1.0, 1.0)),
)


def environment_views(image: Any) -> list[tuple[str, Any]]:
    """Return deterministic non-overlapping spatial views, excluding the global view."""
    if not hasattr(image, "crop") or not hasattr(image, "size"):
        raise TypeError("environment_views expects a PIL-like image with crop() and size")
    width, height = image.size
    views = []
    for name, (left, top, right, bottom) in VIEW_BOXES:
        box = (round(left * width), round(top * height), round(right * width), round(bottom * height))
        views.append((name, image.crop(box)))
    return views


def bounded_score(value: float) -> float:
    import math

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("scene evidence score must be finite")
    return max(0.0, min(1.0, value))


def aggregate_frame_components(global_score: float, regional_scores: Iterable[float]) -> tuple[float, float]:
    regions = [bounded_score(value) for value in regional_scores]
    if not regions:
        raise ValueError("at least one regional score is required")
    global_value = bounded_score(global_score)
    environment_mean = sum(regions) / len(regions)
    return environment_mean, global_value * environment_mean


def aggregate_frame_score(global_score: float, regional_scores: Iterable[float]) -> float:
    """Spatially distributed support: frame_score = s_g * mean(s_1 ... s_K)."""
    return aggregate_frame_components(global_score, regional_scores)[1]


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    global_score: float
    regional_scores: list[float]
    environment_mean: float
    final_frame_score: float


def score_frame(image: Any, scene_label: str, scorer: Any, *, mode: str, frame_index: int) -> FrameEvidence:
    global_score = bounded_score(scorer.score(image, scene_label))
    if mode == "global":
        return FrameEvidence(frame_index, global_score, [], 1.0, global_score)
    if mode != "environment_grounded":
        raise ValueError(f"unsupported repaired mode: {mode}")
    regional_scores = [bounded_score(scorer.score(view, scene_label)) for _, view in environment_views(image)]
    environment_mean, final_score = aggregate_frame_components(global_score, regional_scores)
    return FrameEvidence(frame_index, global_score, regional_scores, environment_mean, final_score)


def to_diagnostic(evidence: FrameEvidence, scene_label: str, mode: str) -> FrameDiagnostic:
    return FrameDiagnostic(
        frame_index=evidence.frame_index,
        global_score=evidence.global_score,
        regional_scores=evidence.regional_scores,
        environment_mean=evidence.environment_mean,
        final_frame_score=evidence.final_frame_score,
        scene_label=scene_label,
        mode=mode,
    )
