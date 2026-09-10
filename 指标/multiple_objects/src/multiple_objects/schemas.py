from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionEvidence:
    label: str
    official_threshold_score: float | None = None
    box: tuple[float, float, float, float] | None = None
    final_instance_confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("detection label must be a non-empty string")
        if self.official_threshold_score is not None:
            score = float(self.official_threshold_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("official threshold score must be finite and in [0, 1]")
            object.__setattr__(self, "official_threshold_score", score)
        if self.box is not None:
            if len(self.box) != 4 or not all(math.isfinite(float(value)) for value in self.box):
                raise ValueError("detection box must contain four finite coordinates")
            object.__setattr__(self, "box", tuple(float(value) for value in self.box))
        if self.final_instance_confidence is not None:
            final_confidence = float(self.final_instance_confidence)
            if not math.isfinite(final_confidence) or not 0.0 <= final_confidence <= 1.0:
                raise ValueError("final instance confidence must be finite and in [0, 1]")
            object.__setattr__(self, "final_instance_confidence", final_confidence)


@dataclass(frozen=True)
class MultipleObjectsConfig:
    official_threshold: float = 0.5
    repair_candidate_threshold: float = 0.0
    softmin_beta: float = 10.0
    aggregation_mode: str = "softmin"

    def __post_init__(self) -> None:
        for name in ("official_threshold", "repair_candidate_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        if self.official_threshold != 0.5:
            raise ValueError("official_threshold is frozen to upstream value 0.5")
        beta = float(self.softmin_beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("softmin_beta must be finite and positive")
        object.__setattr__(self, "softmin_beta", beta)
        if self.aggregation_mode not in {"softmin", "hard_min"}:
            raise ValueError("aggregation_mode must be 'softmin' or 'hard_min'")


@dataclass(frozen=True)
class FrameDetections:
    """Repair detections plus labels from a separate official-threshold pass."""

    detections: tuple[DetectionEvidence, ...]
    official_labels: tuple[str, ...]


@dataclass(frozen=True)
class FrameEvidence:
    frame_index: int
    targets: tuple[str, ...]
    matched_detections: dict[str, tuple[DetectionEvidence, ...]]
    per_target_official_threshold_score: dict[str, float]
    official_frame_decision: bool
    repaired_frame_score: float
    weakest_target: str | None
    weakest_official_threshold_score: float | None
    official_labels: tuple[str, ...]
    official_decision_source: str


def frame_evidence_to_dict(evidence: FrameEvidence) -> dict[str, Any]:
    return {
        "frame_index": evidence.frame_index,
        "target_objects": list(evidence.targets),
        "matched_detections": {
            target: [
                {
                    "label": item.label,
                    "official_threshold_score": item.official_threshold_score,
                    "final_instance_confidence": item.final_instance_confidence,
                    "box": list(item.box) if item.box else None,
                }
                for item in detections
            ]
            for target, detections in evidence.matched_detections.items()
        },
        "per_target_official_threshold_score": evidence.per_target_official_threshold_score,
        "official_binary_frame_decision": evidence.official_frame_decision,
        "repaired_frame_score": evidence.repaired_frame_score,
        "weakest_target": evidence.weakest_target,
        "weakest_official_threshold_score": evidence.weakest_official_threshold_score,
        "official_detected_labels": list(evidence.official_labels),
        "official_decision_source": evidence.official_decision_source,
    }
