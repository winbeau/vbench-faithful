from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActionQuery:
    prompt: str
    target_action: str
    target_source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ClassEvidence:
    target_action: str
    target_index: int
    target_probability: float
    target_rank: int
    top5_actions: tuple[str, ...]
    top5_probabilities: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_action": self.target_action,
            "target_index": self.target_index,
            "target_probability": self.target_probability,
            "target_rank": self.target_rank,
            "top5_actions": list(self.top5_actions),
            "top5_probabilities": list(self.top5_probabilities),
        }


@dataclass(frozen=True)
class TemporalWindowEvidence:
    window_index: int
    source_start: int
    source_stop: int
    sampled_source_frame_indices: tuple[int, ...]
    weight: float
    classification: ClassEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "source_start": self.source_start,
            "source_stop": self.source_stop,
            "sampled_source_frame_indices": list(self.sampled_source_frame_indices),
            "weight": self.weight,
            "classification": self.classification.to_dict(),
        }


@dataclass(frozen=True)
class OfficialActionResult:
    video: str
    target_action: str
    target_source: str
    sampled_frame_count: int
    rounded_top5_actions: tuple[str, ...]
    rounded_top5_probabilities: tuple[float, ...]
    accepted_actions: tuple[str, ...]
    threshold: float
    matched: bool


@dataclass(frozen=True)
class AuditActionResult:
    video: str
    query: ActionQuery
    status: str
    failure_reason: str | None
    score: float | None
    full_clip: ClassEvidence
    windows: tuple[TemporalWindowEvidence, ...]
    temporal_mean_probability: float
    temporal_coverage: float
    threshold: float
    threshold_source: str
    source_frame_count: int
    source_fps: float | None
    scalarization: str

    def task_evidence(self) -> dict[str, Any]:
        return {
            "target_action": self.query.target_action,
            "full_clip_target_probability": self.full_clip.target_probability,
            "temporal_mean_target_probability": self.temporal_mean_probability,
            "temporal_coverage": self.temporal_coverage,
            "evidence_components": [
                "full_clip_target_probability",
                "temporal_mean_target_probability",
                "temporal_coverage",
            ],
            "scalarization": self.scalarization,
        }
