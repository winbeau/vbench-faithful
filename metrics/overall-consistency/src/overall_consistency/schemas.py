from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticCondition:
    text: str
    condition_type: str | None = None
    source: str = "provided"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionSet:
    conditions: tuple[SemanticCondition, ...]
    source: str
    warnings: tuple[str, ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ConditionAggregation:
    mean: float
    minimum: float
    score: float
    weakest_index: int


@dataclass(frozen=True)
class RepairResult:
    video_path: str
    prompt: str
    global_score: float
    conditions: tuple[SemanticCondition, ...]
    condition_scores: tuple[float, ...]
    condition_aggregation: ConditionAggregation
    repair_score: float
    condition_source: str
    warnings: tuple[str, ...]
    fallback_reason: str | None
    alpha: float
    lambda_: float
