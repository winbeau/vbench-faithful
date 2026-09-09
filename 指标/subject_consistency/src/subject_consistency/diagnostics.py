from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SubjectConsistencyDiagnostics:
    num_frames: int
    local_score: float
    global_score: float
    final_score: float
    local_pair_scores: tuple[float, ...]
    global_pair_min: float
    global_pair_mean: float
    global_pair_max: float

    def to_dict(self, *, include_local_pairs: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_local_pairs:
            payload.pop("local_pair_scores")
        return payload


@dataclass(frozen=True)
class OfficialDiagnostics:
    num_frames: int
    local_score: float
    global_score: float
    final_score: float
    aggregation: str = "0.5 * previous_mean + 0.5 * fixed_first_anchor_mean"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
