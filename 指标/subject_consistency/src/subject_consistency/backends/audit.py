from __future__ import annotations

from dataclasses import asdict, dataclass

from ..metric import subject_consistency_diagnostics


def score_features(features):
    return subject_consistency_diagnostics(features)


@dataclass(frozen=True)
class CounterfactualGaps:
    temporal_position_gap: float
    subject_corruption_gap: float
    background_gap: float
    scale_gap: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def counterfactual_gaps(
    reference_score: float,
    *,
    temporal_position_score: float,
    subject_corruption_score: float,
    background_score: float,
    scale_score: float,
) -> CounterfactualGaps:
    """Signed degradation gaps used only for audit diagnostics."""

    reference = float(reference_score)
    return CounterfactualGaps(
        temporal_position_gap=reference - float(temporal_position_score),
        subject_corruption_gap=reference - float(subject_corruption_score),
        background_gap=reference - float(background_score),
        scale_gap=reference - float(scale_score),
    )
