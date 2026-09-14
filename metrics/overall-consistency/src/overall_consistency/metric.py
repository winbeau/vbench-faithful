from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .schemas import ConditionAggregation


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def normalize_feature(feature: Sequence[float]) -> tuple[float, ...]:
    values = tuple(_finite(value, "feature value") for value in feature)
    if not values:
        raise ValueError("feature must not be empty")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("feature norm must be non-zero")
    return tuple(value / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("feature dimensions must match")
    left_normalized = normalize_feature(left)
    right_normalized = normalize_feature(right)
    return _finite(sum(a * b for a, b in zip(left_normalized, right_normalized)), "cosine similarity")


def score_conditions(video_feature: Sequence[float], condition_features: Iterable[Sequence[float]]) -> tuple[float, ...]:
    scores = tuple(cosine_similarity(video_feature, feature) for feature in condition_features)
    if not scores:
        raise ValueError("at least one condition feature is required")
    return scores


def aggregate_condition_scores(scores: Sequence[float], alpha: float = 0.5) -> ConditionAggregation:
    alpha = _finite(alpha, "alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    values = tuple(_finite(score, "condition score") for score in scores)
    if not values:
        raise ValueError("at least one condition score is required")
    all_equal = all(value == values[0] for value in values[1:])
    mean = values[0] if all_equal else sum(values) / len(values)
    weakest_index = min(range(len(values)), key=values.__getitem__)
    minimum = values[weakest_index]
    if alpha == 0.0:
        score = minimum
    elif alpha == 1.0 or mean == minimum:
        score = mean
    else:
        score = alpha * mean + (1.0 - alpha) * minimum
    return ConditionAggregation(mean, minimum, score, weakest_index)


def combine_global_and_condition_scores(global_score: float, condition_score: float, lambda_: float = 0.5) -> float:
    global_score = _finite(global_score, "global score")
    condition_score = _finite(condition_score, "condition score")
    lambda_ = _finite(lambda_, "lambda")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    if lambda_ == 0.0 or global_score == condition_score:
        return global_score
    if lambda_ == 1.0:
        return condition_score
    return _finite((1.0 - lambda_) * global_score + lambda_ * condition_score, "repair score")


def dataset_mean(scores: Sequence[float]) -> float:
    values = tuple(_finite(score, "video score") for score in scores)
    if not values:
        raise ValueError("at least one video score is required")
    return sum(values) / len(values)
