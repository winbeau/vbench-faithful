from __future__ import annotations

from typing import Any

from .schemas import RepairResult


def repair_diagnostics(result: RepairResult) -> dict[str, Any]:
    aggregation = result.condition_aggregation
    conditions = []
    for condition, score in zip(result.conditions, result.condition_scores):
        conditions.append(
            {
                "text": condition.text,
                "type": condition.condition_type,
                "source": condition.source,
                "score": score,
            }
        )
    weakest = conditions[aggregation.weakest_index]
    return {
        "video_path": result.video_path,
        "prompt": result.prompt,
        "official_global_score": result.global_score,
        "semantic_conditions": conditions,
        "condition_scores": list(result.condition_scores),
        "condition_mean": aggregation.mean,
        "condition_min": aggregation.minimum,
        "weakest_condition": weakest,
        "condition_score": aggregation.score,
        "repair_score": result.repair_score,
        "num_conditions": len(result.conditions),
        "condition_source": result.condition_source,
        "warnings": list(result.warnings),
        "fallback_reason": result.fallback_reason,
        "alpha": result.alpha,
        "lambda": result.lambda_,
    }
