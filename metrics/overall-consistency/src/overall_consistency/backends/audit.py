from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..conditions import normalize_conditions
from ..metric import aggregate_condition_scores, combine_global_and_condition_scores, cosine_similarity, score_conditions
from ..schemas import RepairResult


class RepairEvaluator:
    """Same ViCLIP representation plus condition-level semantic verification."""

    def __init__(self, encoder: Any, alpha: float = 0.5, lambda_: float = 0.5):
        # Validate configuration before expensive inference.
        aggregate_condition_scores([0.0], alpha)
        combine_global_and_condition_scores(0.0, 0.0, lambda_)
        self.encoder = encoder
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)

    def evaluate_video(self, video: Path, metadata_item: Mapping[str, Any]) -> RepairResult:
        prompt = metadata_item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("overall consistency requires a non-empty prompt")
        condition_set = normalize_conditions(metadata_item, prompt)

        # The video is encoded exactly once. Text features share the same
        # official ViCLIP text encoder and its native context behavior.
        video_feature = self.encoder.encode_video(video)
        text_cache: dict[str, Any] = {}

        def text_feature(text: str) -> Any:
            if text not in text_cache:
                text_cache[text] = self.encoder.encode_text(text)
            return text_cache[text]

        global_score = cosine_similarity(video_feature, text_feature(prompt))
        condition_scores = score_conditions(
            video_feature, [text_feature(condition.text) for condition in condition_set.conditions]
        )
        aggregation = aggregate_condition_scores(condition_scores, self.alpha)
        repair_score = combine_global_and_condition_scores(global_score, aggregation.score, self.lambda_)
        return RepairResult(
            video_path=str(video),
            prompt=prompt,
            global_score=global_score,
            conditions=condition_set.conditions,
            condition_scores=condition_scores,
            condition_aggregation=aggregation,
            repair_score=repair_score,
            condition_source=condition_set.source,
            warnings=condition_set.warnings,
            fallback_reason=condition_set.fallback_reason,
            alpha=self.alpha,
            lambda_=self.lambda_,
        )
