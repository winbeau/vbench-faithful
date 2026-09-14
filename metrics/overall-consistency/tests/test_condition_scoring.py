import json
import tempfile
import unittest
from pathlib import Path

from overall_consistency.backends.audit import RepairEvaluator
from overall_consistency.conditions import normalize_conditions
from overall_consistency.metric import score_conditions
from overall_consistency.diagnostics import repair_diagnostics
from vbench_audit_core.inputs import load_metadata


class CountingEncoder:
    def __init__(self):
        self.video_calls = 0
        self.text_calls = []

    def encode_video(self, video):
        self.video_calls += 1
        return [1.0, 0.0]

    def encode_text(self, text):
        self.text_calls.append(text)
        return {
            "full prompt": [1.0, 0.0],
            "subject": [0.0, 1.0],
            "action": [1.0, 1.0],
            "object": [-1.0, 0.0],
            "scene": [0.5, 0.5],
        }[text]


class ConditionScoringTests(unittest.TestCase):
    def test_one_and_multiple_condition_cosines(self):
        self.assertEqual(score_conditions([1, 0], [[1, 0]]), (1.0,))
        scores = score_conditions([1, 0], [[1, 0], [0, 1], [-1, 0]])
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], -1.0)

    def test_k_one_and_k_four_produce_one_score_per_condition(self):
        self.assertEqual(len(score_conditions([1, 0], [[1, 0]])), 1)
        self.assertEqual(len(score_conditions([1, 0], [[1, 0], [0, 1], [-1, 0], [1, 1]])), 4)

    def test_explicit_conditions_preserve_binding_and_deduplicate_in_order(self):
        metadata = {"semantic_conditions": ["a woman", "a woman playing guitar", "a woman", ""]}
        condition_set = normalize_conditions(metadata, "full prompt")
        self.assertEqual([item.text for item in condition_set.conditions], ["a woman", "a woman playing guitar"])
        self.assertEqual(condition_set.source, "provided")

    def test_condition_dict_type_is_diagnostic_only(self):
        condition_set = normalize_conditions({"semantic_conditions": [{"text": "a cat left of a dog", "type": "relation"}]}, "full")
        self.assertEqual(condition_set.conditions[0].text, "a cat left of a dog")
        self.assertEqual(condition_set.conditions[0].condition_type, "relation")

    def test_top_level_metadata_precedes_dimension_metadata(self):
        condition_set = normalize_conditions(
            {
                "semantic_conditions": ["top-level"],
                "dimension_metadata": {"semantic_conditions": ["nested"]},
            },
            "full",
        )
        self.assertEqual([item.text for item in condition_set.conditions], ["top-level"])

    def test_dimension_metadata_is_used_when_top_level_is_absent(self):
        condition_set = normalize_conditions(
            {"dimension_metadata": {"semantic_conditions": [{"text": "nested", "type": "subject"}]}},
            "full",
        )
        self.assertEqual(condition_set.conditions[0].text, "nested")
        self.assertEqual(condition_set.conditions[0].condition_type, "subject")

    def test_repair_encodes_video_once_and_reuses_duplicate_full_prompt_text(self):
        encoder = CountingEncoder()
        result = RepairEvaluator(encoder).evaluate_video(
            Path("/tmp/video.mp4"),
            {"prompt": "full prompt", "semantic_conditions": ["full prompt", "subject", "action"]},
        )
        self.assertEqual(encoder.video_calls, 1)
        self.assertEqual(encoder.text_calls.count("full prompt"), 1)
        self.assertEqual(len(result.condition_scores), 3)

    def test_four_conditions_produce_four_scores_and_one_repair_score(self):
        result = RepairEvaluator(CountingEncoder()).evaluate_video(
            Path("/tmp/video.mp4"),
            {"prompt": "full prompt", "semantic_conditions": ["subject", "action", "object", "scene"]},
        )
        self.assertEqual(len(result.condition_scores), 4)
        self.assertIsInstance(result.repair_score, float)

    def test_json_metadata_propagates_conditions_to_scoring_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video.mp4"
            video.touch()
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "video": video.name,
                                "prompt": "full prompt",
                                "dimension_metadata": {
                                    "semantic_conditions": [
                                        {"text": "subject", "type": "subject"},
                                        {"text": "action", "type": "action"},
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            metadata = load_metadata(metadata_path, [video])[video.name]
            payload = repair_diagnostics(RepairEvaluator(CountingEncoder()).evaluate_video(video, metadata))
        self.assertEqual([item["text"] for item in payload["semantic_conditions"]], ["subject", "action"])
        self.assertEqual([item["type"] for item in payload["semantic_conditions"]], ["subject", "action"])

    def test_missing_zero_and_malformed_conditions_fall_back_with_warning(self):
        missing = normalize_conditions({}, "full prompt")
        empty = normalize_conditions({"semantic_conditions": []}, "full prompt")
        malformed = normalize_conditions({"semantic_conditions": {"text": "bad container"}}, "full prompt")
        self.assertEqual(missing.conditions[0].text, "full prompt")
        self.assertEqual(empty.conditions[0].text, "full prompt")
        self.assertEqual(malformed.conditions[0].text, "full prompt")
        self.assertFalse(missing.warnings)
        self.assertTrue(empty.warnings)
        self.assertTrue(malformed.warnings)
        self.assertEqual(missing.fallback_reason, "semantic_conditions_missing")
        self.assertEqual(empty.fallback_reason, "semantic_conditions_empty_or_no_usable_text")
        self.assertEqual(malformed.fallback_reason, "semantic_conditions_malformed")

    def test_malformed_entry_falls_back_instead_of_partially_scoring(self):
        condition_set = normalize_conditions(
            {"semantic_conditions": ["valid", {"text": 7}]}, "full prompt"
        )
        self.assertEqual([item.text for item in condition_set.conditions], ["full prompt"])
        self.assertEqual(condition_set.source, "fallback")
        self.assertTrue(condition_set.warnings)

    def test_full_prompt_single_condition_reduces_exactly_to_global(self):
        encoder = CountingEncoder()
        for alpha in (0.0, 0.1, 0.5, 1.0):
            for lambda_ in (0.0, 0.1, 0.5, 1.0):
                with self.subTest(alpha=alpha, lambda_=lambda_):
                    result = RepairEvaluator(encoder, alpha=alpha, lambda_=lambda_).evaluate_video(
                        Path("/tmp/video.mp4"),
                        {"prompt": "full prompt", "semantic_conditions": ["full prompt"]},
                    )
                    self.assertEqual(result.condition_scores[0], result.global_score)
                    self.assertEqual(result.condition_aggregation.score, result.global_score)
                    self.assertEqual(result.repair_score, result.global_score)

    def test_fallback_preserves_exact_official_prompt_text(self):
        condition_set = normalize_conditions({}, "  full prompt  ")
        self.assertEqual(condition_set.conditions[0].text, "  full prompt  ")

    def test_fallback_reason_is_serialized_from_actual_evaluation(self):
        payload = repair_diagnostics(
            RepairEvaluator(CountingEncoder()).evaluate_video(
                Path("/tmp/video.mp4"), {"prompt": "full prompt", "semantic_conditions": [{"text": 7}]}
            )
        )
        self.assertEqual(payload["condition_source"], "fallback")
        self.assertEqual(payload["fallback_reason"], "semantic_conditions_malformed")
        self.assertTrue(payload["warnings"])

    def test_repair_diagnostics_expose_required_evidence(self):
        result = RepairEvaluator(CountingEncoder()).evaluate_video(
            Path("/tmp/video.mp4"),
            {"prompt": "full prompt", "semantic_conditions": [{"text": "action", "type": "action"}]},
        )
        payload = repair_diagnostics(result)
        required = {
            "video_path", "prompt", "official_global_score", "semantic_conditions",
            "condition_scores", "condition_mean", "condition_min", "weakest_condition",
            "condition_score", "repair_score", "num_conditions", "condition_source",
            "warnings", "alpha", "lambda",
        }
        self.assertTrue(required.issubset(payload))
        self.assertEqual(payload["semantic_conditions"][0]["type"], "action")
        self.assertEqual(
            payload["weakest_condition"],
            {
                "text": "action",
                "type": "action",
                "source": "provided",
                "score": payload["condition_min"],
            },
        )
        self.assertIsNone(payload["fallback_reason"])

    def test_weakest_condition_diagnostics_match_first_minimum(self):
        result = RepairEvaluator(CountingEncoder()).evaluate_video(
            Path("/tmp/video.mp4"),
            {
                "prompt": "full prompt",
                "semantic_conditions": [
                    {"text": "subject", "type": "subject"},
                    {"text": "action", "type": "action"},
                    {"text": "subject", "type": "duplicate"},
                ],
            },
        )
        payload = repair_diagnostics(result)
        weakest = payload["weakest_condition"]
        self.assertEqual(weakest["score"], payload["condition_min"])
        self.assertEqual(weakest["text"], "subject")
        self.assertEqual(weakest["type"], "subject")
        self.assertEqual(weakest["source"], "provided")


if __name__ == "__main__":
    unittest.main()
