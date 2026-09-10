import unittest

from multiple_objects.backends.audit import evaluate_detections
from multiple_objects.metric import aggregate_frame_totals
from multiple_objects.schemas import DetectionEvidence, FrameDetections, MultipleObjectsConfig


class CounterfactualContractTests(unittest.TestCase):
    def metadata(self):
        return {"prompt": "a cat and a dog", "auxiliary_info": {"object": "cat and dog"}}

    def test_sequential_presence_is_not_temporal_union(self):
        frames = [
            [{"label": "cat", "confidence": 0.95}],
            [{"label": "dog", "confidence": 0.95}],
        ]
        result = evaluate_detections("video.mp4", self.metadata(), frames)
        self.assertLess(result["score"], 0.2)

    def test_distractors_do_not_change_target_satisfaction(self):
        clean = evaluate_detections("video.mp4", self.metadata(), [[{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": 0.9}]])
        distractors = evaluate_detections("video.mp4", self.metadata(), [[{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": 0.9}, {"label": "chair", "confidence": 0.99}, {"label": "car", "confidence": 0.99}]])
        self.assertAlmostEqual(clean["score"], distractors["score"])

    def test_threshold_cliff_is_continuous_in_repair(self):
        config = MultipleObjectsConfig(repair_candidate_threshold=0.0)
        values = []
        for confidence in (0.49, 0.50, 0.51):
            result = evaluate_detections("video.mp4", self.metadata(), [[{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": confidence}]], config)
            values.append(result["score"])
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_empty_and_malformed_metadata(self):
        result = evaluate_detections("video.mp4", self.metadata(), [[]])
        self.assertEqual(result["score"], 0.0)
        with self.assertRaises(ValueError):
            evaluate_detections("video.mp4", {"auxiliary_info": {"object": ""}}, [[]])

    def test_complete_pair_and_missing_target_official_decisions(self):
        complete = evaluate_detections(
            "video.mp4", self.metadata(),
            [[{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": 0.9}]],
        )
        missing = evaluate_detections(
            "video.mp4", self.metadata(), [[{"label": "cat", "confidence": 0.9}]],
        )
        self.assertTrue(complete["diagnostics"]["frames"][0]["official_binary_frame_decision"])
        self.assertFalse(missing["diagnostics"]["frames"][0]["official_binary_frame_decision"])

    def test_duplicate_labels_use_max_confidence(self):
        result = evaluate_detections(
            "video.mp4", self.metadata(),
            [[
                {"label": "cat", "confidence": 0.2},
                {"label": "cat", "confidence": 0.8},
                {"label": "dog", "confidence": 0.7},
            ]],
        )
        confidences = result["diagnostics"]["frames"][0]["per_target_official_threshold_score"]
        self.assertEqual(confidences, {"cat": 0.8, "dog": 0.7})

    def test_hard_min_and_softmin_use_the_same_official_score_evidence(self):
        detections = [[
            DetectionEvidence("cat", 0.8, final_instance_confidence=0.3),
            DetectionEvidence("dog", 0.4, final_instance_confidence=0.9),
        ]]
        soft = evaluate_detections("video.mp4", self.metadata(), detections)
        hard = evaluate_detections(
            "video.mp4", self.metadata(), detections,
            MultipleObjectsConfig(aggregation_mode="hard_min"),
        )
        soft_frame = soft["diagnostics"]["frames"][0]
        hard_frame = hard["diagnostics"]["frames"][0]
        self.assertEqual(
            soft_frame["per_target_official_threshold_score"],
            hard_frame["per_target_official_threshold_score"],
        )
        self.assertEqual(hard["score"], 0.4)
        self.assertGreater(soft["score"], hard["score"])
        matched = soft_frame["matched_detections"]["dog"][0]
        self.assertEqual(matched["official_threshold_score"], 0.4)
        self.assertEqual(matched["final_instance_confidence"], 0.9)
        self.assertEqual(soft_frame["evidence_score_source"], "official_grit_roi_threshold_score")

    def test_real_diagnostic_path_uses_separate_official_labels(self):
        frame = FrameDetections(
            detections=(DetectionEvidence("cat", 0.9), DetectionEvidence("dog", 0.2)),
            official_labels=("cat", "dog"),
        )
        result = evaluate_detections("video.mp4", self.metadata(), [frame])
        diagnostics = result["diagnostics"]["frames"][0]
        self.assertTrue(diagnostics["official_binary_frame_decision"])
        self.assertEqual(diagnostics["official_decision_source"], "separate_official_threshold_inference")
        self.assertLess(result["score"], 0.3)

    def test_invalid_confidence_and_zero_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_detections("video.mp4", self.metadata(), [[{"label": "cat", "confidence": 1.1}]])
        with self.assertRaises(ValueError):
            evaluate_detections("video.mp4", self.metadata(), [])

    def test_frame_weighted_shard_aggregation(self):
        self.assertAlmostEqual(aggregate_frame_totals([(1.0, 2), (3.0, 4)]), 4.0 / 6.0)
        self.assertIsNone(aggregate_frame_totals([(0.0, 0)]))
        with self.assertRaises(ValueError):
            aggregate_frame_totals([(2.0, 1)])

    def test_official_and_repaired_outputs_are_isolated(self):
        result = evaluate_detections(
            "video.mp4", self.metadata(),
            [[{"label": "cat", "confidence": 0.9}, {"label": "dog", "confidence": 0.49}]],
        )
        frame = result["diagnostics"]["frames"][0]
        self.assertFalse(frame["official_binary_frame_decision"])
        self.assertGreater(frame["repaired_frame_score"], 0.0)
