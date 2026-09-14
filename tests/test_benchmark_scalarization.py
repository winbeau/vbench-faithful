import math
import unittest

import numpy as np

from dynamic_degree.backends.audit import analyze_timed_flow_sequence
from dynamic_degree.schemas import TimedFrame, TimedFrameSequence
from human_action.backends.audit import AuditHumanActionEvaluator, parse_action_query


def dynamic_sequence():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return TimedFrameSequence(
        (TimedFrame(frame, 0.0, 0), TimedFrame(frame.copy(), 0.125, 1)),
        (64, 64),
        8.0,
        1,
        "test",
    )


class DynamicBenchmarkScalarTests(unittest.TestCase):
    def test_full_score_is_routed_intensity_and_deterministic(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (4.0, 0.0)
        first = analyze_timed_flow_sequence(
            "arbitrary-name.mp4", "a person is running", dynamic_sequence(),
            lambda left, right: flow,
        )
        second = analyze_timed_flow_sequence(
            "other-name.mp4", "a person is running", dynamic_sequence(),
            lambda left, right: flow,
        )
        self.assertAlmostEqual(
            first.score, first.task_relevant_motion_evidence["motion_intensity"]
        )
        self.assertTrue(math.isfinite(first.score))
        self.assertAlmostEqual(first.score, second.score)
        self.assertNotAlmostEqual(first.score, first.apparent.motion_intensity)


class FakeHumanClassifier:
    categories = ("ironing",) + tuple(f"action {index}" for index in range(1, 400))

    @staticmethod
    def decode_video(video):
        return np.zeros((8, 4, 4, 3), dtype=np.uint8), 8.0

    @staticmethod
    def predict_frame_indices(frames, indices):
        values = np.zeros(400)
        values[0] = 0.9 if max(indices) < 4 else 0.1
        return values


class HumanBenchmarkScalarTests(unittest.TestCase):
    def test_score_is_weighted_temporal_target_mean(self):
        query = parse_action_query(
            {
                "prompt": "A person is ironing",
                "dimension_metadata": {"target_action": "ironing"},
            },
            FakeHumanClassifier.categories,
        )
        result = AuditHumanActionEvaluator(FakeHumanClassifier()).evaluate_video(
            "filename-must-not-control-target.mp4", query
        )
        weights = [window.weight for window in result.windows]
        probabilities = [
            window.classification.target_probability for window in result.windows
        ]
        expected = sum(w * p for w, p in zip(weights, probabilities)) / sum(weights)
        self.assertAlmostEqual(result.score, expected)
        self.assertAlmostEqual(result.score, result.temporal_mean_probability)
        self.assertTrue(math.isfinite(result.score))
        self.assertEqual(query.target_source, "metadata_target_action")

    def test_explicit_target_is_filename_invariant_and_invalid_prompt_fails(self):
        metadata = {
            "prompt": "A person is ironing",
            "dimension_metadata": {"target_action": "ironing"},
        }
        self.assertEqual(
            parse_action_query(metadata, FakeHumanClassifier.categories).target_action,
            "ironing",
        )
        with self.assertRaises(ValueError):
            parse_action_query({"prompt": "a person prepares an object"}, FakeHumanClassifier.categories)


if __name__ == "__main__":
    unittest.main()
