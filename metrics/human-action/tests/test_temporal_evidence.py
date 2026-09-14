import unittest

import numpy as np

from human_action.backends.audit import AuditHumanActionEvaluator, aggregate_temporal_evidence, class_evidence
from human_action.models import temporal_window_indices
from human_action.schemas import ActionQuery, TemporalWindowEvidence


def categories():
    return ("ironing",) + tuple(f"action {index}" for index in range(1, 400))


def window(index, probability, weight=1.0):
    values = np.zeros(400)
    values[0] = probability
    return TemporalWindowEvidence(
        window_index=index,
        source_start=index,
        source_stop=index + 1,
        sampled_source_frame_indices=(index,) * 16,
        weight=weight,
        classification=class_evidence(values, categories(), "ironing"),
    )


class TemporalEvidenceTests(unittest.TestCase):
    def test_persistent_execution_exceeds_brief_execution(self):
        brief = tuple(window(index, value) for index, value in enumerate((0.1, 0.9, 0.1, 0.1)))
        persistent = tuple(window(index, 0.9) for index in range(4))
        brief_mean, brief_coverage = aggregate_temporal_evidence(brief)
        persistent_mean, persistent_coverage = aggregate_temporal_evidence(persistent)
        self.assertGreater(persistent_mean, brief_mean)
        self.assertEqual(brief_coverage, 0.25)
        self.assertEqual(persistent_coverage, 1.0)

    def test_duration_weighted_aggregation(self):
        mean, coverage = aggregate_temporal_evidence((window(0, 0.9, 1.0), window(1, 0.1, 3.0)))
        self.assertAlmostEqual(mean, 0.3)
        self.assertAlmostEqual(coverage, 0.25)

    def test_window_sampling_is_deterministic_and_complete(self):
        windows = temporal_window_indices(10, max_windows=4)
        self.assertEqual([(start, stop) for start, stop, _ in windows], [(0, 2), (2, 5), (5, 7), (7, 10)])
        self.assertTrue(all(len(indices) == 16 for _, _, indices in windows))
        self.assertEqual(sum(stop - start for start, stop, _ in windows), 10)

    def test_audit_evaluator_emits_structured_evidence_with_scalar(self):
        class FakeClassifier:
            categories = categories()

            @staticmethod
            def decode_video(video):
                return np.zeros((8, 4, 4, 3), dtype=np.uint8), 8.0

            @staticmethod
            def predict_frame_indices(frames, indices):
                probabilities = np.zeros(400)
                probabilities[0] = 0.9 if max(indices) < 4 else 0.1
                return probabilities

        result = AuditHumanActionEvaluator(FakeClassifier()).evaluate_video(
            "/tmp/video.mp4",
            ActionQuery("A person is ironing", "ironing", "metadata_target_action"),
        )
        self.assertEqual(result.status, "succeeded")
        self.assertAlmostEqual(result.score, result.temporal_mean_probability)
        self.assertTrue(np.isfinite(result.score))
        self.assertEqual(len(result.windows), 4)
        self.assertIn("temporal_coverage", result.task_evidence())
