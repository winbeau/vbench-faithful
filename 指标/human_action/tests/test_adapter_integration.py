from pathlib import Path
import unittest

import numpy as np

from human_action.backends.audit import AuditHumanActionEvaluator
from human_action.diagnostics import DiagnosticsLevel
from human_action.metric import evaluate_audit_batch


def categories():
    return ("ironing",) + tuple(f"action {index}" for index in range(1, 400))


class FakeClassifier:
    @property
    def categories(self):
        return categories()

    @staticmethod
    def decode_video(video):
        return np.zeros((8, 4, 4, 3), dtype=np.uint8), 8.0

    @staticmethod
    def predict_frame_indices(frames, indices):
        values = np.zeros(400)
        values[0] = 0.9
        return values


class AdapterIntegrationTests(unittest.TestCase):
    def test_explicit_metadata_target_reaches_repair_adapter(self):
        video = Path("/tmp/filename-must-not-control-target.mp4")
        metadata = {
            video.name: {
                "prompt": "A person is ironing",
                "dimension_metadata": {"target_action": "ironing"},
            }
        }
        results = evaluate_audit_batch(
            [video], metadata, "cuda:0", Path("/tmp/unused.pth"), categories(),
            DiagnosticsLevel.FULL, evaluator=AuditHumanActionEvaluator(FakeClassifier()),
        )
        self.assertEqual(results[0]["status"], "succeeded")
        self.assertAlmostEqual(results[0]["score"], results[0]["temporal_mean_target_probability"])
        self.assertEqual(results[0]["target_action"], "ironing")
        self.assertEqual(results[0]["target_source"], "metadata_target_action")
        self.assertIn("temporal_coverage", results[0]["task_relevant_action_evidence"])


if __name__ == "__main__":
    unittest.main()
