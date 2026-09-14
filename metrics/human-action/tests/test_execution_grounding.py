import unittest

import numpy as np

from human_action.backends.audit import class_evidence


def categories():
    values = [f"action {index}" for index in range(400)]
    values[9] = "washing dishes"
    return tuple(values)


class ExecutionGroundingTests(unittest.TestCase):
    def test_explicit_target_probability_survives_outside_top5(self):
        probabilities = np.linspace(0.0, 1.0, 400)
        probabilities[9] = 0.61
        evidence = class_evidence(probabilities, categories(), "washing dishes")
        self.assertAlmostEqual(evidence.target_probability, 0.61)
        self.assertGreater(evidence.target_rank, 5)
        self.assertNotIn("washing dishes", evidence.top5_actions)

    def test_non_target_context_class_cannot_substitute_for_target(self):
        probabilities = np.zeros(400)
        probabilities[8] = 0.99
        probabilities[9] = 0.05
        evidence = class_evidence(probabilities, categories(), "washing dishes")
        self.assertEqual(evidence.target_action, "washing dishes")
        self.assertAlmostEqual(evidence.target_probability, 0.05)
        self.assertEqual(evidence.target_rank, 2)

    def test_invalid_prediction_vector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "400-element"):
            class_evidence(np.zeros(399), categories(), "washing dishes")
