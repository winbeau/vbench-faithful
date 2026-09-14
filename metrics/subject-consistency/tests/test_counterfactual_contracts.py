import math
import unittest

import torch

from subject_consistency.backends.audit import counterfactual_gaps
from subject_consistency.metric import global_pairwise_consistency, local_consistency, subject_consistency_score


def angular_features(degrees):
    return torch.tensor([[math.cos(math.radians(x)), math.sin(math.radians(x))] for x in degrees], dtype=torch.float64)


class CounterfactualContractTests(unittest.TestCase):
    def test_local_discontinuity_sensitivity(self):
        clean = torch.tensor([[1.0, 0.0]] * 4)
        corrupted = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        self.assertLess(local_consistency(corrupted), local_consistency(clean))

    def test_accumulated_drift_sensitivity(self):
        drift = angular_features([0, 15, 30, 45, 60])
        self.assertLess(global_pairwise_consistency(drift), local_consistency(drift))

    def test_temporal_position_robustness_of_global_term(self):
        features = angular_features([0, 20, 50, 80])
        moved = features[[2, 0, 3, 1]]
        self.assertAlmostEqual(global_pairwise_consistency(features), global_pairwise_consistency(moved))

    def test_four_gaps_are_diagnostics_not_score_inputs(self):
        reference = subject_consistency_score(torch.tensor([[1.0, 0.0]] * 3))
        gaps = counterfactual_gaps(
            reference,
            temporal_position_score=0.95,
            subject_corruption_score=0.4,
            background_score=0.8,
            scale_score=0.7,
        )
        self.assertAlmostEqual(gaps.temporal_position_gap, 0.05)
        self.assertAlmostEqual(gaps.subject_corruption_gap, 0.6)
        self.assertAlmostEqual(gaps.background_gap, 0.2)
        self.assertAlmostEqual(gaps.scale_gap, 0.3)
        self.assertEqual(reference, 1.0)


if __name__ == "__main__":
    unittest.main()
