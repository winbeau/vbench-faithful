import math
import unittest

import torch

from subject_consistency.metric import global_pairwise_consistency, local_consistency, subject_consistency_diagnostics, subject_consistency_score


def angular_features(degrees):
    return torch.tensor(
        [[math.cos(math.radians(angle)), math.sin(math.radians(angle))] for angle in degrees],
        dtype=torch.float64,
    )


class TemporalAggregationTests(unittest.TestCase):
    def test_identical_features_score_one(self):
        features = torch.tensor([[1.0, 0.0]] * 4)
        diagnostics = subject_consistency_diagnostics(features)
        self.assertEqual(diagnostics.local_score, 1.0)
        self.assertEqual(diagnostics.global_score, 1.0)
        self.assertEqual(diagnostics.final_score, 1.0)

    def test_two_frames_have_equal_local_and_global(self):
        features = angular_features([0, 60])
        self.assertAlmostEqual(local_consistency(features), 0.5)
        self.assertAlmostEqual(global_pairwise_consistency(features), 0.5)
        self.assertAlmostEqual(subject_consistency_score(features), 0.5)

    def test_sudden_local_break_lowers_local_score(self):
        clean = torch.tensor([[1.0, 0.0]] * 4)
        broken = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        self.assertLess(local_consistency(broken), local_consistency(clean))

    def test_gradual_drift_is_more_visible_to_global_pairs(self):
        features = angular_features([0, 15, 30, 45, 60])
        self.assertLess(global_pairwise_consistency(features), local_consistency(features))

    def test_global_pairs_are_permutation_symmetric(self):
        features = angular_features([0, 20, 50, 80])
        permuted = features[[2, 0, 3, 1]]
        self.assertAlmostEqual(global_pairwise_consistency(features), global_pairwise_consistency(permuted))

    def test_negative_cosine_is_clamped_to_zero(self):
        features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        self.assertEqual(subject_consistency_score(features), 0.0)

    def test_fewer_than_two_frames_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            subject_consistency_score(torch.tensor([[1.0, 0.0]]))

    def test_input_must_be_rank_two(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            subject_consistency_score(torch.tensor([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
