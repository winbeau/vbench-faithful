import unittest
import math

from overall_consistency.metric import aggregate_condition_scores, combine_global_and_condition_scores


class ViolationAggregationTests(unittest.TestCase):
    def test_clean_exceeds_one_violation_with_exact_formula(self):
        clean = aggregate_condition_scores([0.9, 0.9, 0.9, 0.9], alpha=0.5)
        violation = aggregate_condition_scores([0.9, 0.9, 0.9, 0.1], alpha=0.5)
        self.assertAlmostEqual(clean.mean, 0.9)
        self.assertAlmostEqual(clean.minimum, 0.9)
        self.assertAlmostEqual(clean.score, 0.9)
        self.assertAlmostEqual(violation.mean, 0.7)
        self.assertAlmostEqual(violation.minimum, 0.1)
        self.assertAlmostEqual(violation.score, 0.4)
        self.assertGreater(clean.score, violation.score)
        self.assertAlmostEqual(combine_global_and_condition_scores(0.8, violation.score, 0.5), 0.6)

    def test_one_condition_mean_equals_minimum(self):
        result = aggregate_condition_scores([-0.2], alpha=0.25)
        self.assertAlmostEqual(result.mean, -0.2)
        self.assertAlmostEqual(result.minimum, -0.2)
        self.assertAlmostEqual(result.score, -0.2)

    def test_alpha_and_lambda_are_validated(self):
        for alpha in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                aggregate_condition_scores([0.1], alpha=alpha)
        for lambda_ in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(lambda_=lambda_), self.assertRaises(ValueError):
                combine_global_and_condition_scores(0.1, 0.2, lambda_=lambda_)

    def test_all_equal_scores_reduce_exactly_to_that_score(self):
        value = 0.12345678901234567
        for alpha in (0.0, 0.1, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                result = aggregate_condition_scores([value] * 4, alpha=alpha)
                self.assertEqual(result.mean, value)
                self.assertEqual(result.minimum, value)
                self.assertEqual(result.score, value)

    def test_alpha_boundaries_select_minimum_and_mean(self):
        scores = [-0.4, 0.2, 0.8, 1.0]
        self.assertEqual(aggregate_condition_scores(scores, alpha=0.0).score, -0.4)
        self.assertEqual(aggregate_condition_scores(scores, alpha=1.0).score, 0.4)

    def test_lambda_boundaries_select_global_and_condition(self):
        self.assertEqual(combine_global_and_condition_scores(-0.2, 0.7, lambda_=0.0), -0.2)
        self.assertEqual(combine_global_and_condition_scores(-0.2, 0.7, lambda_=1.0), 0.7)

    def test_lowering_one_condition_cannot_raise_condition_or_repair_score(self):
        baseline = aggregate_condition_scores([0.8, 0.7, 0.6, 0.5], alpha=0.5)
        lowered = aggregate_condition_scores([0.8, 0.7, 0.6, 0.1], alpha=0.5)
        self.assertLessEqual(lowered.score, baseline.score)
        self.assertLessEqual(
            combine_global_and_condition_scores(0.9, lowered.score, lambda_=0.5),
            combine_global_and_condition_scores(0.9, baseline.score, lambda_=0.5),
        )

    def test_weakest_tie_uses_first_occurrence(self):
        result = aggregate_condition_scores([0.4, -0.2, 0.8, -0.2], alpha=0.5)
        self.assertEqual(result.weakest_index, 1)


if __name__ == "__main__":
    unittest.main()
