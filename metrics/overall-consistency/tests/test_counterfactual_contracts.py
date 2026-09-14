import unittest

from overall_consistency.metric import aggregate_condition_scores


class CounterfactualContractTests(unittest.TestCase):
    def test_more_weak_conditions_monotonically_reduce_synthetic_score(self):
        all_strong = aggregate_condition_scores([0.9, 0.9, 0.9, 0.9]).score
        one_weak = aggregate_condition_scores([0.9, 0.9, 0.9, 0.1]).score
        multiple_weak = aggregate_condition_scores([0.9, 0.1, 0.1, 0.1]).score
        self.assertGreater(all_strong, one_weak)
        self.assertGreater(one_weak, multiple_weak)

    def test_diagnostic_metadata_cannot_change_score(self):
        scores = [0.8, 0.7, 0.6]
        before = aggregate_condition_scores(scores).score
        diagnostic_only = [{"type": "action"}, {"type": "scene"}, {"type": "other", "note": "changed"}]
        self.assertEqual(len(diagnostic_only), len(scores))
        after = aggregate_condition_scores(scores).score
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
