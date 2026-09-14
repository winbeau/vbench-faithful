import unittest

from spatial_relationship.backends.audit import score_predictions
from spatial_relationship.schemas import OrderedRelationQuery


class CounterfactualContractTests(unittest.TestCase):
    def test_ordered_counterfactual_triple(self):
        detections = [[("cat", (0, 0, 2, 2)), ("dog", (8, 0, 10, 2))]]
        a_left_b = score_predictions("v", "p", OrderedRelationQuery("cat", "left", "dog"), detections)
        b_left_a = score_predictions("v", "p", OrderedRelationQuery("dog", "left", "cat"), detections)
        b_right_a = score_predictions("v", "p", OrderedRelationQuery("dog", "right", "cat"), detections)
        self.assertEqual(a_left_b.video_score, 1)
        self.assertEqual(b_left_a.video_score, 0)
        self.assertEqual(b_right_a.video_score, 1)

    def test_temporary_detection_loss_scores_zero_without_tracking(self):
        detections = [
            [("cat", (0, 0, 2, 2)), ("dog", (8, 0, 10, 2))],
            [("cat", (0, 0, 2, 2))],
        ]
        result = score_predictions("v", "p", OrderedRelationQuery("cat", "left", "dog"), detections)
        self.assertEqual(result.video_score, 0.5)
        self.assertEqual(result.missing_object_count, 1)
        self.assertEqual(result.aggregation_method, "mean_over_16_officially_sampled_frames")
