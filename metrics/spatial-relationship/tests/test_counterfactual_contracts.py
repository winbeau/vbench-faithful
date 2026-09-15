import unittest

from spatial_relationship.backends.audit import score_predictions
from spatial_relationship.models import AblationMode
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
        self.assertEqual(result.scored_frame_count, 2)
        self.assertEqual(result.detected_frame_count, 1)

    def test_detection_conditioned_removes_dropouts_from_the_denominator(self):
        """Plan 9.6 setting 1: an abstention must not read as a wrong direction."""

        detections = [
            [("cat", (0, 0, 2, 2)), ("dog", (8, 0, 10, 2))],
            [("cat", (0, 0, 2, 2))],
        ]
        query = OrderedRelationQuery("cat", "left", "dog")
        end_to_end = score_predictions("v", "p", query, detections)
        conditioned = score_predictions("v", "p", query, detections, condition_on_detection=True)
        self.assertEqual(end_to_end.video_score, 0.5)
        self.assertEqual(conditioned.video_score, 1.0)
        self.assertEqual(conditioned.scored_frame_count, 1)
        self.assertEqual(conditioned.detected_frame_count, 1)
        self.assertEqual(conditioned.aggregation_method, "mean_over_frames_with_both_roles_detected")

    def test_detection_conditioned_keeps_direction_sensitivity(self):
        detections = [[("cat", (0, 0, 2, 2)), ("dog", (8, 0, 10, 2))]]
        query = OrderedRelationQuery("cat", "left", "dog")
        original = score_predictions("v", "p", query, detections, condition_on_detection=True)
        flipped = score_predictions(
            "v", "p", query,
            [[("cat", (8, 0, 10, 2)), ("dog", (0, 0, 2, 2))]],
            condition_on_detection=True,
        )
        self.assertEqual(original.video_score, 1.0)
        self.assertEqual(flipped.video_score, 0.0)

    def test_ordered_role_mode_maximises_signed_geometry_over_role_pairs(self):
        """`ordered_role` searches role-consistent pairs instead of binding one instance.

        The existing identity-first contract (see `test_identity_assignment.py`)
        deliberately commits to the highest-confidence instance per role.  This
        mode is the looser signed variant, so a lower-confidence but
        geometrically correct pair still counts.
        """

        detections = [[
            ("cat", (5, 0, 7, 2), 0.95),
            ("dog", (0, 0, 2, 2), 0.90),
            ("dog", (9, 0, 11, 2), 0.51),
        ]]
        query = OrderedRelationQuery("cat", "left", "dog")
        identity = score_predictions("v", "p", query, detections)
        ordered = score_predictions("v", "p", query, detections, mode=AblationMode.ORDERED_ROLE)
        self.assertEqual(identity.video_score, 0.0)
        self.assertEqual(ordered.video_score, 1.0)

    def test_ordered_role_mode_still_rejects_the_mirrored_geometry(self):
        query = OrderedRelationQuery("cat", "left", "dog")
        original = [[("cat", (0, 0, 2, 2), 0.9), ("dog", (8, 0, 10, 2), 0.8)]]
        mirrored = [[("cat", (8, 0, 10, 2), 0.9), ("dog", (0, 0, 2, 2), 0.8)]]
        self.assertEqual(
            score_predictions("v", "p", query, original, mode=AblationMode.ORDERED_ROLE).video_score, 1.0
        )
        self.assertEqual(
            score_predictions("v", "p", query, mirrored, mode=AblationMode.ORDERED_ROLE).video_score, 0.0
        )
