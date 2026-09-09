import unittest

from spatial_relationship.models import assign_role_instance, evaluate_frame
from spatial_relationship.schemas import Detection, OrderedRelationQuery


class IdentityAssignmentTests(unittest.TestCase):
    def test_confidence_then_stable_order(self):
        candidates = [
            Detection(0, "dog", (0, 0, 2, 2), 0.5),
            Detection(1, "dog", (4, 0, 6, 2), 0.9),
        ]
        self.assertEqual(assign_role_instance(candidates).detection_id, 1)
        no_confidence = [Detection(4, "dog", (0, 0, 2, 2)), Detection(3, "dog", (4, 0, 6, 2))]
        self.assertEqual(assign_role_instance(no_confidence).detection_id, 4)

    def test_multi_instance_distractor_cannot_drive_assignment(self):
        query = OrderedRelationQuery("cat", "on the left of", "dog")
        detections = [
            Detection(0, "cat", (5, 0, 7, 2), 0.95),
            Detection(1, "dog", (0, 0, 2, 2), 0.90),  # identity-first, relation false
            Detection(2, "dog", (9, 0, 11, 2), 0.51),  # relation true, must not be selected
        ]
        result = evaluate_frame(query, detections, 0)
        self.assertEqual(result.assigned_object_id, 1)
        self.assertEqual(result.frame_score, 0)

    def test_missing_entity_and_identity_corruption(self):
        query = OrderedRelationQuery("cat", "on the left of", "dog")
        result = evaluate_frame(query, [Detection(0, "cat", (0, 0, 2, 2)), Detection(1, "cat", (8, 0, 10, 2))], 0)
        self.assertEqual(result.frame_score, 0)
        self.assertEqual(result.frame_reason, "missing_object")

    def test_same_class_is_ambiguous(self):
        query = OrderedRelationQuery("cat", "on the left of", "cat")
        result = evaluate_frame(query, [Detection(0, "cat", (0, 0, 2, 2)), Detection(1, "cat", (8, 0, 10, 2))], 0)
        self.assertEqual(result.frame_score, 0)
        self.assertTrue(result.role_identity_ambiguous)
        self.assertEqual(result.frame_reason, "same_class_role_ambiguity")
