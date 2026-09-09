import unittest

from spatial_relationship.backends.audit import score_predictions
from spatial_relationship.schemas import OrderedRelationQuery


class RoleBindingTests(unittest.TestCase):
    detections = [[("cat", (0, 0, 2, 2), 0.9), ("dog", (8, 0, 10, 2), 0.8)]]

    def score(self, subject, relation, object_):
        return score_predictions("video.mp4", "prompt", OrderedRelationQuery(subject, relation, object_), self.detections).video_score

    def test_correct_role(self):
        self.assertEqual(self.score("cat", "on the left of", "dog"), 1)

    def test_role_swap(self):
        self.assertEqual(self.score("dog", "on the left of", "cat"), 0)

    def test_reciprocal(self):
        self.assertEqual(self.score("dog", "on the right of", "cat"), self.score("cat", "on the left of", "dog"))

    def test_horizontal_flip(self):
        original = self.score("cat", "on the left of", "dog")
        flipped = [[("cat", (8, 0, 10, 2), 0.9), ("dog", (0, 0, 2, 2), 0.8)]]
        flipped_score = score_predictions("video.mp4", "prompt", OrderedRelationQuery("cat", "on the left of", "dog"), flipped).video_score
        self.assertGreater(original, flipped_score)
