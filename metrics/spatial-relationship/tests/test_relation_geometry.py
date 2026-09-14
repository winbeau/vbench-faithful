import random
import unittest

from spatial_relationship.relation import ordered_position_score


class OrderedGeometryTests(unittest.TestCase):
    def test_four_directions(self):
        left = (0, 0, 2, 2)
        right = (8, 0, 10, 2)
        top = (0, 0, 2, 2)
        bottom = (0, 8, 2, 10)
        self.assertEqual(ordered_position_score("left", left, right).score, 1)
        self.assertEqual(ordered_position_score("right", right, left).score, 1)
        self.assertEqual(ordered_position_score("top", top, bottom).score, 1)
        self.assertEqual(ordered_position_score("bottom", bottom, top).score, 1)
        self.assertEqual(ordered_position_score("left", right, left).score, 0)

    def test_official_iou_penalty_is_preserved(self):
        evidence = ordered_position_score("left", (0, 0, 10, 10), (6, 0, 16, 10))
        self.assertAlmostEqual(evidence.iou, 40 / 160)
        self.assertAlmostEqual(evidence.score, 0.1 / (40 / 160))

    def test_reciprocal_algebraic_property(self):
        rng = random.Random(42)
        for _ in range(100):
            ax, ay = rng.uniform(0, 20), rng.uniform(0, 20)
            bx, by = rng.uniform(0, 20), rng.uniform(0, 20)
            a = (ax, ay, ax + 2, ay + 2)
            b = (bx, by, bx + 2, by + 2)
            self.assertAlmostEqual(ordered_position_score("left", a, b).score, ordered_position_score("right", b, a).score)
            self.assertAlmostEqual(ordered_position_score("top", a, b).score, ordered_position_score("bottom", b, a).score)
