import unittest

from multiple_objects.metric import hard_min, softmin
from multiple_objects.schemas import MultipleObjectsConfig


class SoftConjunctionTests(unittest.TestCase):
    def test_monotonic_missing_and_symmetry(self):
        self.assertGreater(softmin([0.9, 0.9]), softmin([0.9, 0.6]))
        self.assertGreater(softmin([0.9, 0.6]), softmin([0.9, 0.1]))
        self.assertLess(softmin([1.0, 0.0]), 0.1)
        self.assertAlmostEqual(softmin([0.2, 0.8]), softmin([0.8, 0.2]))

    def test_large_beta_approaches_hard_min(self):
        self.assertAlmostEqual(softmin([0.2, 0.8], beta=10000), hard_min([0.2, 0.8]), places=3)

    def test_invalid_and_empty_inputs(self):
        with self.assertRaises(ValueError):
            softmin([])
        with self.assertRaises(ValueError):
            hard_min([])
        with self.assertRaises(ValueError):
            softmin([0.2, float("nan")])
        with self.assertRaises(ValueError):
            softmin([1.1, 0.2])
        with self.assertRaises(ValueError):
            softmin([0.2], beta=0.0)

    def test_single_target_is_identity(self):
        self.assertAlmostEqual(softmin([0.37]), 0.37)

    def test_official_threshold_is_frozen(self):
        with self.assertRaises(ValueError):
            MultipleObjectsConfig(official_threshold=0.49)
