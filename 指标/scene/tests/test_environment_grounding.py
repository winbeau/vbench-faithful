import unittest

from scene.backends.audit import aggregate_frame_score, environment_views


class EnvironmentGroundingTests(unittest.TestCase):
    def test_global_high_regional_low_is_suppressed(self):
        score = aggregate_frame_score(0.9, [0.1, 0.1, 0.2, 0.1])
        self.assertAlmostEqual(score, 0.1125)
        self.assertLess(score, 0.2)

    def test_distributed_support_remains_high(self):
        score = aggregate_frame_score(0.85, [0.8, 0.82, 0.78, 0.84])
        self.assertAlmostEqual(score, 0.6885)
        self.assertGreater(score, 0.6)

    def test_global_failure_is_suppressed(self):
        score = aggregate_frame_score(0.1, [0.9, 0.9, 0.9, 0.9])
        self.assertAlmostEqual(score, 0.09)
        self.assertLess(score, 0.2)

    def test_regional_support_is_monotonic_for_fixed_global_score(self):
        low = aggregate_frame_score(0.9, [0.1, 0.2, 0.3, 0.4])
        high = aggregate_frame_score(0.9, [0.2, 0.3, 0.4, 0.5])
        self.assertGreaterEqual(high, low)

    def test_environment_views_are_distinct_non_overlapping_tiles(self):
        class RecordingImage:
            size = (100, 80)

            def crop(self, box):
                return box

        views = environment_views(RecordingImage())
        self.assertEqual(
            views,
            [
                ("upper-left", (0, 0, 50, 40)),
                ("upper-right", (50, 0, 100, 40)),
                ("lower-left", (0, 40, 50, 80)),
                ("lower-right", (50, 40, 100, 80)),
            ],
        )
        self.assertEqual(len({box for _, box in views}), 4)


if __name__ == "__main__":
    unittest.main()
