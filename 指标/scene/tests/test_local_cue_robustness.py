import unittest

from scene.backends.audit import aggregate_frame_score


class LocalCueRobustnessTests(unittest.TestCase):
    def test_one_strong_region_cannot_dominate(self):
        local = aggregate_frame_score(0.85, [1.0, 0.05, 0.05, 0.05])
        distributed = aggregate_frame_score(0.85, [0.8, 0.8, 0.75, 0.85])
        self.assertLess(local, distributed)
        self.assertLess(local, 0.3)

    def test_requested_clean_case_is_well_separated_from_one_local_cue(self):
        local = aggregate_frame_score(0.9, [1.0, 0.05, 0.05, 0.05])
        distributed = aggregate_frame_score(0.9, [0.8, 0.82, 0.78, 0.84])
        self.assertLess(local, 0.4 * distributed)


if __name__ == "__main__":
    unittest.main()
