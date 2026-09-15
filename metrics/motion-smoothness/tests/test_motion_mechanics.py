import math
import unittest

import numpy as np

from motion_smoothness.metric import (
    aggregate_temporal_discontinuity,
    align_motion_field,
    analyze_motion_fields,
    discontinuity_to_score,
)
from motion_smoothness.schemas import MotionField, MotionSmoothnessConfig


def fields(values, dts=None, shape=(12, 16)):
    dts = dts or [1.0] * len(values)
    result = []
    for index, (value, dt) in enumerate(zip(values, dts)):
        vector = value if isinstance(value, tuple) else (value, 0.0)
        result.append(MotionField(np.full((*shape, 2), vector, dtype=float), dt, index))
    return result


class MotionMechanicsTests(unittest.TestCase):
    def test_constant_velocity_and_large_constant_velocity_are_smooth(self):
        one = analyze_motion_fields(fields([1] * 8))
        ten = analyze_motion_fields(fields([10] * 8))
        self.assertEqual(one["D_video"], 0.0)
        self.assertEqual(ten["D_video"], 0.0)
        self.assertEqual(one["score"], 1.0)
        self.assertEqual(ten["score"], 1.0)

    def test_gradual_acceleration_is_smoother_than_abrupt_change(self):
        gradual = analyze_motion_fields(fields([1, 2, 3, 4, 5, 6, 7, 8]))
        abrupt = analyze_motion_fields(fields([1, 1, 1, 5, 1, 1, 1, 1]))
        self.assertGreater(gradual["score"], abrupt["score"])

    def test_duplicate_stall_and_abrupt_jump_reduce_smoothness(self):
        constant = analyze_motion_fields(fields([1] * 8))
        duplicate = analyze_motion_fields(fields([1, 1, 1, 0, 1, 1, 1, 1]))
        jump = analyze_motion_fields(fields([1, 1, 1, 4, 1, 1, 1, 1]))
        self.assertLess(duplicate["score"], constant["score"])
        self.assertLess(jump["score"], constant["score"])

    def test_smooth_curved_motion_beats_abrupt_direction_reversal(self):
        angles = np.linspace(0.0, math.pi / 2.0, 8)
        curved = analyze_motion_fields(fields([(math.cos(angle), math.sin(angle)) for angle in angles]))
        reversal = analyze_motion_fields(
            fields([(1, 0), (1, 0), (1, 0), (-1, 0), (1, 0), (1, 0), (1, 0), (1, 0)])
        )
        self.assertGreater(curved["diagnostics"]["direction_discontinuity"], 0.0)
        self.assertGreater(curved["score"], reversal["score"])

    def test_time_normalization_matches_equivalent_physical_velocity(self):
        physical = analyze_motion_fields(fields([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]))
        resampled = analyze_motion_fields(fields([2, 4, 6, 8, 10], [2, 2, 2, 2, 2]))
        self.assertAlmostEqual(physical["score"], resampled["score"], places=12)
        np.testing.assert_allclose(
            physical["diagnostics"]["D_t"], resampled["diagnostics"]["D_t"], atol=1e-12
        )

    def test_bilinear_subpixel_material_point_alignment(self):
        height, width = 8, 10
        xx = np.broadcast_to(np.arange(width, dtype=float), (height, width))
        previous = np.stack([xx, np.zeros_like(xx)], axis=-1)
        current = np.stack([xx - 0.5, np.zeros_like(xx)], axis=-1)
        displacement = np.zeros_like(previous)
        displacement[..., 0] = 0.5
        aligned, mask = align_motion_field(current, displacement)
        self.assertTrue(mask[:, :-1].all())
        self.assertFalse(mask[:, -1].any())
        np.testing.assert_allclose(aligned[mask], previous[mask], atol=1e-12)

        constant_subpixel = analyze_motion_fields(fields([0.5] * 8))
        self.assertLess(constant_subpixel["D_video"], 1e-12)

    def test_alignment_propagates_mask_and_rejects_out_of_bounds(self):
        field = np.ones((6, 8, 2), dtype=float)
        displacement = np.zeros_like(field)
        displacement[..., 0] = 1.0
        source_mask = np.ones((6, 8), dtype=bool)
        source_mask[:, 3] = False
        _, valid = align_motion_field(field, displacement, source_mask)
        self.assertTrue(valid[:, :2].all())
        self.assertFalse(valid[:, 2].any())
        self.assertFalse(valid[:, -1].any())

    def test_upper_tail_mean_detects_rare_failures(self):
        config = MotionSmoothnessConfig(
            tail_weight=0.25, tail_quantile=0.90, temporal_aggregation="mean_tail"
        )
        for bad_percent in (1, 5, 10, 25):
            with self.subTest(bad_percent=bad_percent):
                values = [0.0] * (100 - bad_percent) + [1.0] * bad_percent
                mean, tail, combined = aggregate_temporal_discontinuity(values, config)
                self.assertGreater(tail, mean)
                self.assertGreater(combined, mean)
                self.assertLess(combined, 1.0)

    def test_top_k_aggregation_scores_the_localised_peak(self):
        values = [0.0] * 12 + [1.0, 0.6, 0.4]
        mean, tail, combined = aggregate_temporal_discontinuity(
            values, MotionSmoothnessConfig(top_k=3)
        )
        self.assertAlmostEqual(combined, (1.0 + 0.6 + 0.4) / 3, places=12)
        self.assertGreater(combined, mean)
        self.assertEqual(combined, tail)
        # A single extreme transition is diluted by k=3 but not by k=1.
        single = aggregate_temporal_discontinuity(
            [0.0] * 14 + [1.0], MotionSmoothnessConfig(top_k=3)
        )[2]
        self.assertAlmostEqual(single, 1.0 / 3, places=12)

    def test_temporal_aggregation_and_top_k_validation(self):
        for kwargs in (
            {"temporal_aggregation": "median"},
            {"top_k": 0},
            {"top_k": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionSmoothnessConfig(**kwargs)

    def test_score_mapping_zero_monotonic_finite_and_invalid(self):
        self.assertEqual(discontinuity_to_score(0.0), 1.0)
        scores = [discontinuity_to_score(value) for value in (0.0, 0.1, 1.0, 10.0)]
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertTrue(all(left > right for left, right in zip(scores, scores[1:])))
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                discontinuity_to_score(invalid)

    def test_static_and_short_sequences_are_finite(self):
        static = analyze_motion_fields(fields([0, 0, 0, 0]))
        self.assertEqual(static["score"], 1.0)
        for count in (0, 1, 2):
            with self.subTest(count=count):
                result = analyze_motion_fields(fields([0] * count))
                self.assertTrue(np.isfinite(result["score"]))
                self.assertEqual(result["D_video"], 0.0)

    def test_configuration_rejects_invalid_tail_parameters(self):
        for kwargs in (
            {"tail_quantile": -0.1},
            {"tail_quantile": 1.1},
            {"tail_weight": -0.1},
            {"tail_weight": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionSmoothnessConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
