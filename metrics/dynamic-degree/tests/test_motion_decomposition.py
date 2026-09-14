import unittest

import numpy as np

from dynamic_degree.motion import decompose_motion, top_fraction_mean


def affine_test_flow(height, width, scale=1.0, angle=0.0, tx=0.0, ty=0.0):
    yy, xx = np.mgrid[:height, :width]
    cosine = np.cos(angle) * scale
    sine = np.sin(angle) * scale
    destination_x = cosine * xx - sine * yy + tx
    destination_y = sine * xx + cosine * yy + ty
    return np.stack((destination_x - xx, destination_y - yy), axis=-1)


class MotionDecompositionTests(unittest.TestCase):
    def test_zero_motion(self):
        result = decompose_motion(np.zeros((32, 32, 2), dtype=np.float64))
        self.assertAlmostEqual(result.global_statistics.top5_mean, 0.0)
        self.assertAlmostEqual(result.residual_statistics.top5_mean, 0.0)

    def test_pure_global_translation(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (3.0, 2.0)
        result = decompose_motion(flow)
        self.assertGreater(result.apparent_statistics.top5_mean, 0.0)
        self.assertAlmostEqual(result.global_statistics.top5_mean, np.hypot(3.0, 2.0), places=5)
        self.assertLess(result.residual_statistics.top5_mean, 1e-5)
        self.assertEqual(
            result.affine.candidate_strategy,
            "geometric_inlier_residual_consistent_border_band_prior",
        )
        self.assertEqual(result.affine.candidate_count, 64 * 64 - 32 * 32)
        self.assertNotIn("corner", result.affine.candidate_strategy)

    def test_local_subject_motion_is_residual(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[20:44, 20:44, 0] = 5.0
        result = decompose_motion(flow)
        self.assertLess(result.global_statistics.top5_mean, 1e-5)
        self.assertGreater(result.residual_statistics.top5_mean, 4.9)

    def test_subject_plus_camera(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (2.0, 0.0)
        flow[20:44, 20:44, 0] += 5.0
        result = decompose_motion(flow)
        self.assertAlmostEqual(result.global_statistics.top5_mean, 2.0, places=4)
        self.assertGreater(result.residual_statistics.top5_mean, 4.9)

    def test_large_moving_foreground_keeps_background_camera_fit(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (2.0, 0.0)
        flow[12:52, 12:52, 0] += 5.0
        result = decompose_motion(flow)
        self.assertAlmostEqual(result.global_statistics.top5_mean, 2.0, places=3)
        self.assertGreater(result.residual_statistics.top5_mean, 4.9)

    def test_dominant_foreground_degrades_affine_fit_diagnostics(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (2.0, 0.0)
        flow[8:56, 8:56, 0] += 5.0
        result = decompose_motion(flow)
        self.assertLess(result.affine.inlier_ratio, 0.75)
        self.assertGreater(result.residual_statistics.top5_mean, 4.9)

    def test_strong_piecewise_parallax_lowers_fit_quality(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:, :32, 0] = -3.0
        flow[:, 32:, 0] = 3.0
        result = decompose_motion(flow)
        self.assertLess(result.affine.inlier_ratio, 0.75)
        self.assertGreater(result.residual_statistics.top5_mean, 1.0)

    def test_rotation_and_zoom_partial_affine(self):
        for name, flow in (
            ("rotation", affine_test_flow(64, 64, angle=0.05, tx=2.0, ty=-1.0)),
            ("zoom", affine_test_flow(64, 64, scale=1.04, tx=1.0, ty=2.0)),
        ):
            with self.subTest(name=name):
                result = decompose_motion(flow)
                self.assertLess(result.residual_statistics.top5_mean, 1e-4)
                self.assertGreater(result.affine.inlier_ratio, 0.99)

    def test_small_and_large_regions_keep_top5_nonempty(self):
        one_pixel = np.array([[3.0]])
        self.assertEqual(top_fraction_mean(one_pixel), 3.0)
        for size in (2, 48):
            flow = np.zeros((64, 64, 2), dtype=np.float64)
            flow[:size, :size, 0] = 2.0
            self.assertTrue(np.isfinite(decompose_motion(flow).residual_statistics.top5_mean))
