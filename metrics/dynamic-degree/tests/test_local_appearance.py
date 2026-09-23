import cv2
import numpy as np
import pytest
from scipy.signal import correlate

from dynamic_degree.local_appearance import (
    correlate_full, ncc_surface, correlation_peaks, feature_support, relocate_support,
    bilinear_rgb_gradient, refine_local_appearance,
)


def test_finite_double_correlation_matches_direct_sums():
    rng = np.random.default_rng(71)
    for shape in ((3, 4), (12, 17), (17, 12)):
        a, b = rng.normal(size=(23, 31)), rng.normal(size=shape)
        np.testing.assert_allclose(correlate_full(a, b), correlate(a, b, mode="full", method="direct"), atol=1e-11)


def test_zero_variance_is_missing_not_a_perfect_match():
    flat = np.full((32, 40, 3), 90, np.uint8)
    mask = np.zeros(flat.shape[:2], bool); mask[5:16, 8:23] = True
    assert not correlation_peaks(ncc_surface(flat, flat, mask))
    assert ncc_surface(flat, flat, np.zeros_like(mask)) is None


def test_translation_sign_visible_border_and_returning_motion():
    rng = np.random.default_rng(3)
    frame = rng.integers(30, 220, (40, 56, 3), np.uint8)
    mask = np.zeros(frame.shape[:2], bool); mask[8:32, 8:40] = True
    shifted = cv2.warpAffine(frame, np.array([[1., 0., -5], [0., 1., 3]]), (56, 40))
    result = relocate_support(frame, shifted, mask)
    assert result["hypotheses"][0]["displacement_pixels"] == [-5, 3]
    assert result["hypotheses"][0]["reverse_candidates"][0]["displacement_pixels"] == [5, -3]
    assert result["hypotheses"][0]["reverse_candidates"][0]["closure_error_pixels"] == 0
    assert result["score"] is None
    # Shift a support toward the border: true partially visible placement still
    # participates rather than being excluded by the bounding-box rectangle.
    big = np.ones(frame.shape[:2], bool)
    assert correlation_peaks(ncc_surface(frame, shifted, big), min_overlap=.8)[0]["displacement_pixels"] == [-5, 3]


def test_identity_and_per_channel_offsets_do_not_make_false_displacement():
    rng = np.random.default_rng(6)
    frame = rng.integers(30, 160, (30, 36, 3), np.uint8)
    target = frame + np.array([10, 20, 30], np.uint8)
    mask = np.zeros(frame.shape[:2], bool); mask[4:25, 5:28] = True
    peak = correlation_peaks(ncc_surface(frame, target, mask))[0]
    assert peak["displacement_pixels"] == [0, 0]
    assert peak["correlation"] == pytest.approx(1, abs=1e-12)


def test_feature_relative_support_does_not_require_known_motion_amplitude():
    mask = np.ones((21, 21), bool)
    small = feature_support(mask, [10, 10], 2.5, 1)
    large = feature_support(mask, [10, 10], 2.5, 2)
    assert large.sum() > small.sum() > 0
    assert not (small & ~large).any()
    with pytest.raises(ValueError):
        feature_support(mask, [10, 10], 2.5, 0)


def test_bilinear_gradient_matches_centered_finite_difference():
    rng = np.random.default_rng(53)
    image = rng.normal(size=(16, 21, 3))
    xy = np.array([[2.3, 4.6], [18.4, 12.7]])
    _, gradient = bilinear_rgb_gradient(image, xy)
    for dim in range(2):
        step = np.eye(2)[dim] * 1e-5
        finite = (bilinear_rgb_gradient(image, xy + step)[0] - bilinear_rgb_gradient(image, xy - step)[0]) / 2e-5
        np.testing.assert_allclose(gradient[..., dim], finite, atol=1e-8)


def test_deformable_photometric_fit_recovers_small_affine_change():
    rng = np.random.default_rng(66)
    source = cv2.GaussianBlur(rng.integers(10, 240, (80, 96, 3), np.uint8), (0, 0), 1.2)
    matrix = np.array([[1.04, -.07, 3.], [.06, .95, -2.]])
    target = cv2.warpAffine(source, matrix, (96, 80))
    mask = np.zeros(source.shape[:2], bool); mask[22:58, 26:66] = True
    center = np.array([45.5, 39.5])
    d = matrix @ np.r_[center, 1] - center
    result = refine_local_appearance(source, target, mask, d, model="affine")
    assert result["status"] == "diagnostic_only"
    assert result["jacobian_rank"] == 6
    assert result["correlation"] > result["initial_correlation"]
    np.testing.assert_allclose(np.array(result["affine_matrix"])[:, :2], matrix[:, :2], atol=.015)
    np.testing.assert_allclose(np.array(result["affine_matrix"])[:, 2], matrix[:, 2], atol=.3)
    assert result["score"] is None


def test_refinement_keeps_translation_returning_motion_and_flat_missing():
    rng = np.random.default_rng(32)
    source = cv2.GaussianBlur(rng.integers(10, 240, (64, 64, 3), np.uint8), (0, 0), 1.)
    target = cv2.warpAffine(source, np.array([[1., 0., 5], [0., 1., -3]]), (64, 64))
    mask = np.zeros((64, 64), bool); mask[15:40, 15:40] = True
    for a, b, d in ((source, target, [4.7, -2.8]), (target, source, [-4.7, 2.8])):
        result = refine_local_appearance(a, b, mask, d, model="translation")
        np.testing.assert_allclose(result["center_displacement_pixels"], np.rint(d), atol=.02)
        assert result["fixed_visible_pixels"] == mask.sum()
    flat = np.full_like(source, 100)
    assert refine_local_appearance(flat, flat, mask, [0, 0])["status"] == "insufficient_evidence"
    assert refine_local_appearance(source, target, mask, [100, 0])["status"] == "insufficient_evidence"
