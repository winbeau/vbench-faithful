"""Construction fixtures are synthetic; scientific inputs are official MP4s."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.counterfactual.local_texture_jitter import alternating_phase, local_texture_jitter
from scripts.counterfactual.official_video_jitter import interventions, validate_native


CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "tests/fixtures/construction/construction.local-texture-dev5-v1.json").read_text())


def frames():
    image = np.random.default_rng(7).integers(30, 220, (96, 128, 3), dtype=np.uint8)
    return np.stack([np.roll(image, t, axis=1) for t in range(6)])


@pytest.mark.parametrize("amplitude", [1, 2, 4])
@pytest.mark.parametrize("seed", [1701, 2904])
def test_bounded_local_zero_mean_motion_without_frame_translation(amplitude, seed):
    original = frames()
    actual, info, field = local_texture_jitter(original, amplitude, seed, CONFIG)
    assert info["geometry_qualified"] and info["sampling_coordinates_in_bounds"]
    assert not info["intensity_noise_added"]
    assert info["max_displacement_pixels"] == pytest.approx(amplitude)
    assert info["minimum_warp_jacobian"] >= CONFIG["min_warp_jacobian"]
    assert np.linalg.norm(info["spatial_mean_displacement_pixels"]) < 1e-5
    assert info["temporal_mean_displacement_max_pixels"] == 0
    assert np.array_equal(actual[:, 0], original[:, 0])
    assert np.array_equal(actual[:, -1], original[:, -1])
    assert np.array_equal(actual[:, :, 0], original[:, :, 0])
    assert np.array_equal(actual[:, :, -1], original[:, :, -1])
    # Each original current frame, not the preceding warped frame, is sampled once.
    yy, xx = np.mgrid[:96, :128].astype(np.float32)
    for t, coefficient in enumerate(field["temporal_phase"]):
        expected = cv2.remap(original[t], xx + coefficient * field["displacement_xy"][..., 0],
                             yy + coefficient * field["displacement_xy"][..., 1],
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        assert np.array_equal(actual[t], expected)
    repeat, repeat_info, _ = local_texture_jitter(original, amplitude, seed, CONFIG)
    assert np.array_equal(repeat, actual)
    assert repeat_info == info


def test_no_added_noise_or_synthetic_motion_on_constant_color():
    original = np.full((6, 96, 128, 3), [41, 122, 208], dtype=np.uint8)
    actual, _, _ = local_texture_jitter(original, 4, 1701, CONFIG)
    assert np.array_equal(actual, original)
    actual, _, _ = local_texture_jitter(frames(), 0, 1701, CONFIG)
    assert np.array_equal(actual, frames())


def test_odd_frame_count_still_has_zero_temporal_mean():
    assert alternating_phase(7, 1701).mean() == pytest.approx(0, abs=1e-7)
    with pytest.raises(ValueError):
        alternating_phase(1, 1701)
    with pytest.raises(ValueError):
        local_texture_jitter(frames(), -1, 1701, CONFIG)




def test_native_quality_cannot_accept_invalid_geometry():
    original = frames()
    candidate, info, _ = local_texture_jitter(original, 2, 1701, CONFIG)
    pts = (np.arange(len(original)) / 10).tolist()
    result = validate_native(candidate, candidate, original, pts, 10, pts, 10,
                             "local_texture_alternating", {**info, "geometry_qualified": False}, CONFIG)
    assert not result["qualified"]
    assert "local_displacement_geometry_failed" in result["reason"]
