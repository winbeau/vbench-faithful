import hashlib

import cv2
import numpy as np
import pytest

from dynamic_degree.local_appearance import feature_support
from scripts.counterfactual.probe_appearance_paths import inspect_triplet, inspect_reference_triplet


def record(xy, d, start, shape):
    support = feature_support(np.ones(shape, bool), xy, 2., 4.)
    key = hashlib.sha256(np.packbits(support).tobytes()).hexdigest()
    return {"start": start, "lag": 1, "source_points": [{"xy": xy, "size": 2., "regional_supports": [
        {"region": 0, "factor": 4, "whole_frame_control": True, "support_sha256": key}]}],
        "supports": {key: {"hypotheses": [{"displacement_pixels": d, "correlation": 1.}]}}}


def test_reference_return_does_not_erase_adjacent_motion_and_missing_stays_nan():
    frame = np.random.default_rng(10).integers(0, 255, (40, 48, 3), np.uint8)
    moved = cv2.warpAffine(frame, np.array([[1., 0., 4.], [0., 1., 0.]]), (48, 40))
    first, second = record([20., 20.], [4, 0], 0, (40, 48)), record([24., 20.], [-4, 0], 1, (40, 48))
    result, arrays = inspect_triplet(np.stack([frame, moved, frame]), np.zeros((0, 40, 48), bool), first, second)
    assert result["score"] is None and result["candidate_paths"] == 1
    assert arrays["actual_first_ncc"][0] == pytest.approx(1.)
    assert arrays["reference_ncc"][0] == pytest.approx(1.)
    np.testing.assert_array_equal(arrays["first_offset"], [[4., 0.]])
    np.testing.assert_array_equal(arrays["total_offset"], [[0., 0.]])


def test_reference_mode_does_not_require_a_middle_keypoint():
    source = np.random.default_rng(20).integers(0, 255, (48, 64, 3), np.uint8)
    frames = np.stack([cv2.warpAffine(source, np.array([[1., 0., dx], [0., 1., 0.]]), (64, 48)) for dx in (0, 5, 10)])
    first = record([22., 22.], [5, 0], 0, (48, 64))
    second = {"start": 1, "lag": 1, "source_points": [], "supports": {}}
    result, arrays = inspect_reference_triplet(frames, np.zeros((0, 48, 64), bool), first, second)
    best = result["rankings"][0]["ranked_path_indices"][0]
    assert result["unique_supports"] == 1 and result["score"] is None
    np.testing.assert_array_equal(arrays["total_offset"][best], [10, 0])
