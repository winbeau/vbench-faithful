import copy

import numpy as np
import pytest

from dynamic_degree.appearance_context import evidence
from scripts.counterfactual.audit_appearance_context import expected_groups, literal_evidence, verify_group
from scripts.counterfactual.probe_appearance_context import inspect_pair, mask_hash
from dynamic_degree.local_appearance import feature_support


def test_literal_pixel_audit_matches_warp_and_zero_controls():
    source, target = np.random.default_rng(24).random((2, 28, 32, 3))
    mask = np.zeros((28, 32), bool); mask[4:24, 5:29] = True
    d = np.array([[0, 0], [4, -2], [99, 0]])
    computed = evidence(source, target, mask, d)
    for i, displacement in enumerate(d):
        checked = literal_evidence(source, target, mask, displacement)
        for name, value in checked.items():
            np.testing.assert_allclose(value, computed[name][i], atol=1e-12, equal_nan=True)


def fixture():
    frame = np.random.default_rng(92).integers(0, 255, (48, 48, 3), np.uint8)
    key = mask_hash(feature_support(np.ones((48, 48), bool), [24., 24.], 2., 4))
    pair = {"start": 0, "lag": 1, "seconds": .125, "source_points": [{"xy": [24., 24.], "size": 2., "regional_supports": [
        {"region": 0, "factor": 4, "whole_frame_control": True, "support_sha256": key}]}],
        "supports": {key: {"hypotheses": [{"displacement_pixels": [0, 0], "correlation": 1.}]}}}
    images = np.stack([frame, frame]); masks = np.zeros((0, 48, 48), bool)
    record, arrays = inspect_pair(images, masks, pair)
    geometry = list(expected_groups(pair, masks, (48, 48)).values())[0]
    return record["groups"][0], arrays, pair["supports"][key], geometry, images.astype(float) / 255.


def test_audit_rejects_changed_witness_geometry_or_missing_rank():
    row, arrays, prior, geometry, images = fixture()
    assert verify_group(arrays, row, prior, geometry, images, 0, 0)[0] == 1
    wrong = copy.deepcopy(row); wrong["support_pixels"][1] += 1
    with pytest.raises(ValueError, match="area mismatch"):
        verify_group(arrays, wrong, prior, geometry, images, 0, 0)
    wrong = copy.deepcopy(row); wrong["rankings"][1]["ranked_indices"] = []
    with pytest.raises(ValueError, match="ranking"):
        verify_group(arrays, wrong, prior, geometry, images, 0, 0)


def test_audit_rejects_corrupt_winner_pixels_even_when_ranks_unchanged():
    row, arrays, prior, geometry, images = fixture()
    arrays["inner_mse"][0] = .2
    with pytest.raises(ValueError, match="literal pixel mismatch"):
        verify_group(arrays, row, prior, geometry, images, 0, 0)
