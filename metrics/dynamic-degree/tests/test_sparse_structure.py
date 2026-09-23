import cv2
import numpy as np
import pytest

from dynamic_degree.sparse_structure import (
    METHODS, extract_features, reciprocal_matches, root_descriptors, source_keys_in_mask,
    identity_triangles, distinct_queries,
)
from dynamic_degree.native_region_motion import sift_features, sift_correspondences


def test_root_normalization_and_zero_descriptors():
    d = np.array([[1, 3], [0, 0]], np.float32)
    np.testing.assert_allclose(root_descriptors(d)[0], np.sqrt([.25, .75]))
    assert not root_descriptors(d)[1].any()
    assert root_descriptors(None) is None
    with pytest.raises(ValueError):
        root_descriptors([[-1, 2]])


@pytest.mark.parametrize("method", METHODS)
def test_identity_matches_keep_coordinates_and_empty_stays_missing(method):
    rng = np.random.default_rng(51)
    frame = rng.integers(0, 256, (128, 128, 3), np.uint8)
    cv2.setNumThreads(1)
    features = extract_features(frame, method)
    matches = reciprocal_matches(features, features)
    assert matches
    assert all(m["source_xy"] == m["target_xy"] for m in matches)
    blank = extract_features(np.zeros_like(frame), method)
    assert reciprocal_matches(blank, blank) == []


def test_sift_baseline_parity_and_no_out_of_frame_clamping():
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, (128, 128, 3), np.uint8)
    old = sift_features(frame)
    new = extract_features(frame, "sift")
    np.testing.assert_array_equal(old["xy"], new["xy"])
    reference = sift_correspondences(old, old)
    result = reciprocal_matches(new, new)
    assert [{k: m[k] for k in reference[0]} for m in result] == reference
    mask = np.ones((5, 5), bool)
    matches = [{"source_key": i, "source_xy": xy} for i, xy in enumerate([[-2, 1], [2, 2], [6, 3]])]
    assert source_keys_in_mask(matches, mask) == [1]


def test_invalid_descriptor_method_and_ratio_fail():
    frame = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ValueError):
        extract_features(frame, "unknown")
    feature = extract_features(frame, "sift")
    with pytest.raises(ValueError):
        reciprocal_matches(feature, feature, ratio=1)
    with pytest.raises(ValueError):
        reciprocal_matches(feature, {**feature, "method": "akaze"})


def test_temporal_identity_does_not_filter_real_reversal_or_require_velocity():
    # Spatial path 0 -> 9 -> 0 is not discarded; identities, not coordinates,
    # determine closure. Non-adjacent third frames are equally eligible.
    pairs = [
        {"start": 0, "lag": 1, "matches": [{"source_key": 4, "target_key": 7}]},
        {"start": 1, "lag": 1, "matches": [{"source_key": 7, "target_key": 3}]},
        {"start": 0, "lag": 2, "matches": [{"source_key": 4, "target_key": 3}]},
    ]
    result = identity_triangles(pairs)
    assert all(r["matches"][0]["supported_without_conflict"] for r in result)
    assert [r["matches"][0]["confirmed_frames"] for r in result] == [[2], [0], [1]]
    pairs[-1]["matches"][0]["target_key"] = 8
    result = identity_triangles(pairs)
    assert result[0]["matches"][0]["conflicting_frames"] == [2]
    assert not result[0]["matches"][0]["supported_without_conflict"]


def test_missing_temporal_evidence_and_duplicate_keys_are_not_certificates():
    pair = {"start": 2, "lag": 3, "matches": [{"source_key": 1, "target_key": 6}]}
    result = identity_triangles([pair])[0]["matches"][0]
    assert not result["supported_without_conflict"]
    assert result["confirmed_frames"] == result["conflicting_frames"] == []
    with pytest.raises(ValueError):
        identity_triangles([pair, pair])
    with pytest.raises(ValueError):
        identity_triangles([{**pair, "matches": pair["matches"] * 2}])


def test_query_seeding_keeps_every_spatial_location_and_groups_orientations():
    q, groups = distinct_queries({"xy": np.array([[4, 5], [2, 3], [4, 5]], np.float32)}, 2)
    np.testing.assert_array_equal(q, [[2, 2, 3], [2, 4, 5]])
    assert groups == [[1], [0, 2]]
    q, groups = distinct_queries({"xy": np.empty((0, 2), np.float32)}, 0)
    assert q.shape == (0, 3) and groups == []
