import cv2
import numpy as np
import pytest

from dynamic_degree.appearance_paths import direct_ncc, two_hop_candidates, rank_reference_paths, triplet_ncc, reference_template_paths
from dynamic_degree.local_appearance import ncc_surface


def test_direct_ncc_agrees_with_full_surface_on_integer_offsets_and_missing():
    rng = np.random.default_rng(92)
    source = rng.integers(0, 255, (30, 36, 3), np.uint8)
    target = rng.integers(0, 255, source.shape, np.uint8)
    mask = np.zeros(source.shape[:2], bool); mask[7:20, 9:23] = True
    offsets = np.array([[0, 0], [4, -3], [-7, -6], [100, 100]])
    values, overlap = direct_ncc(source, target, mask, offsets)
    full = ncc_surface(source, target, mask)
    for i, d in enumerate(offsets[:3]):
        at = (full["dx"] == d[0]) & (full["dy"] == d[1])
        if overlap[i] >= .8:
            assert values[i] == pytest.approx(full["correlation"][at][0], abs=1e-12)
        else:
            assert np.isnan(values[i])
    assert np.isnan(values[-1]) and overlap[-1] == 0


def pair(points, offsets):
    return {"source_points": [{"xy": list(xy), "size": 2., "regional_supports": [{"support_sha256": str(i)}]}
                              for i, xy in enumerate(points)],
            "supports": {str(i): {"hypotheses": [{"displacement_pixels": list(d), "correlation": .9} for d in choices]}
                         for i, choices in enumerate(offsets)}}


def test_static_and_moving_paths_survive_without_single_region_majority_rule():
    first = pair([[5, 5]], [[[0, 0], [10, 0]]])
    second = pair([[5, 5], [15, 5]], [[[0, 0]], [[10, 0]]])
    paths, counts = two_hop_candidates(first, second)
    np.testing.assert_array_equal(paths["total_offset"], [[0, 0], [20, 0]])
    assert counts[0]["with_middle_detection"] == 2


def test_returning_motion_keeps_both_steps_not_net_displacement_as_a_score():
    first, second = pair([[5, 5]], [[[10, 0]]]), pair([[15, 5]], [[[-10, 0]]])
    paths, _ = two_hop_candidates(first, second)
    np.testing.assert_array_equal(paths["first_offset"], [[10, 0]])
    np.testing.assert_array_equal(paths["total_offset"], [[0, 0]])
    paths["actual_first_ncc"] = np.array([.9]); paths["reference_ncc"] = np.array([1.])
    ranked, merit = rank_reference_paths(paths)
    assert ranked[0]["ranked_path_indices"] == [0] and merit[0] == .9
    assert "score" not in ranked[0]


def test_missing_middle_detection_and_flat_reference_stay_missing():
    paths, counts = two_hop_candidates(pair([[5, 5]], [[[10, 0]]]), pair([[40, 40]], [[[0, 0]]]))
    assert len(paths["total_offset"]) == 0 and counts[0]["without_middle_detection"] == 1
    flat = np.full((20, 20, 3), 100, np.uint8)
    values, _ = direct_ncc(flat, flat, np.ones((20, 20), bool), np.zeros((1, 2)))
    assert np.isnan(values[0])


def test_large_support_is_chunked_not_subsampled_or_rejected():
    frame = np.random.default_rng(8).integers(0, 255, (190, 190, 3), np.uint8)
    values, overlap = direct_ncc(frame, frame, np.ones(frame.shape[:2], bool), np.zeros((1, 2)))
    assert values[0] == pytest.approx(1., abs=1e-12) and overlap[0] == 1.


def test_identical_supports_shared_by_regions_do_not_duplicate_paths():
    first, second = pair([[5, 5]], [[[1, 0]]]), pair([[6, 5]], [[[1, 0]]])
    first["source_points"][0]["regional_supports"] *= 2
    second["source_points"][0]["regional_supports"] *= 2
    paths, counts = two_hop_candidates(first, second)
    assert len(paths["total_offset"]) == 1 and counts[0]["declared_first_hypotheses"] == 1


def test_three_way_ncc_uses_common_visible_pixels_for_each_pair():
    rng = np.random.default_rng(17)
    frames = rng.integers(0, 255, (3, 22, 30, 3), np.uint8)
    support = np.zeros((22, 30), bool); support[4:19, 6:28] = True
    first, total = np.array([[4, -2], [0, 0], [40, 0]]), np.array([[3, 3], [-1, 0], [0, 0]])
    values, overlap = triplet_ncc(frames, support, first, total, min_overlap=.5, batch=1)
    yy, xx = np.nonzero(support)
    for row in range(2):
        coords = [np.c_[xx, yy] + d for d in ([0, 0], first[row], total[row])]
        valid = np.logical_and.reduce([(q[:, 0] >= 0) & (q[:, 0] < 30) & (q[:, 1] >= 0) & (q[:, 1] < 22) for q in coords])
        pixels = [frames[i, q[valid, 1], q[valid, 0]].astype(float) for i, q in enumerate(coords)]
        centered = [v - v.mean(axis=0) for v in pixels]
        expected = [np.sum(centered[i] * centered[j]) / np.sqrt(np.sum(centered[i] ** 2) * np.sum(centered[j] ** 2)) for i, j in ((0, 1), (0, 2), (1, 2))]
        np.testing.assert_allclose(values[row], expected, atol=1e-12)
        assert overlap[row] == pytest.approx(valid.mean())
    assert overlap[2] == 0 and np.isnan(values[2]).all()


@pytest.mark.parametrize("last_offset", [0, 24])
def test_reference_templates_preserve_returning_and_continuing_motion(last_offset):
    source = np.random.default_rng(51).integers(0, 255, (56, 90, 3), np.uint8)
    frames = np.stack([cv2.warpAffine(source, np.array([[1., 0., dx], [0., 1., 0.]]), (90, 56)) for dx in (0, 12, last_offset)])
    support = np.zeros((56, 90), bool); support[20:30, 26:36] = True
    result, arrays = reference_template_paths(frames, support, [{"displacement_pixels": [12, 0]}])
    k = result["ranked_path_indices"][0]
    np.testing.assert_array_equal(arrays["first_offset"][k], [12, 0])
    np.testing.assert_array_equal(arrays["total_offset"][k], [last_offset, 0])
    assert arrays["bottleneck_ncc"][k] == pytest.approx(1.) and result["score"] is None


def test_common_flat_frame_and_noninteger_offsets_cannot_certify_paths():
    source = np.random.default_rng(7).integers(0, 255, (20, 20, 3), np.uint8)
    frames = np.stack([source, np.full_like(source, 128), source])
    ncc, overlap = triplet_ncc(frames, np.ones((20, 20), bool), [[0, 0]], [[0, 0]])
    assert overlap[0] == 1 and np.isnan(ncc[0, [0, 2]]).all() and ncc[0, 1] == pytest.approx(1.)
    with pytest.raises(ValueError, match="integer"):
        triplet_ncc(frames, np.ones((20, 20), bool), [[.5, 0]], [[0, 0]])
