import cv2
import numpy as np
import pytest

from dynamic_degree.line_structure import (
    sample_strips, line_features, strip_correlations, reciprocal_lines,
    segment_junctions, junction_correspondences,
)
from dynamic_degree.sparse_structure import identity_triangles


def features(frame, lines):
    lines = np.asarray(lines, float)
    strips, visible = sample_strips(frame, lines, np.ones(len(lines)))
    return {"lines": lines, "strips": strips, "visible": visible}


def test_tiled_ncc_matches_literal_common_pixels_and_float32_is_safe():
    rng = np.random.default_rng(51)
    a, b = rng.random((3, 5, 7, 3)).astype(np.float32), rng.random((4, 5, 7, 3)).astype(np.float32)
    ma, mb = rng.random((3, 5, 7)) > .1, rng.random((4, 5, 7)) > .1
    actual, overlap = strip_correlations(a, ma, b, mb, batch=1, min_overlap=.5)
    for i in range(len(a)):
        for j in range(len(b)):
            common = ma[i] & mb[j]
            x, y = a[i][common].astype(float), b[j][common].astype(float)
            x -= x.mean(axis=0); y -= y.mean(axis=0)
            expected = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
            assert actual[i, j] == pytest.approx(expected, abs=1e-12)
            assert overlap[i, j] == pytest.approx(common.mean())
    with pytest.raises(ValueError):
        strip_correlations(a, ma.astype(int), b, mb)


@pytest.mark.parametrize("shift", [(1, 0), (-9, 4), (0, 0)])
def test_full_range_translation_endpoint_reversal_and_small_motion(shift):
    frame = np.random.default_rng(2).integers(0, 256, (96, 96, 3), np.uint8)
    lines = np.array([[[20, 20], [35, 20]], [[45, 35], [45, 50]], [[22, 62], [38, 69]]], float)
    target = cv2.warpAffine(frame, np.array([[1., 0, shift[0]], [0, 1., shift[1]]]), (96, 96))
    source = features(frame, lines)
    result = reciprocal_lines(source, features(target, (lines + shift)[:, ::-1]))
    assert len(result["matches"]) == len(lines)
    for match in result["matches"]:
        assert match["source_key"] == match["target_key"] and match["reversed"]
        np.testing.assert_allclose(match["endpoint_displacements"], np.tile(shift, (2, 1)))
        assert match["correlation"] == pytest.approx(1, abs=1e-12)


def test_quarter_turn_camera_motion_retains_each_endpoint():
    frame = np.random.default_rng(3).integers(0, 256, (96, 96, 3), np.uint8)
    lines = np.array([[[20, 20], [35, 20]], [[45, 35], [45, 50]], [[22, 62], [38, 69]]], float)
    rotated = np.stack((95 - lines[..., 1], lines[..., 0]), axis=-1)
    result = reciprocal_lines(features(frame, lines), features(np.rot90(frame, -1).copy(), rotated))
    assert len(result["matches"]) == 3
    for match in result["matches"]:
        i = match["source_key"]
        np.testing.assert_allclose(match["endpoint_displacements"], rotated[i] - lines[i])


def test_flat_duplicate_and_missing_competitors_are_not_unique_matches():
    image = np.zeros((64, 64, 3), np.uint8)
    empty = line_features(image)
    assert reciprocal_lines(empty, empty) == {"candidates": [], "matches": []}
    frame = np.random.default_rng(4).integers(0, 256, (64, 64, 3), np.uint8)
    duplicate = features(frame, [[[20, 20], [30, 20]], [[20, 20], [30, 20]]])
    result = reciprocal_lines(duplicate, duplicate)
    assert not result["matches"] and all(c["ratio"] == 1 for c in result["candidates"])
    one = features(frame, [[[20, 20], [30, 20]]])
    assert reciprocal_lines(one, one)["matches"] == []
    assert reciprocal_lines(one, empty)["candidates"][0]["hypotheses"] == []
    assert reciprocal_lines(empty, one)["matches"] == []
    flat = features(image, [[[20, 20], [30, 20]], [[30, 30], [40, 40]]])
    assert all(not c["hypotheses"] for c in reciprocal_lines(flat, flat)["candidates"])


def test_out_of_image_is_explicit_missing_not_black_match():
    image = np.random.default_rng(4).integers(0, 256, (64, 64, 3), np.uint8)
    source = features(image, [[[-4, 20], [4, 20]], [[20, 30], [30, 30]]])
    assert source["visible"][0].mean() < .8
    result = reciprocal_lines(source, source)
    assert not result["candidates"][0]["hypotheses"]
    with pytest.raises(ValueError):
        sample_strips(image, [[[1, 1], [1, 1]]], [1])


def test_junction_can_slide_along_stationary_boundary_and_return():
    fixed = np.array([[0., 20], [60, 20]])
    segments = [np.array([fixed, [[x, 10], [x, 30]]]) for x in (15., 24., 15.)]
    frames = [segment_junctions(lines, [1, 1], (64, 64)) for lines in segments]
    pairs = []
    for start, lag in ((0, 1), (1, 1), (0, 2)):
        result = junction_correspondences(frames[start], frames[start + lag],
                                         [{"source_key": 0, "target_key": 0}, {"source_key": 1, "target_key": 1}])
        assert not result["missing"] and len(result["matches"]) == 1
        pairs.append({"start": start, "lag": lag, **result})
    assert [p["matches"][0]["displacement_pixels"] for p in pairs] == [[9., 0.], [-9., 0.], [0., 0.]]
    assert all(p["matches"][0]["supported_without_conflict"] for p in identity_triangles(pairs))


def test_degenerate_remote_intersections_and_missing_identities_are_not_zero():
    assert segment_junctions([[[1, 1], [5, 1]], [[1, 2], [5, 2]]], [1, 1], (64, 64)) == []
    assert segment_junctions([[[1, 1], [5, 1]], [[10, 0], [10, 5]]], [1, 1], (64, 64)) == []
    src = segment_junctions([[[1, 10], [20, 10]], [[5, 0], [5, 20]]], [1, 1], (32, 32))
    absent = junction_correspondences(src, [], [{"source_key": 0, "target_key": 0}])
    assert absent["matches"] == [] and absent["missing"][0]["reason"] == "missing_line_correspondence"
    matches = [{"source_key": i, "target_key": i} for i in (0, 1)]
    assert junction_correspondences(src, [], matches)["missing"][0]["reason"] == "target_intersection_not_visible"
    with pytest.raises(ValueError):
        junction_correspondences(src, src, matches * 2)
