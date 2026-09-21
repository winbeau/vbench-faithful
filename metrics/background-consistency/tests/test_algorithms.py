import numpy as np
import pytest
import torch

from background_consistency.algorithms import score_views, suppress_foreground, temporal_score


def test_foreground_changes_cannot_leak_through_fixed_scoring_support():
    frames = np.arange(3*4*5*3, dtype=np.uint8).reshape(3, 4, 5, 3)
    masks = np.zeros((3, 4, 5), np.uint8)
    masks[:, 1:3, 1:3] = 1
    edited = frames.copy()
    edited[masks.astype(bool)] = 255 - frames[masks.astype(bool)]
    a, fraction = suppress_foreground(frames, masks)
    b, _ = suppress_foreground(edited, masks)
    assert np.array_equal(a, b)
    assert np.array_equal(a[~masks.astype(bool)], frames[~masks.astype(bool)])
    np.testing.assert_allclose(fraction, .8)
    assert not np.array_equal(frames, a)


def test_union_removes_moving_silhouette_without_cropping_background():
    frames = np.zeros((2, 4, 5, 3), np.uint8)
    masks = np.zeros((2, 4, 5), np.uint8)
    masks[0, 0, 0] = masks[1, 3, 4] = 1
    result, fractions = suppress_foreground(frames, masks, temporal_union=True)
    assert result.shape == frames.shape
    assert np.array_equal(result[0], result[1])
    np.testing.assert_allclose(fractions, .9)


def test_all_pairs_is_permutation_invariant_but_official_has_first_frame_bias():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    y = x[[1, 0, 2, 3]]
    assert temporal_score(x, aggregation="all_pairs") == pytest.approx(.5)
    assert temporal_score(x, aggregation="all_pairs") == temporal_score(y, aggregation="all_pairs")
    assert temporal_score(x) != temporal_score(y)


def test_background_inconsistency_is_detected_and_missing_denominator_stays_fixed():
    clean = torch.ones((4, 3))
    broken = torch.eye(3)[[0, 1, 2, 0]]
    for rule in ("official", "all_pairs"):
        assert temporal_score(clean, aggregation=rule) == pytest.approx(1, abs=1e-6)
        assert temporal_score(broken, aggregation=rule) < .5
        assert temporal_score(clean, aggregation=rule, valid=[False]*4) == 0
    assert temporal_score(clean, aggregation="all_pairs", valid=[True, True, False, False]) == pytest.approx(1/6)


def test_empty_foreground_and_empty_background_have_different_policies():
    features = torch.ones((4, 3))
    views = {k: features for k in ("global", "frame", "union")}
    result = score_views(views, {"frame": np.ones(4), "union": np.zeros(4)})
    assert result["frame_official"] == pytest.approx(1)
    assert result["union_official"] == 0
    assert result["official"] == pytest.approx(1)


@pytest.mark.parametrize("features", [torch.ones(1, 3), torch.zeros(4, 3), torch.full((4, 3), float("nan"))])
def test_reject_degenerate_features(features):
    with pytest.raises(ValueError):
        temporal_score(features)


def test_reject_nonbinary_or_wrong_resolution_masks():
    images = np.zeros((2, 3, 4, 3), np.uint8)
    for mask in (np.ones((2, 3, 4))*.5, np.ones((2, 3, 5))):
        with pytest.raises(ValueError):
            suppress_foreground(images, mask)
