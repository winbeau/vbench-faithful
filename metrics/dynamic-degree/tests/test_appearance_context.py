import cv2
import numpy as np
import pytest

from dynamic_degree.appearance_context import context_masks, evidence, rankings


def test_witnesses_are_disjoint_and_do_not_cross_region_boundaries():
    mask = np.zeros((80, 96), bool); mask[10:70, 12:80] = True
    core, (inner, outer) = context_masks(mask, [48, 40], 2, 4)
    assert not (core & inner).any() and not (inner & outer).any() and not (core & outer).any()
    assert not ((core | inner | outer) & ~mask).any()
    assert min(core.sum(), inner.sum(), outer.sum()) > 0


def test_same_visible_pixels_are_used_for_warp_and_zero_control():
    rng = np.random.default_rng(12)
    a, b = rng.random((2, 31, 36, 3))
    mask = np.zeros((31, 36), bool); mask[5:25, 7:34] = True
    d = np.array([[4, -1], [-2, 3], [40, 40]])
    got = evidence(a, b, mask, d, batch=1)
    yy, xx = np.nonzero(mask)
    for index, (dx, dy) in enumerate(d[:2]):
        tx, ty = xx + dx, yy + dy
        keep = (tx >= 0) & (tx < 36) & (ty >= 0) & (ty < 31)
        source = a[yy[keep], xx[keep]]
        for name, target in (("", b[ty[keep], tx[keep]]), ("zero_", b[yy[keep], xx[keep]])):
            sa, sb = source - source.mean(0), target - target.mean(0)
            expected = np.sum(sa * sb) / np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
            assert got[name + "ncc"][index] == pytest.approx(expected, abs=1e-12)
            assert got[name + "mse"][index] == pytest.approx(np.mean((source - target) ** 2), abs=1e-12)
    assert got["overlap"][2] == 0 and np.isnan(got["ncc"][2]) and np.isnan(got["zero_mse"][2])


def test_disjoint_unique_context_can_disambiguate_a_periodic_seed_template():
    source = np.random.default_rng(7).random((128, 128, 3))
    yy, xx = np.indices(source.shape[:2]); periodic = (xx - 64) ** 2 + (yy - 64) ** 2 <= 18 ** 2
    stripes = .5 + .3 * np.cos(2 * np.pi * xx / 4)
    source[periodic] = stripes[periodic, None]
    target = cv2.warpAffine(source, np.array([[1., 0., 8.], [0., 1., 0.]]), (128, 128))
    core, contexts = context_masks(np.ones((128, 128), bool), [64, 64], 2, 4)
    offsets = np.array([[0, 0], [8, 0]])
    seed = evidence(source, target, core, offsets)
    witnesses = [evidence(source, target, mask, offsets) for mask in contexts]
    np.testing.assert_allclose(seed["ncc"], [1, 1], atol=1e-12)
    ranked = rankings(seed, witnesses)
    assert ranked[-1]["ranked_indices"][0] == 1 and all(r["score"] is None for r in ranked)


def test_flat_or_empty_witness_is_missing_not_zero_control_promotion():
    flat = np.full((32, 32, 3), .5)
    empty = evidence(flat, flat, np.zeros((32, 32), bool), [[0, 0]])
    constant = evidence(flat, flat, np.ones((32, 32), bool), [[0, 0]])
    assert np.isnan(empty["ncc"][0]) and np.isnan(constant["ncc"][0]) and constant["mse"][0] == 0
    assert all(r["ranked_indices"] == [] and r["score"] is None for r in rankings(constant, [empty, constant]))


def test_no_motion_amplitude_threshold_and_no_video_metadata_inputs():
    source = np.random.default_rng(19).random((64, 64, 3))
    target = cv2.warpAffine(source, np.array([[1., 0., 1.], [0., 1., 0.]]), (64, 64))
    mask = np.zeros((64, 64), bool); mask[8:48, 8:48] = True
    measured = evidence(source, target, mask, [[0, 0], [1, 0]])
    assert measured["ncc"][1] == pytest.approx(1.) and measured["mse"][1] < measured["zero_mse"][1]
