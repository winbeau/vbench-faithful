import numpy as np
import pytest

from dynamic_degree.regional_motion import (RegionMotionConfig, region_weights,
    displacement_table, refine_translation, fit_regions, alignment_error)


def features(seed=2, h=10, w=12, c=12):
    a = np.random.default_rng(seed).normal(size=(h, w, c)).astype(np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def support(a):
    weight = np.zeros(a.shape[:2]); weight[2:-2, 2:-2] = 1
    return weight.reshape(1, -1)


def test_fractional_mask_occupancy_keeps_thin_regions():
    m = np.zeros((1, 16, 24), bool); m[0, 5, 1:20] = True
    r = region_weights(m, (4, 6))
    assert np.count_nonzero(r) == 5
    assert r.sum() == pytest.approx(19 / 16)


def test_identical_structures_have_exact_zero_not_peak_parabola_bias():
    a = features(); r = fit_regions(a, a, support(a))[0]
    assert r['full']['displacement_grid'] == [0, 0]
    assert r['full']['loss'] == pytest.approx(0, abs=1e-14)
    assert r['fold_disagreement_grid'] == 0
    assert all(f['heldout_loss'] == pytest.approx(0, abs=1e-14) for f in r['folds'])


@pytest.mark.parametrize('delta', [(2, 1), (-2, -1), (0, 2)])
def test_joint_motion_translation_and_reverse_preserved(delta):
    a = features(); dx, dy = delta
    b = np.roll(a, (dy, dx), axis=(0, 1))
    first = fit_regions(a, b, support(a))[0]
    second = fit_regions(b, a, support(a))[0]
    assert first['full']['displacement_grid'] == pytest.approx(delta, abs=1e-5)
    assert second['full']['displacement_grid'] == pytest.approx([-dx, -dy], abs=1e-5)
    assert first['fold_disagreement_grid'] < 1e-5


def test_direct_refinement_resolves_fractional_translation():
    h, w = 16, 18
    yy, xx = np.mgrid[:h, :w]
    a = np.stack((xx / w, yy / h, np.ones_like(xx)), axis=-1)
    b = np.stack(((xx - .3) / w, (yy + .4) / h, np.ones_like(xx)), axis=-1)
    r = refine_translation(a, b, support(a)[0], [0, 0], RegionMotionConfig())
    assert r['displacement_grid'] == pytest.approx([.3, -.4], abs=1e-5)


def test_flat_features_are_ambiguous_not_certified_stationary():
    a = np.ones((8, 8, 4), np.float32)
    r = fit_regions(a, a, support(a))[0]
    assert r['full']['rank'] == 0
    assert r['full']['alternative_loss_gap'] == 0


def test_small_missing_support_null_and_not_dropped():
    a = features(); weights = np.zeros((2, a.shape[0] * a.shape[1])); weights[1, 20] = 1
    r = fit_regions(a, a, weights)
    assert len(r) == 2
    assert all(x['full']['displacement_grid'] is None for x in r)


def test_out_of_view_never_becomes_zero_error():
    a = features()
    loss, overlap = alignment_error(a, a, support(a)[0], [100, 100])
    assert loss is None and overlap == 0


def test_full_frame_small_camera_shift_not_pinned_by_boundary_cost():
    h, w = 20, 22
    yy, xx = np.mgrid[:h, :w]
    a = np.stack((xx / w, yy / h, np.ones_like(xx)), axis=-1)
    b = np.stack(((xx - .15) / w, (yy + .2) / h, np.ones_like(xx)), axis=-1)
    r = refine_translation(a, b, np.ones(h * w), [0, 0], RegionMotionConfig())
    assert r['displacement_grid'] == pytest.approx([.15, -.2], abs=1e-5)
    assert 0.8 < r['overlap'] < 1
    # End-to-end normalized descriptors must also respond to subgrid camera
    # motion; previous interior-only tests did not cover this boundary case.
    actual = fit_regions(a, b, np.ones((1, h * w)))[0]
    assert actual['full']['displacement_grid'] == pytest.approx([.15, -.2], abs=.025)


def test_out_of_view_initial_guess_remains_explicitly_unresolved():
    a = features()
    r = refine_translation(a, a, support(a)[0], [100, 100], RegionMotionConfig())
    assert r['displacement_grid'] is None and r['loss'] is None


def test_nonfinite_and_zero_descriptors_rejected():
    a = features(); a[0, 0] = 0
    with pytest.raises(ValueError, match='missing evidence'): fit_regions(a, a, support(a))
    with pytest.raises(ValueError): displacement_table(np.full((16, 16), np.nan), (4, 4))


def test_periodic_return_is_not_removed_by_temporal_net_displacement():
    a = features(); frames = [np.roll(a, d, axis=1) for d in (0, 2, 0)]
    shifts = [fit_regions(first, second, support(a))[0]['full']['displacement_grid']
              for first, second in zip(frames[:-1], frames[1:])]
    assert np.linalg.norm(shifts, axis=-1).sum() == pytest.approx(4)
    assert np.linalg.norm(np.sum(shifts, axis=0)) < 1e-5
