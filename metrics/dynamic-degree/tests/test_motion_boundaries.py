import numpy as np
import pytest

from dynamic_degree.image_plane_modes import affine_field_split, boundary_motion, fit_temporal_field


def grid(size=32):
    y, x = np.mgrid[:size, :size]
    return np.c_[x.ravel(), y.ravel()], x, y


def test_synchronized_parts_are_predictable_but_boundary_jumps_remain():
    xy, x, y = grid()
    owners = (x // 8).astype(int)
    # Four parts share one temporal waveform, but move independently.
    part = np.stack((np.choose(owners, [0., 2., -1., 3.]), y * 0), axis=-1)
    times = np.arange(16.) / 8
    wave = np.sin(np.arange(15.) * 1.3)
    values = wave[:, None, None] * part.reshape(-1, 2)
    field, constant, valid = fit_temporal_field(values, times, wave, np.ones(values.shape[:2], bool))
    np.testing.assert_allclose(field, part.reshape(-1, 2), atol=1e-12)
    np.testing.assert_allclose(constant, 0, atol=1e-12)
    affine, residual = affine_field_split(field, xy, (32, 32), valid)
    for stride in (1, 2):
        stats, _ = boundary_motion(field.reshape(32, 32, 2), owners, valid.reshape(32, 32), stride=stride)
        assert stats["cross_region"]["jump_over_difference_energy"] == pytest.approx(1.)
        assert stats["same_region"]["jump_energy"] < 1e-20
        guarded, _ = boundary_motion(residual.reshape(32, 32, 2), owners, valid.reshape(32, 32), stride=stride)
        assert guarded["cross_region"]["jump_energy"] == pytest.approx(stats["cross_region"]["jump_energy"])
    np.testing.assert_allclose(affine + residual, field, atol=1e-12)


def test_smooth_quadratic_warp_crosses_region_boundaries_without_jumps():
    xy, x, y = grid()
    field = np.stack((x * y / 200, x ** 2 / 100 - y ** 2 / 200), axis=-1)
    owners = x // 8
    for stride in (1, 2, 4):
        stats, _ = boundary_motion(field, owners, np.ones(x.shape, bool), stride=stride)
        assert stats["cross_region"]["observed_stencils"] > 0
        assert stats["cross_region"]["jump_rms"] < 1e-12


def test_camera_component_is_exact_and_missing_is_not_stationary():
    xy, x, y = grid()
    camera = np.c_[1 + xy[:, 0] * .2, -.8 + xy[:, 1] * .1]
    valid = np.ones(len(xy), bool); valid[:32] = False
    field = camera.copy(); field[~valid] = np.nan
    affine, residual = affine_field_split(field, xy, (32, 32), valid)
    np.testing.assert_allclose(affine[valid], camera[valid], atol=1e-12)
    np.testing.assert_allclose(residual[valid], 0, atol=1e-12)
    assert np.isnan(affine[~valid]).all()
    stats, _ = boundary_motion(field.reshape(32, 32, 2), x // 8, np.zeros(x.shape, bool))
    assert stats["cross_region"]["observed_stencils"] == 0
    assert stats["cross_region"]["jump_rms"] is None


def test_temporal_fit_uses_observations_not_unreliable_fill_and_preserves_velocity():
    rng = np.random.default_rng(4)
    times = np.cumsum(np.r_[0., rng.uniform(.08, .2, 12)])
    wave = np.sin(np.arange(12.) * 1.2)
    field, drift = rng.normal(size=(2, 8, 2))
    values = drift + wave[:, None, None] * field
    valid = np.ones((12, 8), bool); valid[:4, 0] = False; valid[:, 1] = False
    values[~valid] = 1e6
    inferred, constant, observed = fit_temporal_field(values, times, wave, valid)
    np.testing.assert_allclose(inferred[observed], field[observed], atol=1e-10)
    np.testing.assert_allclose(constant[observed], drift[observed], atol=1e-10)
    assert np.isnan(inferred[1]).all() and not observed[1]


def test_temporal_fit_does_not_invent_a_mode_from_constant_wave():
    field, constant, observed = fit_temporal_field(np.ones((6, 3, 2)), np.arange(7.), np.ones(6), np.ones((6, 3), bool))
    assert not observed.any()
    assert np.isnan(field).all() and np.isnan(constant).all()


def test_smooth_natural_deformation_is_explicitly_indistinguishable_by_boundary_statistic():
    _, x, y = grid()
    # A physical flexible sheet and a digital warp can have identical fields.
    motion = np.stack((x * y / 50, y ** 2 / 100), axis=-1)
    real, _ = boundary_motion(motion, x // 8, np.ones(x.shape, bool))
    digital, _ = boundary_motion(motion.copy(), x // 8, np.ones(x.shape, bool))
    assert real == digital
    assert "suppress" not in real and "score" not in real
