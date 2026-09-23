import numpy as np
import pytest

from dynamic_degree.image_plane_modes import constrained_mode_residual, regional_reversal_ablation, spatial_design
from dynamic_degree.residual_reversal import residual_first_reversal


def geometry():
    yy, xx = np.mgrid[:12, :12]
    xy = np.c_[xx.ravel(), yy.ravel()].astype(float)
    owners = (xx//4).ravel()
    times = np.arange(16.)/8
    return xy, owners, times, np.ones((15, 144), bool)


def test_residual_first_finds_small_common_reversal_hidden_by_large_camera_motion():
    xy, owners, times, known = geometry()
    field = np.c_[np.sin(xy[:, 0]*.7)*np.cos(xy[:, 1]*.6), np.cos(xy[:, 0]*.5+xy[:, 1]*.7)]
    artifact, _ = constrained_mode_residual(np.sin(np.arange(15.)*2.3)[:, None, None]*field,
                                           times, xy, owners, known, (12, 12))
    camera = np.c_[np.linspace(0., 100., 15), np.linspace(0., -20., 15)][:, None, :]
    values = camera + artifact
    old, _ = regional_reversal_ablation(values, times, xy, owners, known, (12, 12))
    assert not old['subtracted']
    result, arrays = residual_first_reversal(values, times, xy, owners, known, (12, 12))
    assert result['extra_subtracted']
    np.testing.assert_allclose(arrays['corrected_velocity'], np.broadcast_to(camera, values.shape), atol=1e-8)


def test_region_translation_rotation_periodic_motion_and_small_parts_are_preserved():
    xy, owners, times, known = geometry()
    owners[:2] = 99
    offsets = owners[:, None] * [1.5, -2.]
    field = offsets + np.c_[xy[:, 1], -xy[:, 0]]
    values = np.sin(np.arange(15.)*2.3)[:, None, None] * field + [4., -1.]
    result, arrays = residual_first_reversal(values, times, xy, owners, known, (12, 12))
    assert not result['subtracted']
    np.testing.assert_array_equal(arrays['corrected_velocity'], values)


def test_slow_nonaffine_change_is_not_removed_just_for_being_nonrigid():
    xy, owners, times, known = geometry()
    values = np.linspace(-1., 1., 15)[:, None, None] * np.sin(xy)[None]
    result, arrays = residual_first_reversal(values, times, xy, owners, known, (12, 12))
    assert not result['subtracted']
    np.testing.assert_array_equal(arrays['corrected_velocity'], values)


def test_unknown_motion_and_uneven_timestamp_constraints_are_preserved():
    xy, owners, _, known = geometry()
    rng = np.random.default_rng(21)
    times = np.r_[0., np.cumsum(rng.uniform(.08, .2, 15))]
    known[::3, :25] = False
    values = np.sin(np.arange(15.)*2.3)[:, None, None]*np.sin(xy)[None]+[3., -2.]
    result, arrays = residual_first_reversal(values, times, xy, owners, known, (12, 12))
    np.testing.assert_array_equal(arrays['corrected_velocity'][~known], values[~known])
    removal = arrays['removal']
    np.testing.assert_allclose(np.sum(removal*np.diff(times)[:, None, None], axis=0), 0, atol=1e-7)
    design = spatial_design(xy, (12, 12), 0)
    for t in range(15):
        for owner in np.unique(owners):
            selected = known[t] & (owners==owner)
            np.testing.assert_allclose(design[selected].T@removal[t, selected], 0, atol=1e-7)
    assert result['physical_artifact_classification'] == 'NOT VERIFIED'


def test_natural_control_report_exposes_cancelling_losses_and_increases():
    from scripts.counterfactual.replay_residual_reversal import natural_change
    rows = [{'base_id': str(i), 'prompt_id': str(i//3),
             'previous_repair': {'score': .5}, 'repair': {'score': .5}} for i in range(63)]
    rows[0]['repair']['score'] = .4
    rows[1]['repair']['score'] = .6
    result = natural_change(rows)
    assert result['mean_change'] == pytest.approx(0)
    assert result['mean_absolute_change'] == pytest.approx(.2/63)
    assert result['largest_drop'] == pytest.approx(-.1)
    assert result['changed_videos'] == ['0', '1']
    assert result['unchanged_within_1e12'] == 61
    with pytest.raises(ValueError, match='all 63'):
        natural_change(rows[:-1])
