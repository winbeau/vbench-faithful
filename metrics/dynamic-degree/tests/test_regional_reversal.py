import numpy as np

from dynamic_degree.image_plane_modes import constrained_mode_residual, regional_reversal_ablation, spatial_design


def inputs(size=12):
    y, x = np.mgrid[:size, :size]
    xy = np.c_[x.ravel(), y.ravel()]
    owners = (x // (size // 3)).ravel()
    times = np.arange(16.) / 8
    wave = np.sin(np.arange(15.) * 2.3)
    return xy, owners, times, wave


def test_synchronized_articulation_and_camera_are_preserved_not_called_jitter():
    xy, owners, times, wave = inputs()
    part = np.c_[np.choose(owners, [-2., 1., 3.]) + xy[:, 1] * .1,
                 np.choose(owners, [1., -2., -1.]) - xy[:, 0] * .1]
    drift = np.c_[np.ones(len(xy)) * 4, xy[:, 1] * .2]
    values = drift + wave[:, None, None] * part
    result, arrays = regional_reversal_ablation(values, times, xy, owners, np.ones(values.shape[:2], bool), (12, 12))
    assert result["heldout_prediction"]["gain"] > .99
    assert not result["retention_reasons"]  # passes common-mode gates, protected by geometry
    assert not result["subtracted"] and result["score"] is None
    np.testing.assert_array_equal(arrays["corrected_velocity"], values)


def test_nonaffine_component_can_be_reduced_without_changing_camera_or_mean_velocity():
    xy, owners, times, wave = inputs()
    field = np.c_[np.sin(xy[:, 0] * .7) * np.cos(xy[:, 1] * .6), np.cos(xy[:, 0] * .5 + xy[:, 1] * .7)]
    values = wave[:, None, None] * field + [2., -1.]
    result, arrays = regional_reversal_ablation(values, times, xy, owners, np.ones(values.shape[:2], bool), (12, 12))
    assert result["subtracted"] and result["score"] is None
    assert result["conditional_guarded_speed"] < result["conditional_raw_speed"]
    removal = arrays["removal"]
    np.testing.assert_allclose(removal.mean(axis=0), 0, atol=1e-8)
    design = spatial_design(xy, (12, 12), 0)
    for region in np.unique(owners):
        points = owners == region
        for t in range(len(values)):
            np.testing.assert_allclose(design[points].T @ removal[t, points], 0, atol=1e-8)


def test_intersection_constraints_hold_with_irregular_times_and_missing_pairs():
    rng = np.random.default_rng(20)
    xy, owners, _, _ = inputs()
    times = np.cumsum(np.r_[0., rng.uniform(.08, .2, 15)])
    known = rng.random((15, len(xy))) > .2
    component = rng.normal(size=(15, len(xy), 2))
    removal, result = constrained_mode_residual(component, times, xy, owners, known, (12, 12))
    assert result["converged"]
    assert np.count_nonzero(removal[~known]) == 0
    np.testing.assert_allclose(np.sum(removal * np.diff(times)[:, None, None], axis=0), 0, atol=1e-8)
    design = spatial_design(xy, (12, 12), 0)
    for t in range(15):
        for region in np.unique(owners):
            points = known[t] & (owners == region)
            np.testing.assert_allclose(design[points].T @ removal[t, points], 0, atol=1e-7)


def test_missing_evidence_does_not_become_success_and_small_part_is_protected():
    xy, owners, times, wave = inputs()
    owners[:2] = 99
    values = wave[:, None, None] * np.sin(xy)[None]
    reliable = np.ones(values.shape[:2], bool)
    removal, projected = constrained_mode_residual(values, times, xy, owners, reliable, (12, 12))
    assert projected["converged"]
    assert np.count_nonzero(removal[:, :2]) == 0
    reliable[:, len(xy) // 2:] = False
    result, arrays = regional_reversal_ablation(values, times, xy, owners, reliable, (12, 12))
    assert "insufficient_pair_evidence" in result["retention_reasons"]
    assert not result["subtracted"] and result["score"] is None
    np.testing.assert_array_equal(arrays["corrected_velocity"], values)


def test_nonconvergence_is_exposed_not_silently_accepted():
    xy, owners, times, wave = inputs()
    rng = np.random.default_rng(41)
    values = rng.normal(size=(15, len(xy), 2))
    known = rng.random(values.shape[:2]) > .3
    _, result = constrained_mode_residual(values, times, xy, owners, known, (12, 12), iterations=1)
    assert not result["converged"]


def test_time_varying_small_region_support_stays_exactly_protected():
    xy, owners, times, _ = inputs()
    rng = np.random.default_rng(19)
    values = rng.normal(size=(15, len(xy), 2))
    known = np.ones(values.shape[:2], bool)
    small = np.flatnonzero(owners == 0)
    known[::2, small[3:]] = False
    removal, result = constrained_mode_residual(values, times, xy, owners, known, (12, 12))
    assert result["converged"]
    assert result["removable_pairs"] < result["observed_pairs"]
    assert np.count_nonzero(removal[::2, small[:3]]) == 0
    np.testing.assert_allclose(np.sum(removal * np.diff(times)[:, None, None], axis=0), 0, atol=1e-8)
