import numpy as np

from dynamic_degree.image_plane_modes import region_partition, leave_region_prediction, temporal_modes, leave_region_time_prediction


def test_overlaps_partition_without_double_counting_or_dropping_uncovered():
    masks = np.zeros((2, 20, 20), bool)
    masks[0, :12, :12] = True; masks[1, :4, :4] = True
    owners = region_partition(masks, np.array([[2, 2], [8, 8], [15, 15]]))
    assert owners.tolist() == [1, 0, -1]


def test_camera_reversals_have_affine_explanation_not_nuisance_label():
    x, y = np.meshgrid(np.arange(10.), np.arange(8.))
    xy = np.c_[x.ravel(), y.ravel()]
    field = np.c_[2 + xy[:, 0] * .1, -.3 + xy[:, 1] * .2]
    values = np.array([1., -1., 1., -1., 1.])[:, None, None] * field[None]
    owners = (xy[:, 0] >= 5).astype(int)
    modes = temporal_modes(values, np.arange(6.) * .125, xy, (8, 10), owners)
    assert modes[0]["energy_fraction"] > .999
    assert modes[0]["reversal_fraction"] == 1
    assert modes[0]["affine_explained_fraction"] > .999
    prediction, _ = leave_region_prediction(values, xy, owners, np.ones(len(xy), bool), (8, 10), grid=0)
    np.testing.assert_allclose(prediction, values, atol=1e-10)


def test_holdout_region_values_do_not_leak_and_missing_not_zero():
    rng = np.random.default_rng(5)
    xy = rng.uniform(1, 18, (70, 2))
    owners = np.arange(70) % 4
    values = rng.normal(size=(5, 70, 2))
    valid = np.ones(70, bool); valid[0] = False
    before, _ = leave_region_prediction(values, xy, owners, valid, (20, 20), grid=3)
    modified = values.copy(); modified[:, owners == 2] += 1000
    after, _ = leave_region_prediction(modified, xy, owners, valid, (20, 20), grid=3)
    np.testing.assert_array_equal(before[:, owners == 2], after[:, owners == 2])
    assert np.isnan(before[:, 0]).all()


def test_region_time_prediction_never_reads_held_region_at_held_times():
    rng = np.random.default_rng(88)
    owners = np.arange(21) % 3
    values = rng.normal(size=(12, 21, 2))
    reliable = np.ones((12, 21), bool)
    times = np.arange(13.) / 8
    before, _ = leave_region_time_prediction(values, times, owners, reliable, rank=2)
    changed = values.copy()
    changed[::3, owners == 1] += 100
    after, _ = leave_region_time_prediction(changed, times, owners, reliable, rank=2)
    np.testing.assert_array_equal(before[::3, owners == 1], after[::3, owners == 1])


def test_shared_camera_waveform_predicts_but_localized_periodic_motion_does_not():
    owners = np.arange(20) % 2
    times = np.arange(16.) / 8
    wave = np.sin(np.arange(15.) * 1.7)
    values = np.zeros((15, 20, 2))
    values[..., 0] = wave[:, None]
    reliable = np.ones((15, 20), bool)
    predicted, _ = leave_region_time_prediction(values, times, owners, reliable, rank=1)
    np.testing.assert_allclose(predicted, values, atol=1e-10)
    values[:, owners == 1] = 0
    predicted, records = leave_region_time_prediction(values, times, owners, reliable, rank=1)
    assert records[0]["temporal_rank"] == 0
    assert np.linalg.norm(predicted[:, owners == 0] - values[:, owners == 0]) > 1
