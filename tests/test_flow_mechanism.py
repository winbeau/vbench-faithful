import numpy as np
import pytest

from dynamic_degree.local_trajectory import LocalTrajectoryConfig, plan_windows
from scripts.counterfactual.analyze_flow_mechanism import (
    dense_magnitude_summary, field_frequency, interval_owners, spatial_predictability, temporal_diagnostics,
)


def paths(xy, timestamps, reliable=True):
    windows, _ = plan_windows(timestamps, LocalTrajectoryConfig())
    result = {"queries": xy[0], "windows": np.array(windows)}
    for k, (a, b) in enumerate(windows):
        result[f"window_{k}_tracks"] = xy[a:b]
        result[f"window_{k}_reliable_pair"] = np.full((b - a - 1, xy.shape[1]), reliable, bool)
    return result


def test_interval_ownership_every_start_phase_once_and_matches_lag1():
    t = np.arange(16) / 8
    windows, owner = plan_windows(t, LocalTrajectoryConfig())
    assert np.array_equal(owner, interval_owners(t, windows, 1))
    for lag in range(1, 5):
        selected = interval_owners(t, windows, lag)
        assert len(selected) == len(t) - lag
        assert (selected >= 0).all()
        for start, k in enumerate(selected):
            a, b = windows[k]
            assert a <= start < start + lag < b


def test_uncovered_intervals_not_silently_omitted_or_assigned():
    owner = interval_owners(np.arange(10.), [(0, 4), (6, 10)], 2)
    assert owner.tolist() == [0, 0, -1, -1, -1, -1, 1, 1]


@pytest.mark.parametrize("span", [0, -1, 16, 1.5])
def test_invalid_interval_rejected(span):
    with pytest.raises(ValueError):
        interval_owners(np.arange(16.), [(0, 16)], span)


def test_persistent_translation_has_unit_efficiency_and_correct_physical_speed():
    t = np.arange(16) / 8
    xy = np.zeros((16, 5, 2)); xy[:, :, 0] = t[:, None] * 7
    d = temporal_diagnostics(paths(xy, t), t, 100.)
    for r in d["lags"].values():
        assert r["all"]["net_to_arc"] == pytest.approx(1.)
        assert r["all"]["endpoint_speed"] == pytest.approx(.07)
        assert r["reliable_only"]["point_interval_fraction"] == 1.
    assert d["adjacent_velocity"]["all"]["weighted_cosine"] == pytest.approx(1.)


def test_periodic_real_motion_and_periodic_jitter_are_not_temporally_separable():
    # A physical translating oscillator and a texture displacement can produce
    # exactly the same trajectories. Diagnostics must NOT invent class labels.
    t = np.arange(16) / 8
    xy = np.zeros((16, 5, 2)); xy[:, :, 0] = ((-1.) ** np.arange(16))[:, None] * 3
    d = temporal_diagnostics(paths(xy, t), t, 100.)
    assert d["adjacent_velocity"]["all"]["weighted_cosine"] == pytest.approx(-1.)
    assert d["lags"]["2"]["all"]["net_to_arc"] == 0.
    assert d["lags"]["3"]["all"]["net_to_arc"] == pytest.approx(1 / 3)
    assert d["lags"]["4"]["all"]["net_to_arc"] == 0.
    assert "score" not in d and "class" not in d


def test_unreliable_subset_is_null_not_zero_or_stable():
    t = np.arange(16) / 8
    xy = np.zeros((16, 2, 2)); xy[:, :, 0] = t[:, None]
    d = temporal_diagnostics(paths(xy, t, False), t, 100.)
    for r in d["lags"].values():
        assert r["reliable_only"]["endpoint_speed"] is None
        assert r["reliable_only"]["net_to_arc"] is None
        assert r["reliable_only"]["point_interval_fraction"] == 0


def test_zero_motion_has_no_defined_direction_or_path_efficiency():
    t = np.arange(16) / 8
    xy = np.zeros((16, 2, 2))
    d = temporal_diagnostics(paths(xy, t), t, 100.)
    assert d["lags"]["1"]["all"]["arc_speed"] == 0
    assert d["lags"]["2"]["all"]["net_to_arc"] is None
    assert d["adjacent_velocity"]["all"]["weighted_cosine"] is None


def test_spatial_affine_prediction_does_not_include_center():
    y, x = np.mgrid[:7, :7]
    q = np.stack((x, y), axis=-1).reshape(-1, 2).astype(float)
    v = np.zeros((2, len(q), 2)); v[:, len(q) // 2, 0] = 20
    result = spatial_predictability(v, q, np.ones(v.shape[:2], bool), 7., radii=(None,))
    # A single moving point cannot be reconstructed from its static neighbors.
    assert result["global"]["all"]["explained_energy"] < 0
    assert result["global"]["all"]["tested_point_pair_fraction"] == 1.


def test_both_affine_real_motion_and_affine_jitter_have_full_spatial_support():
    y, x = np.mgrid[:7, :7]
    q = np.stack((x, y), axis=-1).reshape(-1, 2).astype(float)
    pattern = q @ np.array([[1., .2], [-.1, 2.]]) + .5
    v = np.stack((pattern, -pattern))
    result = spatial_predictability(v, q, np.ones(v.shape[:2], bool), 7., radii=(.4, None))
    for r in result.values():
        assert r["all"]["explained_energy"] == pytest.approx(1.)
        assert r["reliable_only"]["explained_energy"] == pytest.approx(1.)


def test_spatial_missingness_is_explicit():
    q = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 2.]])
    v = np.ones((2, len(q), 2))
    r = spatial_predictability(v, q, np.zeros(v.shape[:2], bool), 3., radii=(None,))
    assert r["global"]["reliable_only"]["explained_energy"] is None
    assert r["global"]["reliable_only"]["tested_point_pair_fraction"] == 0


@pytest.mark.parametrize("steps", [15, 16])
def test_frequency_parseval_and_dc_accounted_for_odd_and_even_lengths(steps):
    rng = np.random.default_rng(4)
    v = rng.normal(size=(steps, 6, 2))
    t = np.arange(steps + 1) / 8
    result = field_frequency(v, t)
    assert result["mean_square_velocity"] == pytest.approx(np.mean(np.sum((8 * v) ** 2, axis=(1, 2))))
    assert sum(result["energy_fraction"]) == pytest.approx(1.)
    constant = field_frequency(np.ones_like(v), t)
    assert constant["dc_energy_fraction"] == pytest.approx(1.)


def test_frequency_irregular_samples_are_not_silently_resampled():
    t = np.array([0., .1, .21, .3])
    assert field_frequency(np.ones((3, 2, 2)), t)["status"] == "not_computed_irregular_sampling"


def test_spatial_and_temporal_unit_scaling():
    t = np.arange(16) / 8
    xy = np.zeros((16, 2, 2)); xy[:, :, 0] = t[:, None]
    a = temporal_diagnostics(paths(xy, t), t, 100.)
    b = temporal_diagnostics(paths(3 * xy, t), t, 300.)
    assert a == b


def test_dense_summary_detects_a_grid_that_misses_small_motion():
    field = np.zeros((3, 20, 20, 2)); field[:, 9:11, 9:11, 0] = 10
    sampled = np.zeros((3, 4, 2))
    r = dense_magnitude_summary(field, sampled)
    assert r["dense_mean"] == pytest.approx(.1)
    assert r["sampled_grid_mean"] == 0
    assert r["dense_top5_mean"] == pytest.approx(2.)
    assert r["dense_max"] == 10


def test_dense_summary_does_not_call_small_flow_static_truth():
    r = dense_magnitude_summary(np.zeros((3, 4, 4, 2)), np.zeros((3, 4, 2)))
    assert r["dense_mean"] == 0
    assert "static" not in r and "score" not in r
