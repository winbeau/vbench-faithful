import numpy as np
import pytest

from dynamic_degree.structural_decomposition import DecompositionConfig, decompose_tracks, smooth_trajectories


def grid():
    return np.stack(np.meshgrid(np.linspace(16, 240, 12), np.linspace(16, 240, 12)), axis=-1).reshape(-1, 2)


def tracks(movement):
    return grid()[None] + np.asarray(movement)


def test_constant_velocity_and_irregular_time_are_preserved_exactly():
    times = np.cumsum(np.random.default_rng(6).uniform(.1, .16, 12))
    xy = tracks(times[:, None, None] * np.array([15, -3]))
    result = decompose_tracks(xy, times, 256, DecompositionConfig())
    assert np.allclose(result["full_deltas"], np.diff(xy, axis=0), atol=1e-9)
    assert np.allclose(smooth_trajectories(xy, times, .2), xy, atol=1e-9)
    assert set(result["lag_speeds"]) == {"1", "2", "3", "4"}


@pytest.mark.parametrize("period", [2, 3, 4, 8])
def test_coherent_true_periodic_motion_is_not_erased(period):
    times = np.arange(16) / 8
    phase = np.cos(2 * np.pi * np.arange(16) / period)
    xy = tracks(phase[:, None, None] * np.array([6., 1.]))
    result = decompose_tracks(xy, times, 256, DecompositionConfig())
    assert np.allclose(result["full_deltas"], result["raw_deltas"], atol=1e-7)
    assert np.sum(abs(result["temporal_only_deltas"])) < np.sum(abs(result["raw_deltas"]))


def test_incoherent_reversing_residual_suppressed_but_linear_motion_retained():
    rng = np.random.default_rng(5)
    times = np.arange(16) / 8
    nuisance = rng.normal(size=(1, 144, 2)) * 3
    movement = times[:, None, None] * np.array([12., 0.])
    phase = (-1.) ** np.arange(16)
    xy = tracks(movement + phase[:, None, None] * nuisance)
    result = decompose_tracks(xy, times, 256, DecompositionConfig())
    assert np.mean(np.linalg.norm(result["full_deltas"], axis=-1)) < .5 * np.mean(np.linalg.norm(result["raw_deltas"], axis=-1))
    # It is not a net-displacement or constant-zero metric.
    assert np.mean(result["full_deltas"][..., 0]) > 1


def test_uncertain_or_small_region_is_not_automatically_suppressed():
    times = np.arange(16) / 8
    movement = np.zeros((16, 144, 2))
    movement[:, 0, 0] = (-1.) ** np.arange(16) * 4
    colors = np.zeros((144, 3))
    colors[0] = 1  # Appearance-isolated small object; not enough matching neighbors.
    result = decompose_tracks(tracks(movement), times, 256, DecompositionConfig(), colors=colors)
    assert np.array_equal(result["residual_gate"][:, 0], np.ones(15))
    assert np.allclose(result["full_deltas"][:, 0], result["raw_deltas"][:, 0])
    uncertain = decompose_tracks(tracks(movement), times, 256, DecompositionConfig(), reliable=np.zeros((15, 144), bool))
    assert np.allclose(uncertain["full_deltas"], uncertain["raw_deltas"])


def test_affine_periodic_motion_preserves_rotation_not_just_translation():
    positions = grid() - 128
    displacement = np.stack((-positions[:, 1], positions[:, 0]), axis=-1) * .05
    xy = tracks(((-1.) ** np.arange(16))[:, None, None] * displacement)
    result = decompose_tracks(xy, np.arange(16) / 8, 256, DecompositionConfig())
    assert np.allclose(result["full_deltas"], result["raw_deltas"], atol=1e-7)


def test_spatial_units_and_phase_do_not_change_conclusion():
    rng = np.random.default_rng(9)
    offset = rng.normal(size=(144, 2))
    phases = (-1.) ** np.arange(16)
    xy = tracks(phases[:, None, None] * offset)
    a = decompose_tracks(xy, np.arange(16) / 8, 256, DecompositionConfig())
    b = decompose_tracks(xy * 2, np.arange(16) / 8, 512, DecompositionConfig())
    c = decompose_tracks(tracks(-phases[:, None, None] * offset), np.arange(16) / 8, 256, DecompositionConfig())
    assert np.allclose(a["full_deltas"] * 2, b["full_deltas"], atol=1e-8)
    assert np.allclose(a["full_deltas"], -c["full_deltas"], atol=1e-8)


def test_invalid_evidence_fails_not_silently_smoothed():
    with pytest.raises(ValueError):
        smooth_trajectories(np.zeros((4, 3, 2)), [0, 1, 1, 2], .2)
    with pytest.raises(ValueError):
        DecompositionConfig(temporal_scale_seconds=0)
