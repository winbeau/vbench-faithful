import numpy as np
import pytest

from dynamic_degree.common_mode import CommonModeConfig, decompose_common_modes
from dynamic_degree.structural_decomposition import smooth_trajectories


def grid():
    return np.stack(np.meshgrid(np.linspace(16, 240, 12), np.linspace(16, 240, 12)), axis=-1).reshape(-1, 2)


def evaluate(movement, *, times=None, reliable=None):
    times = np.arange(len(movement)) / 8 if times is None else times
    return decompose_common_modes(grid()[None] + movement, times, 256, CommonModeConfig(), reliable=reliable)


def test_common_mode_preserves_linear_motion_at_irregular_times():
    times = np.cumsum(np.random.default_rng(2).uniform(.10, .16, 16))
    out = evaluate(times[:, None, None] * np.array([13., -7.]), times=times)
    assert np.allclose(out["raw_deltas"], out["full_deltas"], atol=1e-10)
    assert out["modes"] == []


@pytest.mark.parametrize("period", [2, 3, 4, 8])
@pytest.mark.parametrize("kind", ["translation", "rotation", "parallax", "two_parts", "small_part"])
def test_common_mode_preserves_periodic_structural_controls(period, kind):
    points = grid()
    field = np.tile([5., 0.], (len(points), 1))
    if kind == "rotation":
        field = np.stack((points[:, 1] - 128, 128 - points[:, 0]), axis=-1) * .05
    elif kind == "parallax":
        field *= np.random.default_rng(3).uniform(.2, 2, (len(points), 1))
    elif kind == "two_parts":
        field[points[:, 0] < 128] *= -1
    elif kind == "small_part":
        field[4:] = 0
    phase = np.cos(2 * np.pi * np.arange(16) / period)
    out = evaluate(phase[:, None, None] * field)
    assert np.allclose(out["full_deltas"], out["raw_deltas"], atol=1e-8), out["modes"]


def test_common_mode_suppresses_distributed_reversal_but_keeps_drift():
    rng = np.random.default_rng(12)
    field = rng.normal(size=(144, 2)) * 4
    times = np.arange(16) / 8
    phase = np.cos(2 * np.pi * np.arange(16) / 3)
    out = evaluate(times[:, None, None] * np.array([2., 0.]) + phase[:, None, None] * field)
    assert any(m["removed"] for m in out["modes"])
    assert np.linalg.norm(out["full_deltas"], axis=-1).sum() < .25 * np.linalg.norm(out["raw_deltas"], axis=-1).sum()
    assert np.mean(out["full_deltas"][..., 0]) > .15


def test_common_mode_does_not_certify_unreliable_motion():
    movement = ((-1.) ** np.arange(16))[:, None, None] * np.random.default_rng(8).normal(size=(144, 2)) * 4
    out = evaluate(movement, reliable=np.zeros((15, 144), bool))
    assert np.array_equal(out["full_deltas"], out["raw_deltas"])
    assert all("insufficient_correspondence_or_neighborhood" in m["retention_reasons"] for m in out["modes"])
    assert "score" not in out


def test_nuisance_cannot_remove_camera_motion_with_the_same_waveform():
    points = grid()
    design = np.c_[np.ones(len(points)), (points - 128) / 256]
    field = np.random.default_rng(17).normal(size=(144, 2)) * 5
    field -= design @ np.linalg.lstsq(design, field, rcond=None)[0]
    camera = np.tile([2., -1.], (len(points), 1))
    phase = (-1.) ** np.arange(16)
    out = evaluate(phase[:, None, None] * (camera + field))
    assert any(m["removed"] for m in out["modes"])
    camera_deltas = np.diff(phase)[:, None, None] * camera[None]
    # Suppression leaves the common affine signal exactly; the time-trend part
    # of the nuisance may still remain, so this is not a claim of full recovery.
    for step in range(15):
        projection = design @ np.linalg.lstsq(design, out["full_deltas"][step], rcond=None)[0]
        assert np.allclose(projection, camera_deltas[step], atol=1e-8)


def test_common_mode_spatial_units_phase_rotation_and_time_reversal():
    field = np.random.default_rng(11).normal(size=(144, 2)) * 4
    xy = grid()[None] + ((-1.) ** np.arange(16))[:, None, None] * field
    times = np.arange(16) / 8
    cfg = CommonModeConfig()
    out = decompose_common_modes(xy, times, 256, cfg)
    double = decompose_common_modes(xy * 2, times, 512, cfg)
    reverse = decompose_common_modes(xy[::-1], times, 256, cfg)
    rotation = np.array([[0., -1.], [1., 0.]])
    rotated = decompose_common_modes(xy @ rotation, times, 256, cfg)
    assert np.allclose(out["full_deltas"] * 2, double["full_deltas"], atol=1e-8)
    assert np.allclose(out["full_deltas"], -reverse["full_deltas"][::-1], atol=1e-8)
    assert np.allclose(out["full_deltas"] @ rotation, rotated["full_deltas"], atol=1e-8)


def test_equal_energy_modes_are_not_classified_from_an_arbitrary_basis():
    times = np.arange(16) / 8
    # Build two equal-energy modes of the exact implemented residual operator.
    basis = np.repeat(np.eye(16)[..., None], 2, axis=-1)
    smooth = smooth_trajectories(basis, times, .2)[..., 0]
    operator = np.diff(np.eye(16) - smooth, axis=0) / np.sqrt(np.diff(times)[:, None])
    _, singular, temporal = np.linalg.svd(operator, full_matrices=False)
    fields, _ = np.linalg.qr(np.random.default_rng(31).normal(size=(288, 2)))
    movement = ((temporal[:2].T / singular[:2]) @ fields.T * 20).reshape(16, 144, 2)
    out = evaluate(movement)
    assert len(out["modes"]) == 2
    assert all("degenerate_mode_basis" in m["retention_reasons"] for m in out["modes"])
    assert np.array_equal(out["full_deltas"], out["raw_deltas"])


def test_common_mode_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        CommonModeConfig(minimum_mode_energy_fraction=float("nan"))
    with pytest.raises(ValueError):
        CommonModeConfig(minimum_neighbors=4.5)
    with pytest.raises(ValueError):
        decompose_common_modes(np.zeros((8, 3, 2)), np.arange(8), 0, CommonModeConfig())
    with pytest.raises(ValueError):
        evaluate(np.zeros((8, 144, 2)), reliable=np.ones((8, 144)))
