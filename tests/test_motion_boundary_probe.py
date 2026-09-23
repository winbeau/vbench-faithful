import numpy as np
import pytest

from scripts.counterfactual.probe_motion_boundaries import inspect_boundaries


def test_boundary_probe_retains_synchronized_part_jumps_and_all_modes():
    y, x = np.mgrid[:32, :32]
    xy = np.c_[x.ravel(), y.ravel()]
    owners = (x // 8).ravel()
    wave = np.sin(np.arange(15.) * 1.3)
    field = np.c_[np.choose(owners, [0., 1., -2., 3.]), np.zeros(len(xy))]
    values = wave[:, None, None] * field
    reliable = np.ones((15, len(xy)), bool)
    result, arrays = inspect_boundaries(values, np.arange(16.) / 8, xy, owners, reliable, (32, 32))
    assert result["score"] is None
    assert len(result["modes"]) == 3  # no favorable-mode selection
    assert result["modes"][0]["energy_fraction"] > .999
    assert result["modes"][0]["boundaries"][0]["cross_region"]["jump_over_difference_energy"] == pytest.approx(1.)
    assert set(arrays["timestamps"]) == set(np.arange(16.) / 8)


def test_boundary_probe_zero_flow_does_not_invent_motion_or_score():
    y, x = np.mgrid[:32, :32]
    xy = np.c_[x.ravel(), y.ravel()]
    result, _ = inspect_boundaries(np.zeros((15, 1024, 2)), np.arange(16.) / 8, xy, (x // 8).ravel(),
                                  np.ones((15, 1024), bool), (32, 32))
    assert result["modes"] == [] and result["score"] is None
    with pytest.raises(ValueError, match="equally spaced"):
        wrong = xy.copy(); wrong[1, 0] += 1
        inspect_boundaries(np.zeros((15, 1024, 2)), np.arange(16.) / 8, wrong, (x // 8).ravel(),
                           np.ones((15, 1024), bool), (32, 32))
