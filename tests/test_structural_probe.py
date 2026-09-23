import numpy as np
import pytest

from dynamic_degree.local_trajectory import LocalTrajectoryConfig, plan_windows
from dynamic_degree.structural_decomposition import DecompositionConfig
from dynamic_degree.common_mode import CommonModeConfig
from dynamic_degree.trajectory import query_grid
from scripts.counterfactual.probe_structural_decomposition import probe


@pytest.mark.parametrize("variant, decomposition", [("local-support", DecompositionConfig()),
                                                    ("common-mode", CommonModeConfig())])
def test_probe_fixed_ownership_preserves_translation_with_overlap(variant, decomposition):
    config = LocalTrajectoryConfig(grid_size=4)
    times = np.arange(16) / 8
    frames = np.zeros((16, 96, 96, 3), np.uint8)
    queries = query_grid(96, 96, config)
    windows, owner = plan_windows(times, config)
    cache = {"windows": np.asarray(windows), "transition_owner": owner, "queries": queries}
    for i, (start, stop) in enumerate(windows):
        xy = queries[None] + np.arange(stop - start)[:, None, None] * np.array([1., 0.])
        cache.update({f"window_{i}_tracks": xy, f"window_{i}_reverse_tracks": xy.copy(),
                      f"window_{i}_visible": np.ones(xy.shape[:2], bool),
                      f"window_{i}_reverse_visible": np.ones(xy.shape[:2], bool)})
    result = probe(frames, times, cache, config, decomposition, variant=variant)
    for value in result["all_track_diagnostics"].values():
        assert value == pytest.approx(8 / 96)
    assert sorted(sum([w["owned_pairs"] for w in result["windows"]], [])) == list(range(15))
    assert "score" not in result  # Numerical diagnostics are not a formal metric score.


def test_probe_all_unreliable_tracks_stay_explicitly_uncertified():
    config = LocalTrajectoryConfig(grid_size=4)
    frames = np.zeros((4, 64, 64, 3), np.uint8)
    times = np.arange(4) / 8
    queries = query_grid(64, 64, config)
    xy = queries[None] + np.array([0., 1., 0., 1.])[:, None, None] * np.array([2., 0.])
    cache = {"windows": np.array([[0, 4]]), "transition_owner": np.zeros(3, int), "queries": queries,
             "window_0_tracks": xy, "window_0_reverse_tracks": xy.copy(),
             "window_0_visible": np.zeros(xy.shape[:2], bool),
             "window_0_reverse_visible": np.zeros(xy.shape[:2], bool)}
    result = probe(frames, times, cache, config, DecompositionConfig())
    assert result["structurally_tested_fraction"] == 0
    assert result["all_track_diagnostics"]["full"] == pytest.approx(result["all_track_diagnostics"]["raw"])
    assert "score" not in result
