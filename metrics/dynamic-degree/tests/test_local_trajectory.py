from dataclasses import asdict
from inspect import signature
import json
from pathlib import Path

import numpy as np
import pytest

from dynamic_degree import local_trajectory as local
from dynamic_degree.trajectory import TrajectoryConfig


def test_reliability_config_preserves_all_previous_thresholds():
    root = Path(__file__).resolve().parents[3]
    old = TrajectoryConfig.read(root / "configs/dynamic-static-jitter/trajectory.dev-v4-tracker-denoise.json")
    new = local.LocalTrajectoryConfig.read(root / "configs/dynamic-static-jitter/trajectory.local-dev-v1.json")
    assert {k: v for k, v in asdict(new).items() if k in asdict(old)} == asdict(old)


@pytest.mark.parametrize("length", [3, 4, 8, 9, 16, 19, 40, 128])
def test_every_transition_has_exactly_one_fixed_owner(length):
    config = local.LocalTrajectoryConfig()
    times = np.arange(length) / 8
    windows, owner = local.plan_windows(times, config)
    assert len(owner) == length - 1
    assert windows[0][0] == 0 and windows[-1][1] == length
    for pair, index in enumerate(owner):
        start, stop = windows[index]
        assert start <= pair and pair + 1 < stop
    assert np.array_equal(owner, local.plan_windows(times, config)[1])


def test_native_irregular_timing_is_used_and_no_endpoint_dropped():
    times = np.cumsum(np.random.default_rng(7).uniform(.09, .17, 32))
    windows, owner = local.plan_windows(times, local.LocalTrajectoryConfig())
    assert len(windows) > 1 and len(owner) == 31 and min(owner) >= 0


@pytest.mark.parametrize("kwargs", [{"window_seconds": 0}, {"stride_seconds": 1},
                                    {"stride_seconds": np.nan}])
def test_bad_temporal_config_fails(kwargs):
    with pytest.raises(ValueError):
        local.LocalTrajectoryConfig(**kwargs)


def setup_fake(monkeypatch, length=16):
    frames = np.zeros((length, 64, 64, 3), np.uint8)
    frames[:, 0, 0, 0] = np.arange(length)
    monkeypatch.setattr(local, "decode_video", lambda *args: (frames, np.arange(length) / 8, {}))
    calls = []

    class Tracker:
        def track(self, clip, grid):
            calls.append(clip[:, 0, 0, 0].tolist())
            xy = np.repeat(grid[None], len(clip), axis=0)
            return {"tracks": xy, "reverse_tracks": xy.copy(),
                    "visible": np.ones(xy.shape[:2], bool), "reverse_visible": np.ones(xy.shape[:2], bool)}

    def evidence(clip, prediction, config):
        shape = (len(clip) - 1, len(prediction["tracks"][0]))
        return {"displacement": np.full(shape, 2.), "moving_error": np.zeros(shape),
                "stationary_error": np.ones(shape), "cycle_error": np.zeros(shape),
                "visible_pair": np.ones(shape, bool), "inside_pair": np.ones(shape, bool)}

    monkeypatch.setattr(local, "correspondence_evidence", evidence)
    return Tracker(), calls


def test_overlapping_windows_do_not_double_count_path_or_time(monkeypatch, tmp_path):
    tracker, calls = setup_fake(monkeypatch)
    config = local.LocalTrajectoryConfig(grid_size=4)
    evaluator = local.LocalTrajectoryEvaluator(config, tmp_path, tmp_path, "cpu", tracker=tracker)
    result = evaluator.evaluate_video(tmp_path / "arbitrary.mp4", evidence_path=tmp_path / "evidence.npz")
    assert calls == [list(range(8)), list(range(4, 12)), list(range(8, 16))]
    assert result["score"] == pytest.approx(2 * 8 / 64)
    assert result["coverage"] == 1 and result["total_pairs"] == 15 * 16
    saved = np.load(tmp_path / "evidence.npz")
    assert saved["pair_tracks"].shape == (15, 16, 2, 2)
    assert sorted(sum([w["owned_transitions"] for w in result["windows"]], [])) == list(range(15))
    assert "not_jitter_repair" in result["interpretation"]


def test_single_window_equals_old_formula_and_is_filename_invariant(monkeypatch, tmp_path):
    tracker, calls = setup_fake(monkeypatch, length=4)
    config = local.LocalTrajectoryConfig(grid_size=4)
    evaluator = local.LocalTrajectoryEvaluator(config, tmp_path, tmp_path, "cpu", tracker=tracker)
    a = evaluator.evaluate_video(tmp_path / "original.mp4")
    b = evaluator.evaluate_video(tmp_path / "8px_seed_1701.mp4")
    assert a.pop("video") != b.pop("video") and a == b
    assert a["score"] == pytest.approx(16 / 64)
    assert set(signature(evaluator.evaluate_video).parameters) == {"video", "evidence_path"}
