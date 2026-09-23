import numpy as np
import pytest

from dynamic_degree.trajectory import (
    TrajectoryConfig, correspondence_evidence, query_grid, score_evidence,
)


def evidence(displacement=1.0, moving_error=0.0, stationary_error=1.0, count=4):
    shape = (3, count)
    return {"displacement": np.full(shape, displacement), "moving_error": np.full(shape, moving_error),
            "stationary_error": np.full(shape, stationary_error), "cycle_error": np.zeros(shape),
            "visible_pair": np.ones(shape, bool), "inside_pair": np.ones(shape, bool)}


def score(e, **kwargs):
    return score_evidence(e, np.arange(4) / 8, 192, 256, TrajectoryConfig(**kwargs))


def test_static_supported_does_not_become_missing_or_tracker_motion():
    result = score(evidence(displacement=2, moving_error=0.1, stationary_error=0.01))
    assert result["score"] == 0 and result["status"] == "succeeded"
    assert result["coverage"] == 1 and result["ablations"]["raw_tracks"] > 0


def test_known_displacement_has_physical_units_and_strict_ladder():
    values = [score(evidence(d))["score"] for d in [1, 2, 4]]
    assert values == pytest.approx([8 / 192, 16 / 192, 32 / 192])


def test_invalid_evidence_not_reported_as_zero():
    result = score(evidence(moving_error=3, stationary_error=3))
    assert result["status"] == "insufficient_evidence" and result["score"] is None


def test_fixed_denominator_does_not_inflate_remaining_points():
    full = evidence()
    sparse = evidence()
    sparse["inside_pair"][:, :2] = False
    assert score(sparse, min_coverage=0.5)["score"] == pytest.approx(score(full)["score"] / 2)


def test_reverse_cycle_blocks_unsupported_motion_but_not_valid_stillness():
    item = evidence()
    item["cycle_error"][:] = 20
    assert score(item)["score"] is None
    item["stationary_error"][:] = 0
    assert score(item)["score"] == 0


def test_noise_floor_ablation_and_reversal_have_no_direction_penalty():
    small = score(evidence(0.2))
    assert small["score"] == 0 and small["ablations"]["verified_correspondence"] > 0
    # Pairwise arc length counts both directions, not net displacement.
    item = evidence()
    assert score(item)["score"] == pytest.approx(8 / 192)


def test_real_patch_evidence_favors_stationary_explanation_for_brightness_change():
    rng = np.random.default_rng(3)
    image = rng.integers(30, 210, (64, 64, 3), dtype=np.uint8)
    frames = np.stack([image, image + 10, image - 10, image])
    config = TrajectoryConfig(grid_size=4, patch_radius=4)
    grid = query_grid(64, 64, config)
    tracks = np.repeat(grid[None], 4, axis=0)
    tracks[1:, :, 0] += 1  # Inject a false tracker displacement.
    prediction = {"tracks": tracks, "reverse_tracks": tracks.copy(),
                  "visible": np.ones(tracks.shape[:2], bool), "reverse_visible": np.ones(tracks.shape[:2], bool)}
    observed = correspondence_evidence(frames, prediction, config)
    result = score_evidence(observed, np.arange(4) / 8, 64, 64, config)
    assert result["score"] == 0 and result["coverage"] == 1


@pytest.mark.parametrize("field,value", [("sample_fps", 0), ("noise_floor_pixels", -1), ("min_coverage", 1.1)])
def test_bad_config_fails(field, value):
    with pytest.raises(ValueError):
        TrajectoryConfig(**{field: value})


def test_evaluator_is_filename_invariant_and_takes_no_pair_metadata(monkeypatch, tmp_path):
    from dynamic_degree import trajectory
    from inspect import signature

    frame = np.random.default_rng(8).integers(20, 230, (64, 64, 3), dtype=np.uint8)
    frames = np.repeat(frame[None], 4, axis=0)
    monkeypatch.setattr(trajectory, "decode_video", lambda video, config: (frames, np.arange(4) / 8, {}))

    class Tracker:
        def track(self, actual_frames, grid):
            assert np.array_equal(actual_frames, frames)
            tracks = np.repeat(grid[None], 4, axis=0)
            return {"tracks": tracks, "reverse_tracks": tracks.copy(),
                    "visible": np.ones(tracks.shape[:2], bool), "reverse_visible": np.ones(tracks.shape[:2], bool)}

    evaluator = trajectory.TrajectoryEvaluator(TrajectoryConfig(grid_size=4), tmp_path, tmp_path, "cpu", tracker=Tracker())
    a = evaluator.evaluate_video(tmp_path / "clean.mp4")
    b = evaluator.evaluate_video(tmp_path / "strong_seed_999.mp4")
    assert a.pop("video") != b.pop("video") and a == b
    assert set(signature(evaluator.evaluate_video).parameters) == {"video", "evidence_path"}


def test_illumination_highpass_preserves_real_translation():
    import cv2

    rng = np.random.default_rng(20)
    image = rng.integers(40, 210, (96, 96, 3), dtype=np.uint8)
    frames = np.stack([cv2.warpAffine(image, np.float32([[1, 0, 2 * t], [0, 1, 0]]),
                                      (96, 96), borderMode=cv2.BORDER_REFLECT) for t in range(4)])
    config = TrajectoryConfig(grid_size=4, illumination_sigma=6)
    grid = query_grid(96, 96, config)
    xy = np.repeat(grid[None], 4, axis=0)
    xy[:, :, 0] += 2 * np.arange(4)[:, None]
    tracks = {"tracks": xy, "reverse_tracks": xy.copy(),
              "visible": np.ones(xy.shape[:2], bool), "reverse_visible": np.ones(xy.shape[:2], bool)}
    observed = correspondence_evidence(frames, tracks, config)
    value = score_evidence(observed, np.arange(4) / 8, 96, 96, config)
    assert value["status"] == "succeeded" and value["score"] > 0.1


def test_bad_highpass_scale_fails():
    with pytest.raises(ValueError, match="illumination scale"):
        TrajectoryConfig(blur_sigma=2, illumination_sigma=1)
