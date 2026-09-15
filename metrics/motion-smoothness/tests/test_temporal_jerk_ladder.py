"""The declared temporal-jerk ladder must be strictly ordered on clean motion.

Plan section 13.3 declares `jerk_0_original > jerk_1_duplicate >
jerk_2_duplicate_skip > jerk_3_local_reverse > jerk_4_multiple`.  The shipped
estimator aggregated a temporal difference of the *already saturated* magnitude
change and never read its direction-change series, so a hold-and-jump and a
local reversal of similar length produced the same ``D_video`` and level 2 and
level 3 tied (or inverted, on a curved trajectory).

These tests build the ladder directly as source-frame index maps on a canonical
constant-velocity trajectory, so they depend only on the declared operation
semantics, not on the counterfactual build scripts.  They pin both the composite
ordering and the specific mechanism (a reversal must be penalised more than a
stall).
"""
from __future__ import annotations

import unittest

import numpy as np

from motion_smoothness.metric import analyze_motion_fields
from motion_smoothness.schemas import MotionField, MotionSmoothnessConfig

HEIGHT, WIDTH = 12, 16
FPS = 8.0
DT = 1.0 / FPS
FRAMES = 16


def _identity(count: int) -> list[int]:
    return list(range(count))


def _reversed_segment(indices: list[int], start: int, stop: int) -> list[int]:
    out = list(indices)
    out[start:stop] = out[start:stop][::-1]
    return out


def ladder_indices(count: int = FRAMES) -> dict[str, list[int]]:
    """The four declared operations as source-index maps (centre = count // 2)."""
    centre = count // 2
    quarter = max(1, count // 4)
    level1 = _identity(count)
    level1[centre] = centre - 1
    level2 = _identity(count)
    level2[centre] = centre - 1
    level2[centre + 1] = centre - 1
    level2[centre + 2] = centre + 3
    level2[centre + 3] = centre + 3
    level3 = _identity(count)
    start = max(0, centre - quarter // 2)
    stop = min(count, start + max(4, quarter))
    level3 = _reversed_segment(level3, start, stop)
    level4 = _identity(count)
    for anchor in (count // 5, count // 2, (4 * count) // 5):
        seg_start = max(0, min(count - 4, anchor - 2))
        level4 = _reversed_segment(level4, seg_start, seg_start + 4)
    return {
        "jerk_0_original": _identity(count),
        "jerk_1_duplicate": level1,
        "jerk_2_duplicate_skip": level2,
        "jerk_3_local_reverse": level3,
        "jerk_4_multiple": level4,
    }


def _position(count: int, step: float = 1.0, profile: str = "constant") -> np.ndarray:
    if profile == "constant":
        per_step = np.full(count, step)
    elif profile == "ramp":
        per_step = np.linspace(0.25 * step, 2.0 * step, count)
    else:  # sinusoid
        per_step = step * (1.0 + 0.75 * np.sin(np.linspace(0.0, 3.0 * np.pi, count)))
    return np.concatenate([np.zeros(1), np.cumsum(per_step)[:-1]])


def _fields(indices: list[int], position: np.ndarray) -> list[MotionField]:
    result = []
    for k in range(len(indices) - 1):
        flow = np.zeros((HEIGHT, WIDTH, 2), dtype=float)
        flow[..., 0] = position[indices[k + 1]] - position[indices[k]]
        result.append(MotionField(flow=flow, dt=DT, source_frame_index=k))
    return result


def ladder_scores(profile: str = "constant", step: float = 1.0) -> dict[str, float]:
    position = _position(FRAMES, step=step, profile=profile)
    return {
        name: analyze_motion_fields(_fields(indices, position))["score"]
        for name, indices in ladder_indices().items()
    }


class TemporalJerkLadderTests(unittest.TestCase):
    def test_declared_ladder_is_strictly_ordered(self):
        for profile in ("constant", "ramp", "sinusoid"):
            for step in (0.5, 1.0, 2.0):
                with self.subTest(profile=profile, step=step):
                    scores = ladder_scores(profile=profile, step=step)
                    order = [
                        "jerk_0_original",
                        "jerk_1_duplicate",
                        "jerk_2_duplicate_skip",
                        "jerk_3_local_reverse",
                        "jerk_4_multiple",
                    ]
                    for high, low in zip(order, order[1:]):
                        self.assertGreater(
                            scores[high], scores[low],
                            f"{profile}/{step}: {high} ({scores[high]:.6f}) "
                            f"must outrank {low} ({scores[low]:.6f})",
                        )
                    self.assertGreater(
                        scores["jerk_2_duplicate_skip"] - scores["jerk_3_local_reverse"],
                        0.02,
                        "level 2 and level 3 must be separated, not tied",
                    )

    def test_direction_change_is_what_separates_reversal_from_stall(self):
        """A reversal outranks a stall only because direction change is scored."""
        reversal = analyze_motion_fields(
            _fields(ladder_indices()["jerk_3_local_reverse"], _position(FRAMES))
        )
        stall = analyze_motion_fields(
            _fields(ladder_indices()["jerk_2_duplicate_skip"], _position(FRAMES))
        )
        self.assertGreater(
            reversal["diagnostics"]["direction_discontinuity"],
            stall["diagnostics"]["direction_discontinuity"],
        )
        self.assertLess(reversal["score"], stall["score"])
        magnitude_only = MotionSmoothnessConfig(magnitude_weight=1.0, direction_weight=0.0)
        reversal_only = analyze_motion_fields(
            _fields(ladder_indices()["jerk_3_local_reverse"], _position(FRAMES)), magnitude_only
        )
        stall_only = analyze_motion_fields(
            _fields(ladder_indices()["jerk_2_duplicate_skip"], _position(FRAMES)), magnitude_only
        )
        self.assertGreaterEqual(
            reversal_only["score"],
            stall_only["score"],
            "without the direction term the reversal is not scored below the stall, "
            "so the direction term is load-bearing",
        )

    def test_constant_velocity_still_scores_one(self):
        for name, indices in ladder_indices().items():
            if name != "jerk_0_original":
                continue
            result = analyze_motion_fields(_fields(indices, _position(FRAMES)))
            self.assertEqual(result["D_video"], 0.0)
            self.assertEqual(result["score"], 1.0)

    def test_weights_reject_degenerate_configurations(self):
        for kwargs in (
            {"magnitude_weight": -0.1},
            {"direction_weight": -0.1},
            {"magnitude_weight": float("nan")},
            {"magnitude_weight": 0.0, "direction_weight": 0.0},
            {"temporal_aggregation": "median"},
            {"top_k": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionSmoothnessConfig(**kwargs)

    def test_shipped_defaults_are_the_localized_direction_variant(self):
        config = MotionSmoothnessConfig()
        self.assertEqual(config.temporal_aggregation, "topk")
        self.assertEqual(config.top_k, 3)
        self.assertEqual((config.magnitude_weight, config.direction_weight), (0.5, 0.5))
        self.assertFalse(config.direction_alignment)
        # The legacy mean+tail variant remains selectable for the plan 13.5
        # ablation and must still order the canonical ladder.
        legacy = MotionSmoothnessConfig(
            temporal_aggregation="mean_tail", magnitude_weight=0.5, direction_weight=0.5
        )
        scores = {
            name: analyze_motion_fields(_fields(indices, _position(FRAMES)), legacy)["score"]
            for name, indices in ladder_indices().items()
        }
        self.assertGreater(scores["jerk_2_duplicate_skip"], scores["jerk_3_local_reverse"])


if __name__ == "__main__":
    unittest.main()
