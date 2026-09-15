"""Sampling-lag calibration of the time normalisation.

The counterfactual FPS ladder holds the duration fixed and changes only the
sampling interval.  Dividing a displacement by ``dt`` (the ballistic
normalisation) is exact only for constant-velocity motion; the audit measures
the clip's own lag exponent instead, so the reported intensity does not carry
the sampling interval.  These tests build trajectories whose lag law is known
exactly, sample them at two rates, and check both directions: that the
ballistic field still makes the two rates disagree, and that the lag-normalised
intensity makes them agree.
"""

import unittest

import numpy as np

from dynamic_degree.backends.audit import (
    AuditAblation,
    AuditConfig,
    LagExponentMode,
    analyze_timed_flow_sequence,
    resolve_time_normalization_exponent,
)
from dynamic_degree.motion import (
    fit_power_law_exponent,
    normalized_intensity,
    resolve_lags,
)
from dynamic_degree.schemas import LagScalingEvidence, TimedFrame, TimedFrameSequence

HEIGHT, WIDTH = 24, 32
DIAGONAL = float(np.hypot(HEIGHT, WIDTH))


def trajectory(indices, dt):
    """Frames that carry their own source index, so a provider can read the lag.

    The index difference between two frames is the lag in source frames; the
    motion law itself is supplied by ``flow_provider`` and is therefore the only
    place a test declares how displacement grows with the lag.
    """
    frames = []
    for position, index in enumerate(indices):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[0, 0, 0] = index
        frames.append(TimedFrame(frame, position * dt, int(index)))
    return TimedFrameSequence(tuple(frames), (HEIGHT, WIDTH), 1.0 / dt, 1, "synthetic")


def flow_provider(amplitude=2.0, exponent=1.0):
    """Uniform flow whose magnitude is ``amplitude * lag ** exponent`` pixels."""

    def compute(left, right):
        lag = int(right[0, 0, 0]) - int(left[0, 0, 0])
        magnitude = amplitude * float(lag) ** exponent
        flow = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float64)
        flow[..., 0] = magnitude
        return flow

    return compute


NATIVE_INDICES = tuple(range(16))      # 8 fps: adjacent sampled frames are 1 source frame apart
COARSE_INDICES = (0, 4, 8, 12)         # 2 fps rung of the same trajectory


def analyze(indices=NATIVE_INDICES, dt=0.125, law=1.0, config=None, amplitude=2.0,
            prompt="a person running"):
    """Sample one trajectory at ``dt`` and score it with a provider using ``law``."""
    sequence = trajectory(indices, dt)
    return analyze_timed_flow_sequence(
        "synthetic.mp4", prompt, sequence, flow_provider(amplitude, law), config=config
    )


def native_clip(**kwargs):
    return analyze(NATIVE_INDICES, 0.125, **kwargs)


def coarse_clip(**kwargs):
    return analyze(COARSE_INDICES, 0.5, **kwargs)


class LagExponentEstimatorTests(unittest.TestCase):
    def test_ballistic_displacements_fit_exponent_one(self):
        fit = fit_power_law_exponent([0.125, 0.25, 0.5], [0.25, 0.5, 1.0])
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit[0], 1.0, places=9)
        self.assertAlmostEqual(fit[1], 0.0, places=9)

    def test_square_root_displacements_fit_exponent_half(self):
        fit = fit_power_law_exponent([0.5, 1.0, 1.5], [2.0, 2.0 * 2**0.5, 2.0 * 3**0.5])
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit[0], 0.5, places=9)

    def test_fit_needs_two_distinct_positive_lags(self):
        self.assertIsNone(fit_power_law_exponent([0.5, 0.5], [1.0, 2.0]))
        self.assertIsNone(fit_power_law_exponent([0.5], [1.0]))
        self.assertIsNone(fit_power_law_exponent([0.5, 0.4], [1.0, 0.0]))

    def test_resolve_lags_clamps_to_clip_length(self):
        self.assertEqual(resolve_lags((1, 2, 4), 16), (1, 2, 4))
        self.assertEqual(resolve_lags((1, 2, 4), 4), (1, 2, 3))
        self.assertEqual(resolve_lags((1, 2, 4), 2), (1,))
        with self.assertRaises(ValueError):
            resolve_lags((0,), 8)

    def test_normalized_intensity_reduces_to_speed_at_exponent_one(self):
        displacement, intensity = normalized_intensity(4.0, (HEIGHT, WIDTH), 0.2, 1.0)
        self.assertAlmostEqual(displacement, 4.0 / DIAGONAL)
        self.assertAlmostEqual(intensity, 4.0 / DIAGONAL / 0.2)

    def test_normalized_intensity_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            normalized_intensity(1.0, (HEIGHT, WIDTH), 0.0, 1.0)
        with self.assertRaises(ValueError):
            normalized_intensity(1.0, (HEIGHT, WIDTH), 0.1, float("nan"))


class MeasuredExponentTests(unittest.TestCase):
    def test_native_clip_measures_ballistic_exponent(self):
        result = native_clip(law=1.0)
        self.assertIsNotNone(result.lag_scaling)
        self.assertAlmostEqual(result.lag_scaling.exponent, 1.0, places=6)
        self.assertEqual(result.lag_scaling.exponent_source, "measured_within_clip_power_law")
        # the default applies the pre-registered exponent, not the per-clip fit
        self.assertEqual(result.time_normalization_exponent, 0.5)
        self.assertIn("diffusive_sqrt_lag_default", result.time_normalization_exponent_source)

    def test_measured_mode_applies_the_fitted_exponent(self):
        config = AuditConfig(lag_exponent_mode=LagExponentMode.MEASURED)
        result = native_clip(law=0.5, config=config)
        self.assertAlmostEqual(result.lag_scaling.exponent, 0.5, places=6)
        self.assertEqual(result.time_normalization_exponent, result.lag_scaling.exponent)
        self.assertEqual(result.time_normalization_exponent_source, "measured_within_clip_power_law")

    def test_default_mode_is_the_fixed_benchmark_exponent(self):
        result = native_clip(law=0.5)
        self.assertEqual(result.time_normalization_exponent, 0.5)
        self.assertIn("diffusive_sqrt_lag_default", result.time_normalization_exponent_source)

    def test_ballistic_mode_restores_the_old_normalisation(self):
        config = AuditConfig(lag_exponent_mode=LagExponentMode.BALLISTIC)
        result = native_clip(law=1.0, config=config)
        self.assertEqual(result.time_normalization_exponent, 1.0)
        self.assertEqual(result.time_normalization_exponent_source, "ballistic_mode")

    def test_non_finite_default_exponent_rejected(self):
        config = AuditConfig(default_lag_exponent=float("inf"))
        with self.assertRaisesRegex(ValueError, "default_lag_exponent"):
            native_clip(law=0.5, config=config)

    def test_native_clip_measures_square_root_exponent(self):
        result = native_clip(law=0.5)
        self.assertAlmostEqual(result.lag_scaling.exponent, 0.5, places=6)
        self.assertEqual(result.lag_scaling.lags, (1, 2, 4))

    def test_straightness_is_one_for_ballistic_motion(self):
        result = native_clip(law=1.0)
        for value in result.lag_scaling.straightness:
            self.assertAlmostEqual(value, 1.0, places=9)

    def test_straightness_separates_diffusive_motion_from_ballistic(self):
        """chord/path = k**(-1/2) is the fingerprint of non-ballistic motion."""
        result = native_clip(law=0.5)
        by_lag = dict(zip(result.lag_scaling.lags, result.lag_scaling.straightness))
        self.assertAlmostEqual(by_lag[1], 1.0, places=9)
        self.assertAlmostEqual(by_lag[2], 2**-0.5, places=6)
        self.assertAlmostEqual(by_lag[4], 4**-0.5, places=6)

    def test_short_clip_cannot_measure_and_keeps_the_benchmark_exponent(self):
        result = analyze(indices=(0, 1), dt=0.5)
        self.assertEqual(result.lag_scaling.exponent_source, "ballistic_default_insufficient_lags")
        self.assertIsNone(result.lag_scaling.exponent)
        self.assertEqual(result.time_normalization_exponent, 0.5)
        self.assertIn("diffusive_sqrt_lag_default", result.time_normalization_exponent_source)


class FpsInvarianceTests(unittest.TestCase):
    """The point of the exercise: two samplings of one trajectory must agree."""

    def test_each_law_is_invariant_under_its_own_exponent(self):
        ballistic = AuditConfig(lag_exponent_mode=LagExponentMode.BALLISTIC)
        self.assertAlmostEqual(
            native_clip(law=1.0, config=ballistic).apparent.motion_intensity,
            coarse_clip(law=1.0, config=ballistic).apparent.motion_intensity,
            places=9,
        )
        self.assertAlmostEqual(
            native_clip(law=0.5).apparent.motion_intensity,
            coarse_clip(law=0.5).apparent.motion_intensity,
            places=6,
        )

    def test_wrong_exponent_is_not_invariant(self):
        """The default 0.5 exponent cannot fix ballistic content, and vice versa."""
        ballistic = AuditConfig(lag_exponent_mode=LagExponentMode.BALLISTIC)
        self.assertNotAlmostEqual(
            native_clip(law=1.0).apparent.motion_intensity,
            coarse_clip(law=1.0).apparent.motion_intensity,
            places=3,
        )
        self.assertNotAlmostEqual(
            native_clip(law=0.5, config=ballistic).apparent.motion_intensity,
            coarse_clip(law=0.5, config=ballistic).apparent.motion_intensity,
            places=3,
        )

    def test_ballistic_normalisation_is_not_invariant_for_diffusive_motion(self):
        """Exponent 1 reproduces the gap measured on real data.

        For ``d ~ lag**0.5`` the displacement grows 2x from 8 to 2 fps while
        ``d/dt`` falls 2x -- the mirror-image pair the counterfactual measured for
        the official raw flow and for the naive ``/dt`` repair.
        """
        native = native_clip(law=0.5)
        coarse = coarse_clip(law=0.5)
        self.assertAlmostEqual(
            coarse.transitions[0].apparent_displacement
            / native.transitions[0].apparent_displacement,
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            coarse.transitions[0].apparent_speed / native.transitions[0].apparent_speed,
            0.5,
            places=6,
        )

    def test_default_exponent_removes_the_sampling_interval(self):
        native = native_clip(law=0.5)
        coarse = coarse_clip(law=0.5)
        self.assertAlmostEqual(native.lag_scaling.exponent, 0.5, places=6)
        self.assertAlmostEqual(coarse.lag_scaling.exponent, 0.5, places=6)
        self.assertAlmostEqual(
            native.apparent.motion_intensity, coarse.apparent.motion_intensity, places=6
        )
        self.assertNotAlmostEqual(
            native.apparent.motion_intensity, native.transitions[0].apparent_speed
        )

    def test_scalar_follows_the_routed_lag_normalised_intensity(self):
        result = native_clip(law=0.5)
        self.assertAlmostEqual(result.score, result.task_relevant_motion_evidence["motion_intensity"])

    def test_coverage_and_threshold_share_the_intensity_domain(self):
        """The static/moving decision no longer mixes a pixel threshold with a
        per-frame displacement: threshold, flags and coverage all use dt**alpha."""
        result = native_clip(law=0.5)
        self.assertEqual(result.threshold.exponent, 0.5)
        self.assertEqual(result.threshold.units, "image_diagonals_per_second_pow_0.5")
        valid = [t for t in result.transitions if t.apparent_intensity is not None]
        expected = sum(
            t.dt_seconds for t in valid if t.apparent_intensity > result.threshold.value
        ) / sum(t.dt_seconds for t in valid)
        self.assertAlmostEqual(result.apparent.temporal_coverage, expected)
        for transition in valid:
            self.assertEqual(
                transition.apparent_significant,
                transition.apparent_intensity > result.threshold.value,
            )

    def test_threshold_is_a_clip_independent_constant(self):
        """It is a config constant in the intensity domain, so it cannot drift
        with the sampling interval the way a per-frame pixel threshold does."""
        native = native_clip(law=0.5)
        coarse = coarse_clip(law=0.5)
        self.assertEqual(native.threshold.value, coarse.threshold.value)
        self.assertEqual(native.threshold.exponent, coarse.threshold.exponent)
        self.assertNotAlmostEqual(
            native.transitions[0].apparent_speed, coarse.transitions[0].apparent_speed
        )


class ExponentProvenanceTests(unittest.TestCase):
    def test_without_lag_scaling_ablation_restores_ballistic(self):
        config = AuditConfig(ablation=AuditAblation.WITHOUT_LAG_SCALING)
        result = native_clip(law=0.5, config=config)
        self.assertEqual(result.time_normalization_exponent, 1.0)
        self.assertEqual(
            result.time_normalization_exponent_source,
            "ballistic_default_without_lag_scaling_ablation",
        )
        self.assertIsNone(result.lag_scaling)

    def test_explicit_exponent_requires_a_source(self):
        with self.assertRaisesRegex(ValueError, "lag_exponent_source"):
            resolve_time_normalization_exponent(
                {"time_normalization": True}, AuditConfig(lag_exponent=0.5), None
            )

    def test_explicit_exponent_is_recorded_with_provenance(self):
        config = AuditConfig(lag_exponent=0.5, lag_exponent_source="dev_calibrated_constant")
        result = native_clip(law=1.0, config=config)
        self.assertEqual(result.time_normalization_exponent, 0.5)
        self.assertEqual(result.time_normalization_exponent_source, "dev_calibrated_constant")

    def test_disabling_lag_scaling_leaves_the_ballistic_default(self):
        config = AuditConfig(lag_scaling_enabled=False)
        result = native_clip(law=0.5, config=config)
        self.assertEqual(result.time_normalization_exponent, 1.0)
        self.assertEqual(result.time_normalization_exponent_source, "lag_scaling_disabled_ballistic_default")
        self.assertIsNone(result.lag_scaling)

    def test_time_normalization_ablation_uses_raw_displacement(self):
        config = AuditConfig(ablation=AuditAblation.WITHOUT_TIME_NORMALIZATION)
        result = native_clip(law=0.5, config=config)
        self.assertEqual(result.time_normalization_exponent, 0.0)
        self.assertEqual(result.time_normalization_exponent_source, "time_normalization_disabled")
        self.assertAlmostEqual(
            result.apparent.motion_intensity, result.transitions[0].apparent_displacement
        )

    def test_evidence_serializes_with_lags_and_fit_quality(self):
        result = native_clip(law=0.5)
        payload = result.lag_scaling.to_dict()
        self.assertEqual(payload["lags_frames"], [1, 2, 4])
        self.assertEqual(len(payload["chord_displacement_diagonals"]), 3)
        self.assertEqual(len(payload["straightness"]), 3)
        self.assertLess(payload["fit_rmse_log_log"], 1e-6)
        self.assertFalse(payload["independently_calibrated"])

    def test_unavailable_evidence_still_serializes(self):
        evidence = LagScalingEvidence(
            exponent=None, exponent_source="ballistic_default_insufficient_lags",
            measured_on_channel="apparent", lags=(1,), lag_seconds=(), chord_displacement=(),
            path_displacement=(), straightness=(), pair_counts=(), fit_rmse=None,
            independently_calibrated=False,
        )
        self.assertIsNone(evidence.to_dict()["exponent"])


if __name__ == "__main__":
    unittest.main()
