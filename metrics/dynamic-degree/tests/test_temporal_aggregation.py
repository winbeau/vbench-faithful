import unittest

from dynamic_degree.diagnostics import aggregate_channel
from dynamic_degree.backends.audit import AuditConfig, derive_motion_threshold
from dynamic_degree.schemas import TransitionEvidence


def transition(speed, dt=1.0, valid=True, intensity=None):
    return TransitionEvidence(
        source_frame_index_a=0,
        source_frame_index_b=1,
        timestamp_a=0.0,
        timestamp_b=dt,
        dt_seconds=dt,
        frame_shape=(64, 64),
        valid=valid,
        invalid_reason=None if valid else "invalid",
        apparent_speed=speed if valid else None,
        global_speed=speed if valid else None,
        residual_speed=speed if valid else None,
        residual_intensity=intensity,
    )


class TemporalAggregationTests(unittest.TestCase):
    def test_default_threshold_provenance_is_explicitly_uncalibrated(self):
        threshold = derive_motion_threshold((720, 1280), AuditConfig(), 1.0)
        self.assertFalse(threshold.independently_calibrated)
        self.assertIn("not_independently_calibrated", threshold.source)

    def test_explicit_threshold_requires_source(self):
        with self.assertRaisesRegex(ValueError, "requires threshold_source"):
            derive_motion_threshold((64, 64), AuditConfig(significant_motion_threshold=0.1), 1.0)

    def test_threshold_is_expressed_in_the_applied_exponent_domain(self):
        ballistic = derive_motion_threshold((24, 32), AuditConfig(), 1.0)
        diffusive = derive_motion_threshold((24, 32), AuditConfig(), 0.5)
        self.assertEqual(ballistic.units, "image_diagonals_per_second")
        self.assertEqual(diffusive.units, "image_diagonals_per_second_pow_0.5")
        self.assertEqual(diffusive.exponent, 0.5)
        # the same Official pixel threshold, re-expressed at the reference lag
        self.assertAlmostEqual(diffusive.value, ballistic.value * 0.125**0.5)
        raw = derive_motion_threshold((24, 32), AuditConfig(), 0.0)
        self.assertEqual(raw.units, "image_diagonals_per_transition")
        self.assertAlmostEqual(raw.value, ballistic.value * 0.125)

    def test_coverage_follows_the_intensity_domain_when_supplied(self):
        transitions = [transition(0.0, dt=1.0, intensity=5.0), transition(0.0, dt=3.0, intensity=0.0)]
        ballistic_domain = aggregate_channel(transitions, "residual", 1.0)
        intensity_domain = aggregate_channel(transitions, "residual", 1.0, intensity_field="residual_intensity")
        self.assertEqual(ballistic_domain.temporal_coverage, 0.0)
        self.assertEqual(intensity_domain.temporal_coverage, 0.25)
        self.assertAlmostEqual(intensity_domain.motion_intensity, 1.25)

    def test_duration_weighted_intensity(self):
        evidence = aggregate_channel([transition(1.0, 1.0), transition(3.0, 3.0)], "residual", 2.0)
        self.assertAlmostEqual(evidence.motion_intensity, 2.5)
        self.assertAlmostEqual(evidence.temporal_coverage, 0.75)
        self.assertAlmostEqual(evidence.valid_duration, 4.0)

    def test_coverage_levels_are_monotonic(self):
        levels = []
        for moving_count in range(5):
            speeds = [2.0] * moving_count + [0.0] * (4 - moving_count)
            levels.append(aggregate_channel([transition(speed) for speed in speeds], "residual", 1.0).temporal_coverage)
        self.assertEqual(levels, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_speed_change_increases_intensity(self):
        intensities = [aggregate_channel([transition(speed)], "residual", 1.0).motion_intensity for speed in (0.0, 1.0, 2.0, 4.0)]
        self.assertEqual(intensities, sorted(intensities))

    def test_invalid_transition_is_excluded(self):
        evidence = aggregate_channel([transition(100.0, valid=False), transition(2.0)], "apparent", 1.0)
        self.assertEqual(evidence.valid_transition_count, 1)
        self.assertEqual(evidence.motion_intensity, 2.0)
