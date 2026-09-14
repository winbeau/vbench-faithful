import unittest
import numpy as np

from dynamic_degree.backends.audit import AuditConfig, AuditVariant, analyze_timed_flow_sequence
from dynamic_degree.backends.audit import audit_result_payload
from dynamic_degree.diagnostics import DiagnosticsLevel
from dynamic_degree.schemas import TimedFrame, TimedFrameSequence


def sequence(dt=0.125):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return TimedFrameSequence((TimedFrame(frame, 0.0, 0), TimedFrame(frame, dt, 1)), (8, 8), 8.0, 1, 'test')


def flow(dx):
    value = np.zeros((8, 8, 2), dtype=np.float64); value[..., 0] = dx
    return lambda a, b: value


class VariantComponentTests(unittest.TestCase):
    def evaluate(self, variant, dt=0.125, dx=1.0):
        return analyze_timed_flow_sequence('x', 'camera pans while a subject runs', sequence(dt), flow(dx), config=AuditConfig(variant=variant))

    def test_component_provenance_and_no_source_affine_leakage(self):
        expected = {
            AuditVariant.TIME_ONLY: (True, False, False, False),
            AuditVariant.SOURCE_ONLY: (False, True, False, False),
            AuditVariant.DURATION_ONLY: (False, False, True, False),
            AuditVariant.SOURCE_TIME: (True, True, False, False),
            AuditVariant.FULL: (True, True, True, True),
        }
        for variant, flags in expected.items():
            result = self.evaluate(variant)
            self.assertEqual(tuple(result.component_provenance.values()), flags)
            self.assertEqual(result.target_decision.target, 'generic' if not flags[3] else 'both')
            if not flags[1]: self.assertIsNone(result.transitions[0].affine)
            if not flags[2]: self.assertIsNone(result.apparent.temporal_coverage)
            diagnostics = audit_result_payload(result, DiagnosticsLevel.COMPACT)['diagnostics']
            self.assertIsNotNone(diagnostics)
            self.assertEqual(
                diagnostics['audit_video']['aggregation_method'],
                'duration_weighted_mean_speed_and_duration_fraction' if flags[2] else 'unweighted_mean_transition_motion_intensity_no_temporal_coverage',
            )

    def test_time_only_is_dt_invariant_and_source_only_is_not_time_normalized(self):
        left = self.evaluate(AuditVariant.TIME_ONLY, .125, 1.0).apparent.motion_intensity
        right = self.evaluate(AuditVariant.TIME_ONLY, .25, 2.0).apparent.motion_intensity
        self.assertAlmostEqual(left, right)
        self.assertNotEqual(self.evaluate(AuditVariant.SOURCE_ONLY, .125, 1.0).apparent.motion_intensity, self.evaluate(AuditVariant.SOURCE_ONLY, .25, 2.0).apparent.motion_intensity)

    def test_duration_only_has_coverage_without_source_or_time(self):
        result = self.evaluate(AuditVariant.DURATION_ONLY)
        self.assertIsNone(result.transitions[0].affine)
        self.assertIsNotNone(result.apparent.temporal_coverage)
        self.assertEqual(result.threshold.units, 'image_diagonals_per_transition')
