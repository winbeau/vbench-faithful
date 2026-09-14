import unittest
import json

import numpy as np

from dynamic_degree.backends.audit import AuditConfig, analyze_timed_flow_sequence
from dynamic_degree.schemas import TimedFrame, TimedFrameSequence


def two_frame_sequence(dt=0.125):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return TimedFrameSequence(
        (TimedFrame(frame, 0.0, 0), TimedFrame(frame.copy(), dt, 1)),
        (64, 64),
        1.0 / dt,
        1,
        "synthetic_test_timestamp",
    )


class CounterfactualContractTests(unittest.TestCase):
    def test_camera_translation_does_not_replace_subject_motion(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (4.0, 0.0)
        subject = analyze_timed_flow_sequence(
            "camera.mp4", "a person is running", two_frame_sequence(), lambda left, right: flow
        )
        generic = analyze_timed_flow_sequence(
            "camera.mp4", "a dynamic scene", two_frame_sequence(), lambda left, right: flow
        )
        self.assertAlmostEqual(subject.score, subject.task_relevant_motion_evidence["motion_intensity"])
        self.assertAlmostEqual(generic.score, generic.task_relevant_motion_evidence["motion_intensity"])
        self.assertNotEqual(subject.score, subject.apparent.motion_intensity)
        self.assertTrue(np.isfinite(subject.score))
        self.assertLess(
            subject.task_relevant_motion_evidence["motion_intensity"],
            generic.task_relevant_motion_evidence["motion_intensity"],
        )
        self.assertIn("temporal_coverage", subject.task_relevant_motion_evidence)
        self.assertEqual(
            subject.task_relevant_motion_evidence["evidence_components"],
            ["motion_intensity", "temporal_coverage"],
        )
        self.assertGreater(subject.camera.motion_intensity, subject.residual.motion_intensity)

    def test_subject_camera_two_by_two_channels_stay_separate(self):
        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (2.0, 0.0)
        flow[20:44, 20:44, 0] += 5.0
        result = analyze_timed_flow_sequence(
            "both.mp4", "the camera follows a running dog", two_frame_sequence(), lambda left, right: flow
        )
        self.assertIsNone(result.score)
        self.assertEqual(result.selected_evidence_channel, "camera_and_residual")
        self.assertIn("camera", result.task_relevant_motion_evidence)
        self.assertIn("residual", result.task_relevant_motion_evidence)
        self.assertEqual(result.task_relevant_motion_evidence["combination"], "structured_not_summed")

    def test_unknown_abstains_without_subject_fallback(self):
        result = analyze_timed_flow_sequence(
            "unknown.mp4", "a red vase on a table", two_frame_sequence(), lambda left, right: np.zeros((64, 64, 2))
        )
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.selected_evidence_channel)
        self.assertIsNone(result.score)

    def test_no_global_compensation_ablation_is_explicit(self):
        from dynamic_degree.backends.audit import AuditAblation

        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[:] = (2.0, 0.0)
        result = analyze_timed_flow_sequence(
            "ablation.mp4",
            "a person is running",
            two_frame_sequence(),
            lambda left, right: flow,
            config=AuditConfig(ablation=AuditAblation.WITHOUT_GLOBAL_COMPENSATION),
        )
        self.assertGreater(result.residual.motion_intensity, 0.0)
        self.assertEqual(result.transitions[0].affine.fallback_reason, "global_compensation_disabled")

    def test_diagnostics_are_json_safe_and_exclude_motion_integral(self):
        from dynamic_degree.backends.audit import audit_result_payload
        from dynamic_degree.diagnostics import DiagnosticsLevel

        result = analyze_timed_flow_sequence(
            "safe.mp4", "a person is running", two_frame_sequence(), lambda left, right: np.zeros((64, 64, 2))
        )
        serialized = json.dumps(audit_result_payload(result, DiagnosticsLevel.FULL))
        self.assertNotIn("motion_integral", serialized.lower())
        self.assertIn("magnitude_statistics", serialized)

    def test_without_persistence_ablation_keeps_intensity_only_evidence(self):
        from dynamic_degree.backends.audit import AuditAblation

        flow = np.zeros((64, 64, 2), dtype=np.float64)
        flow[20:44, 20:44, 0] = 3.0
        result = analyze_timed_flow_sequence(
            "aggregation-ablation.mp4",
            "a person is running",
            two_frame_sequence(),
            lambda left, right: flow,
            config=AuditConfig(ablation=AuditAblation.WITHOUT_CONTINUOUS_PERSISTENCE_AGGREGATION),
        )
        self.assertAlmostEqual(result.score, result.task_relevant_motion_evidence["motion_intensity"])
        self.assertGreater(result.residual.motion_intensity, 0.0)
        self.assertIsNone(result.residual.temporal_coverage)
        self.assertGreater(result.task_relevant_motion_evidence["motion_intensity"], 0.0)
        self.assertEqual(result.task_relevant_motion_evidence["evidence_components"], ["motion_intensity"])
        self.assertEqual(
            result.task_relevant_motion_evidence["aggregation"],
            "intensity_only_without_persistence_ablation",
        )

    def test_optional_scalar_hook_is_explicitly_sourced(self):
        result = analyze_timed_flow_sequence(
            "scalar.mp4",
            "a person is running",
            two_frame_sequence(),
            lambda left, right: np.zeros((64, 64, 2)),
            config=AuditConfig(
                scalar_score_hook=lambda evidence: evidence["motion_intensity"],
                scalar_score_source="unit_test_uncalibrated_intensity_hook",
            ),
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.scalar_score_source, "unit_test_uncalibrated_intensity_hook")
        self.assertFalse(result.scalar_score_independently_calibrated)

    def test_scalar_hook_without_source_fails_before_flow(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "requires scalar_score_source"):
            analyze_timed_flow_sequence(
                "scalar.mp4",
                "a person is running",
                two_frame_sequence(),
                lambda left, right: calls.append(True),
                config=AuditConfig(scalar_score_hook=lambda evidence: 0.0),
            )
        self.assertEqual(calls, [])
