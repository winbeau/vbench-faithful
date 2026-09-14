import unittest

import numpy as np

from dynamic_degree.backends.audit import AuditConfig, analyze_timed_flow_sequence, audit_effective_count_num, official_count_num
from dynamic_degree.backends.vbench import official_check_move
from dynamic_degree.schemas import TimedFrame, TimedFrameSequence


def sequence(frame_count):
    frames = tuple(
        TimedFrame(np.zeros((16, 16, 3), dtype=np.uint8), index * 0.125, index)
        for index in range(frame_count)
    )
    return TimedFrameSequence(frames, (16, 16), 8.0, 1, "synthetic_test_timestamp")


class ShortVideoBoundaryTests(unittest.TestCase):
    def test_official_count_zero_boundary_is_reproducible(self):
        self.assertEqual(official_count_num(2), 0)
        self.assertTrue(official_check_move([0.0], threshold=99.0, count_num=0))

    def test_audit_effective_count_is_never_zero_with_transition(self):
        self.assertIsNone(audit_effective_count_num(1))
        self.assertEqual(audit_effective_count_num(2), 1)

    def test_single_frame_is_insufficient(self):
        scalar_calls = []
        result = analyze_timed_flow_sequence(
            "one.mp4",
            "a person running",
            sequence(1),
            lambda left, right: np.zeros((16, 16, 2)),
            config=AuditConfig(
                scalar_score_hook=lambda evidence: scalar_calls.append(evidence) or 1.0,
                scalar_score_source="unit_test_hook",
            ),
        )
        self.assertEqual(result.status, "insufficient_transitions")
        self.assertIsNone(result.score)
        self.assertEqual(scalar_calls, [])

    def test_two_static_frames_do_not_become_true(self):
        result = analyze_timed_flow_sequence(
            "two.mp4", "a person running", sequence(2), lambda left, right: np.zeros((16, 16, 2))
        )
        self.assertEqual(result.status, "succeeded")
        self.assertAlmostEqual(result.score, 0.0)
        self.assertAlmostEqual(result.task_relevant_motion_evidence["motion_intensity"], 0.0)
        self.assertAlmostEqual(result.task_relevant_motion_evidence["temporal_coverage"], 0.0)
        self.assertTrue(result.boundary_fix_applied)
        self.assertEqual(result.audit_effective_count_num, 1)

    def test_boundary_fix_can_be_disabled_only_for_robustness_ablation(self):
        result = analyze_timed_flow_sequence(
            "two.mp4", "a person running", sequence(2), lambda left, right: np.zeros((16, 16, 2)),
            config=AuditConfig(boundary_fix_enabled=False),
        )
        self.assertEqual(result.audit_effective_count_num, 0)
        self.assertFalse(result.boundary_fix_applied)
        self.assertAlmostEqual(result.score, 0.0)
        self.assertAlmostEqual(result.task_relevant_motion_evidence["motion_intensity"], 0.0)

    def test_boundary_diagnostic_does_not_change_task_evidence(self):
        enabled = analyze_timed_flow_sequence(
            "two.mp4", "a person running", sequence(2), lambda left, right: np.zeros((16, 16, 2))
        )
        disabled = analyze_timed_flow_sequence(
            "two.mp4",
            "a person running",
            sequence(2),
            lambda left, right: np.zeros((16, 16, 2)),
            config=AuditConfig(boundary_fix_enabled=False),
        )
        self.assertEqual(enabled.task_relevant_motion_evidence, disabled.task_relevant_motion_evidence)
        self.assertEqual(enabled.score, disabled.score)
