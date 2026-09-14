import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dynamic_degree.backends.audit import analyze_timed_flow_sequence, normalized_speed
from dynamic_degree.models import decode_timed_frames
from dynamic_degree.schemas import TimedFrame, TimedFrameSequence


class FakeCapture:
    def __init__(self, fps, timestamps):
        self.fps = fps
        self.timestamps = timestamps
        self.index = 0
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.timestamps):
            return False, None
        self.index += 1
        return True, np.zeros((8, 10, 3), dtype=np.uint8)

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[max(0, self.index - 1)] * 1000.0
        return 0.0

    def release(self):
        self.opened = False


class TimeNormalizationTests(unittest.TestCase):
    def test_same_real_speed_across_dt(self):
        _, speed_a = normalized_speed(2.0, (60, 80), 0.1)
        _, speed_b = normalized_speed(4.0, (60, 80), 0.2)
        self.assertAlmostEqual(speed_a, speed_b)

    def test_spatial_diagonal_normalization(self):
        displacement, speed = normalized_speed(10.0, (60, 80), 0.5)
        self.assertAlmostEqual(displacement, 0.1)
        self.assertAlmostEqual(speed, 0.2)

    def test_invalid_dt_rejected(self):
        for dt in (0.0, -1.0, float("nan")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                normalized_speed(1.0, (60, 80), dt)

    def test_variable_timestamps_are_preserved(self):
        capture = FakeCapture(16.0, [0.0, 0.04, 0.13, 0.19, 0.28])
        with patch("dynamic_degree.models.cv2.VideoCapture", return_value=capture):
            sequence = decode_timed_frames(Path("variable.mp4"))
        self.assertEqual(sequence.timestamp_source, "opencv_pos_msec")
        self.assertEqual([frame.source_frame_index for frame in sequence.frames], [0, 2, 4])
        self.assertEqual([frame.timestamp_seconds for frame in sequence.frames], [0.0, 0.13, 0.28])

    def test_nominal_fps_fallback_is_explicit(self):
        capture = FakeCapture(16.0, [0.0, 0.0, 0.0, 0.0])
        with patch("dynamic_degree.models.cv2.VideoCapture", return_value=capture):
            sequence = decode_timed_frames(Path("nominal.mp4"))
        self.assertEqual(sequence.timestamp_source, "nominal_fps")
        self.assertEqual([frame.timestamp_seconds for frame in sequence.frames], [0.0, 0.125])

    def test_missing_fps_uses_valid_timestamps_for_sampling(self):
        capture = FakeCapture(0.0, [0.0, 0.1, 0.2])
        with patch("dynamic_degree.models.cv2.VideoCapture", return_value=capture):
            sequence = decode_timed_frames(Path("pts-only.mp4"))
        self.assertIsNone(sequence.source_fps)
        self.assertEqual(sequence.timestamp_source, "opencv_pos_msec")
        self.assertEqual(sequence.sampling_interval, 1)

    def test_missing_fps_and_timestamps_fails(self):
        capture = FakeCapture(0.0, [0.0, 0.0])
        with patch("dynamic_degree.models.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "neither valid timestamps"):
                decode_timed_frames(Path("invalid-time.mp4"))

    def test_invalid_transition_is_recorded_without_flow_call(self):
        frames = (
            TimedFrame(np.zeros((8, 8, 3), dtype=np.uint8), 0.0, 0),
            TimedFrame(np.zeros((8, 8, 3), dtype=np.uint8), 0.0, 1),
        )
        sequence = TimedFrameSequence(frames, (8, 8), 8.0, 1, "synthetic")
        calls = []
        result = analyze_timed_flow_sequence(
            "invalid.mp4", "a person running", sequence, lambda left, right: calls.append(True)
        )
        self.assertEqual(result.status, "insufficient_transitions")
        self.assertEqual(result.transitions[0].invalid_reason, "non_positive_dt")
        self.assertEqual(calls, [])
