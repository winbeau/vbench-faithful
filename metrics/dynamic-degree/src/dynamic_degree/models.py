from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .schemas import TimedFrame, TimedFrameSequence

# Compatibility re-export: model ownership lives in audit-models so metric
# packages do not depend on one another's scoring implementation.
from vbench_audit_models import RaftFlowModel


def _strictly_increasing(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values) and all(right > left for left, right in zip(values, values[1:]))


def _derive_fps_from_timestamps(timestamps: list[float]) -> float | None:
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    valid = deltas[np.isfinite(deltas) & (deltas > 0)]
    if valid.size == 0:
        return None
    return float(1.0 / np.median(valid))


def decode_timed_frames(video: Path) -> TimedFrameSequence:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    source_indices: list[int] = []
    source_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            source_indices.append(source_index)
            source_index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {video}")

    pts_valid = len(timestamps) == 1 or _strictly_increasing(timestamps)
    fps_valid = math.isfinite(reported_fps) and reported_fps > 0
    sampling_fps = reported_fps if fps_valid else _derive_fps_from_timestamps(timestamps)
    if sampling_fps is None or not math.isfinite(sampling_fps) or sampling_fps <= 0:
        raise ValueError("video has neither valid timestamps nor a reliable FPS")
    interval = max(1, round(sampling_fps / 8.0))

    if pts_valid:
        selected_timestamps = timestamps
        timestamp_source = "opencv_pos_msec"
    elif fps_valid:
        selected_timestamps = [index / reported_fps for index in source_indices]
        timestamp_source = "nominal_fps"
    else:
        raise ValueError("video timestamps are invalid and FPS fallback is unavailable")

    sampled = tuple(
        TimedFrame(frames[index], float(selected_timestamps[index]), source_indices[index])
        for index in range(0, len(frames), interval)
    )
    height, width = frames[0].shape[:2]
    return TimedFrameSequence(
        frames=sampled,
        frame_shape=(height, width),
        source_fps=reported_fps if fps_valid else None,
        sampling_interval=interval,
        timestamp_source=timestamp_source,
    )

