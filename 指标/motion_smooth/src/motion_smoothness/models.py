from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schemas import TimedFrame


class MotionFieldEstimator(Protocol):
    def estimate(self, frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        ...


def decode_timed_frames(video: Path) -> tuple[tuple[TimedFrame, ...], float]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[TimedFrame] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(timestamp) or (frames and timestamp <= frames[-1].timestamp_seconds):
                timestamp = index / fps if np.isfinite(fps) and fps > 0 else float(index)
            frames.append(TimedFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp, index))
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {video}")
    if not np.isfinite(fps) or fps <= 0:
        deltas = np.diff([frame.timestamp_seconds for frame in frames])
        valid = deltas[np.isfinite(deltas) & (deltas > 0)]
        if valid.size == 0:
            raise ValueError("video has neither valid FPS nor reliable timestamps")
        fps = float(1.0 / np.median(valid))
    return tuple(frames), fps


class RaftFlowEstimator:
    """Adapter around the already audited Dynamic Degree RAFT wrapper."""

    def __init__(self, device: Any, model_weight: Path, upstream_path: Path):
        from dynamic_degree.models import RaftFlowModel

        self._model = RaftFlowModel(device, model_weight, upstream_path)

    def estimate(self, frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.compute_flow(frame0, frame1), dtype=np.float64)
