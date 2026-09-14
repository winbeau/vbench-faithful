from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..metric import analyze_motion_fields
from ..models import MotionFieldEstimator
from ..schemas import MotionField, MotionSmoothnessConfig, MotionSmoothnessResult, TimedFrame


def evaluate_timed_frames(
    video: str | Path,
    frames: tuple[TimedFrame, ...],
    estimator: MotionFieldEstimator,
    config: MotionSmoothnessConfig | None = None,
) -> MotionSmoothnessResult:
    fields: list[MotionField] = []
    for first, second in zip(frames, frames[1:]):
        dt = float(second.timestamp_seconds - first.timestamp_seconds)
        if not np.isfinite(dt) or dt <= 0:
            continue
        flow = estimator.estimate(np.asarray(first.frame), np.asarray(second.frame))
        fields.append(MotionField(flow=flow, dt=dt, source_frame_index=first.source_frame_index))
    analyzed = analyze_motion_fields(fields, config)
    diagnostics = dict(analyzed["diagnostics"])
    diagnostics.update({
        "video_path": str(video),
        "fps": float(1.0 / np.median([field.dt for field in fields])) if fields else None,
        "num_frames": len(frames),
        "num_motion_fields": len(fields),
    })
    return MotionSmoothnessResult(str(video), float(analyzed["score"]), float(analyzed["D_video"]), diagnostics)
