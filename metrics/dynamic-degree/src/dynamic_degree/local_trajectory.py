"""G1 reliability ablation: fixed, overlapping local point-track windows.

This deliberately retains the OLD motion formula. It does not yet distinguish
periodic real motion from jitter. Missing correspondences still make its scalar
an observed-motion lower bound, not a completed structural-motion Repair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .trajectory import (
    TrajectoryConfig, correspondence_evidence, decode_video, query_grid, score_evidence,
)


@dataclass(frozen=True)
class LocalTrajectoryConfig(TrajectoryConfig):
    window_seconds: float = 1.0
    stride_seconds: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        if not np.isfinite(self.stride_seconds) or not 0 < self.stride_seconds < self.window_seconds:
            raise ValueError("stride_seconds must lie strictly between zero and window_seconds")


def plan_windows(timestamps: np.ndarray, config: LocalTrajectoryConfig):
    """Time-based windows and outcome-independent transition ownership.

    Overlap is context only. Each transition contributes exactly once, from the
    window in which it lies furthest from a temporal boundary (earliest tie).
    The final suffix window ensures the end is not dropped.
    """
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("at least three strictly increasing finite timestamps required")
    last = len(times) - 1
    windows = []
    start = 0
    while True:
        stop = int(np.searchsorted(times, times[start] + config.window_seconds, side="left"))
        stop = min(len(times), max(start + 3, stop))
        windows.append((start, stop))
        if stop == len(times):
            break
        next_start = int(np.searchsorted(times, times[start] + config.stride_seconds, side="left"))
        start = min(stop - 2, max(start + 1, next_start))
    # A short trailing window is replaced by a full-duration suffix if possible.
    if len(windows) > 1:
        tail = max(0, min(last - 2, int(np.searchsorted(times, times[-1] - config.window_seconds, side="right"))))
        windows[-1] = (tail, len(times))
    windows = sorted(set(windows))
    owner = np.full(last, -1, dtype=int)
    interior = np.full(last, -np.inf)
    for index, (start, stop) in enumerate(windows):
        pairs = np.arange(start, stop - 1)
        margin = np.minimum(times[pairs] - times[start], times[stop - 1] - times[pairs + 1])
        replace = margin > interior[pairs]
        owner[pairs[replace]] = index
        interior[pairs[replace]] = margin[replace]
    if (owner < 0).any():
        raise ValueError("window plan failed to cover every transition")
    return windows, owner


class LocalTrajectoryEvaluator:
    def __init__(self, config: LocalTrajectoryConfig, source_root: Path, checkpoint: Path,
                 device: str, *, tracker=None):
        from vbench_audit_models.point_tracker import CoTracker2Model

        self.config = config
        self.tracker = tracker if tracker is not None else CoTracker2Model(source_root, checkpoint, device)

    def evaluate_video(self, video: Path, *, evidence_path: Path | None = None):
        frames, timestamps, sampling = decode_video(video, self.config)
        h, w = frames.shape[1:3]
        grid = query_grid(h, w, self.config)
        windows, owner = plan_windows(timestamps, self.config)
        model_frames = (np.stack([cv2.GaussianBlur(f, (0, 0), self.config.tracker_blur_sigma) for f in frames])
                        if self.config.tracker_blur_sigma else frames)
        sampling["model_input_temporal_rgb_mae"] = float(np.mean(abs(np.diff(model_frames.astype(float), axis=0))))
        selected = {}
        selected_xy = np.empty((len(frames) - 1, len(grid), 2, 2), np.float32)
        cache = {"timestamps": timestamps, "shape": np.asarray(frames.shape), "queries": grid,
                 "windows": np.asarray(windows), "transition_owner": owner}
        window_stats = []
        for index, (start, stop) in enumerate(windows):
            prediction = self.tracker.track(model_frames[start:stop], grid)
            evidence = correspondence_evidence(frames[start:stop], prediction, self.config)
            pairs = np.flatnonzero(owner == index)
            local = pairs - start
            for key, value in evidence.items():
                if key not in selected:
                    selected[key] = np.empty((len(frames) - 1, *value.shape[1:]), dtype=value.dtype)
                selected[key][pairs] = value[local]
            xy = prediction["tracks"]
            selected_xy[pairs, :, 0] = xy[local]
            selected_xy[pairs, :, 1] = xy[local + 1]
            cache.update({f"window_{index}_{key}": value for key, value in prediction.items()})
            window_stats.append({"start_frame": start, "stop_frame_exclusive": stop,
                                 "owned_transitions": pairs.tolist(),
                                 "visible_fraction": float(np.mean(evidence["visible_pair"])),
                                 "cycle_pass_fraction": float(np.mean(evidence["cycle_error"] <= self.config.cycle_error_max))})
        result = score_evidence(selected, timestamps, h, w, self.config)
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(evidence_path, **cache, **selected, pair_tracks=selected_xy)
        return {"video": str(video), "backend": "audit", "variant": "local_trajectory_reliability_v1",
                **result, "sampling": sampling, "config": asdict(self.config), "windows": window_stats,
                "interpretation": "reliability_only_ablation_of_old_observed_motion_lower_bound; not_jitter_repair"}
