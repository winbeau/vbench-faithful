"""Video-only candidate Dynamic Degree: verified point displacement per second.

This is deliberately opt-in, distinct from historical RAFT/prompt-routed repair.
All spatial/time denominators are fixed before seeing tracking reliability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TrajectoryConfig:
    sample_fps: float = 8.0
    max_side: int = 256
    grid_size: int = 12
    patch_radius: int = 7
    blur_sigma: float = 1.2
    illumination_sigma: float = 0.0
    tracker_blur_sigma: float = 0.0
    contrast_floor: float = 8.0
    match_error_max: float = 0.45
    stationary_error_max: float = 0.30
    improvement_ratio: float = 0.85
    improvement_absolute: float = 0.015
    cycle_error_max: float = 2.0
    noise_floor_pixels: float = 0.35
    min_coverage: float = 0.60
    max_frames: int = 128

    def __post_init__(self):
        positive = (self.sample_fps, self.max_side, self.grid_size, self.patch_radius,
                    self.blur_sigma, self.contrast_floor, self.match_error_max,
                    self.stationary_error_max, self.cycle_error_max, self.max_frames)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("trajectory dimensions and tolerances must be finite and positive")
        if not 0 < self.min_coverage <= 1 or not 0 < self.improvement_ratio < 1:
            raise ValueError("invalid coverage or improvement ratio")
        if not np.isfinite(self.noise_floor_pixels) or self.noise_floor_pixels < 0:
            raise ValueError("noise floor must be finite and nonnegative")
        if not np.isfinite(self.improvement_absolute) or self.improvement_absolute < 0:
            raise ValueError("improvement threshold must be finite and nonnegative")
        if not np.isfinite(self.illumination_sigma) or self.illumination_sigma < 0:
            raise ValueError("illumination scale must be finite and nonnegative")
        if self.illumination_sigma and self.illumination_sigma <= self.blur_sigma:
            raise ValueError("illumination scale must exceed noise-smoothing scale")
        if not np.isfinite(self.tracker_blur_sigma) or self.tracker_blur_sigma < 0:
            raise ValueError("tracker smoothing scale must be finite and nonnegative")

    @classmethod
    def read(cls, path: Path) -> "TrajectoryConfig":
        return cls(**json.loads(Path(path).read_text()))


def decode_video(video: Path, config: TrajectoryConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nearest-frame physical-time sampling; no synthetic/interpolated frames.

    Prefer decoder PTS. Nominal-FPS timestamps are an explicit CFR fallback.
    Fail on excessive duration instead of silently scoring an arbitrary prefix.
    """
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames, pts = [], []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            pts.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
            h, w = bgr.shape[:2]
            scale = min(1.0, config.max_side / max(h, w))
            shape = (max(1, round(w * scale)), max(1, round(h * scale)))
            frames.append(cv2.cvtColor(cv2.resize(bgr, shape, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < 3:
        raise ValueError("at least three decodable frames are required")
    timestamps = np.asarray(pts, dtype=np.float64)
    source = "decoder_pts"
    if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("no reliable timestamps or nominal FPS")
        timestamps = np.arange(len(frames)) / fps
        source = "nominal_fps_fallback"
    timestamps -= timestamps[0]
    targets = np.arange(0, timestamps[-1] + 1e-9, 1 / config.sample_fps)
    right = np.minimum(np.searchsorted(timestamps, targets), len(timestamps) - 1)
    left = np.maximum(0, right - 1)
    indices = np.unique(np.where(abs(timestamps[left] - targets) <= abs(timestamps[right] - targets), left, right))
    if len(indices) < 3 or len(indices) > config.max_frames:
        raise ValueError(f"sample count {len(indices)} outside [3, {config.max_frames}]")
    sampled = np.stack([frames[i] for i in indices])
    return sampled, timestamps[indices], {
        "timestamp_source": source, "source_fps": fps, "decoded_count": len(frames),
        "source_indices": indices.tolist(), "timestamps": timestamps[indices].tolist(),
        "sampled_shape": list(frames[0].shape),
        "sampled_temporal_rgb_mae": float(np.mean(abs(np.diff(sampled.astype(float), axis=0)))),
    }


def query_grid(height: int, width: int, config: TrajectoryConfig) -> np.ndarray:
    margin = config.patch_radius + 2
    if min(height, width) <= 2 * margin:
        raise ValueError("video too small for fixed patch grid")
    xs = np.linspace(margin, width - 1 - margin, config.grid_size)
    ys = np.linspace(margin, height - 1 - margin, config.grid_size)
    return np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)


def _patches(gray: np.ndarray, xy: np.ndarray, config: TrajectoryConfig) -> np.ndarray:
    offsets = np.arange(-config.patch_radius, config.patch_radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(offsets, offsets)
    mx = xy[:, 0, None] + xx.ravel()[None]
    my = xy[:, 1, None] + yy.ravel()[None]
    patches = cv2.remap(gray, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    patches -= patches.mean(axis=1, keepdims=True)
    return patches / np.maximum(patches.std(axis=1, keepdims=True), config.contrast_floor)


def correspondence_evidence(frames: np.ndarray, tracks: dict[str, np.ndarray],
                            config: TrajectoryConfig) -> dict[str, np.ndarray]:
    """Compare stationary and tracked explanations using local normalized structure."""
    xy = np.asarray(tracks["tracks"], dtype=np.float32)
    time, height, width, _ = frames.shape
    if xy.ndim != 3 or xy.shape[0] != time or xy.shape[-1] != 2 or not np.isfinite(xy).all():
        raise ValueError("invalid tracker coordinates")
    descriptors = []
    gray_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        filtered = cv2.GaussianBlur(gray, (0, 0), config.blur_sigma)
        if config.illumination_sigma:
            filtered -= cv2.GaussianBlur(gray, (0, 0), config.illumination_sigma)
        gray_frames.append(filtered)
    for index in range(time - 1):
        before = _patches(gray_frames[index], xy[index], config)
        moved = _patches(gray_frames[index + 1], xy[index + 1], config)
        still = _patches(gray_frames[index + 1], xy[index], config)
        descriptors.append(np.stack((np.mean((before - moved) ** 2, axis=1),
                                     np.mean((before - still) ** 2, axis=1))))
    errors = np.stack(descriptors)
    reverse = np.asarray(tracks["reverse_tracks"], dtype=np.float32)
    visible = np.asarray(tracks["visible"], dtype=bool)
    reverse_visible = np.asarray(tracks["reverse_visible"], dtype=bool)
    if reverse.shape != xy.shape or visible.shape != xy.shape[:2] or reverse_visible.shape != visible.shape:
        raise ValueError("inconsistent tracker output shapes")
    cycle = np.linalg.norm(xy - reverse, axis=-1)
    r = config.patch_radius
    inside = ((xy[..., 0] >= r) & (xy[..., 0] <= width - 1 - r)
              & (xy[..., 1] >= r) & (xy[..., 1] <= height - 1 - r))
    return {
        "displacement": np.linalg.norm(np.diff(xy, axis=0), axis=-1),
        "moving_error": errors[:, 0], "stationary_error": errors[:, 1],
        "cycle_error": np.maximum(cycle[:-1], cycle[1:]),
        "visible_pair": visible[:-1] & visible[1:] & reverse_visible[:-1] & reverse_visible[1:],
        "inside_pair": inside[:-1] & inside[1:],
    }


def score_evidence(evidence: dict[str, np.ndarray], timestamps: np.ndarray,
                   height: int, width: int, config: TrajectoryConfig) -> dict[str, Any]:
    displacement = np.asarray(evidence["displacement"], dtype=float)
    dt = np.diff(np.asarray(timestamps, dtype=float))
    if displacement.ndim != 2 or displacement.shape[0] != len(dt) or not np.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("displacements require a positive timestamp interval per pair")
    if not np.isfinite(displacement).all() or (displacement < 0).any() or min(height, width) <= 0:
        raise ValueError("invalid displacement or image size")
    moved = evidence["moving_error"]
    stationary = evidence["stationary_error"]
    tracked = (evidence["visible_pair"] & evidence["inside_pair"]
               & (evidence["cycle_error"] <= config.cycle_error_max)
               & (moved <= config.match_error_max))
    moving = (tracked & (moved + config.improvement_absolute < config.improvement_ratio * stationary))
    # Static support is an actual same-position match, never an invisible-track fallback.
    static = evidence["inside_pair"] & (stationary <= config.stationary_error_max) & ~moving
    valid = moving | static
    duration = float(dt.sum())
    coverage = float(np.sum(valid * dt[:, None]) / (duration * valid.shape[1]))
    denominator = duration * displacement.shape[1] * min(height, width)
    verified = np.where(moving, displacement, 0)
    full = np.where(moving & (displacement > config.noise_floor_pixels), displacement, 0)
    score = float(full.sum() / denominator)
    return {
        "status": "succeeded" if coverage >= config.min_coverage else "insufficient_evidence",
        "score": score if coverage >= config.min_coverage else None,
        "units": "short_side_lengths_per_second",
        "coverage": coverage,
        "moving_fraction": float(np.mean(moving)), "static_fraction": float(np.mean(static)),
        "noise_floor_pixels": config.noise_floor_pixels,
        "ablations": {"raw_tracks": float(displacement.sum() / denominator),
                      "verified_correspondence": float(verified.sum() / denominator),
                      "full": score},
        "valid_pairs": int(valid.sum()), "total_pairs": int(valid.size),
    }


class TrajectoryEvaluator:
    def __init__(self, config: TrajectoryConfig, source_root: Path, checkpoint: Path,
                 device: str, *, tracker=None):
        from vbench_audit_models.point_tracker import CoTracker2Model

        self.config = config
        self.tracker = tracker if tracker is not None else CoTracker2Model(source_root, checkpoint, device)

    def evaluate_video(self, video: Path, *, evidence_path: Path | None = None) -> dict[str, Any]:
        # No metadata/pair/seed arguments on purpose; filename is only used for I/O.
        frames, timestamps, sampling = decode_video(video, self.config)
        h, w = frames.shape[1:3]
        grid = query_grid(h, w, self.config)
        model_frames = (np.stack([cv2.GaussianBlur(frame, (0, 0), self.config.tracker_blur_sigma) for frame in frames])
                        if self.config.tracker_blur_sigma else frames)
        sampling["model_input_temporal_rgb_mae"] = float(np.mean(abs(np.diff(model_frames.astype(float), axis=0))))
        tracks = self.tracker.track(model_frames, grid)
        evidence = correspondence_evidence(frames, tracks, self.config)
        result = score_evidence(evidence, timestamps, h, w, self.config)
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(evidence_path, **tracks, **evidence, timestamps=timestamps,
                                shape=np.asarray(frames.shape), queries=grid)
        return {"video": str(video), "backend": "audit", "variant": "trajectory_candidate_v1",
                **result, "sampling": sampling, "config": asdict(self.config)}
