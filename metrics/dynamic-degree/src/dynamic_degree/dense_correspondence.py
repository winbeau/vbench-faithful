"""Video-only dense local correspondence development ablation.

RAFT is an independently pretrained correspondence estimator, not an artifact
classifier. Forward/backward verification and a deformable photometric warp
replace the CoTracker visibility head and rigid patch translation. Static
support must not contradict a better moving match. The observed path accumulator
is retained only as a named reliability ablation, NOT the complete Repair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .trajectory import TrajectoryConfig, _patches, decode_video, query_grid


@dataclass(frozen=True)
class DenseCorrespondenceConfig(TrajectoryConfig):
    # Shared preprocessing stays explicit; no temporal filtering is performed.
    tracker_blur_sigma: float = 0.0


def sample_field(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sampling; callers must separately check geometric support."""
    shape = xy.shape[:-1]
    result = cv2.remap(np.asarray(field, np.float32), xy[..., 0].reshape(-1, 1).astype(np.float32),
                       xy[..., 1].reshape(-1, 1).astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    return result.reshape(*shape, *field.shape[2:])


def dense_evidence(frames: np.ndarray, forward: np.ndarray, backward: np.ndarray,
                   config: DenseCorrespondenceConfig, *, query_positions=None):
    time, height, width, _ = frames.shape
    expected = (time - 1, height, width, 2)
    if forward.shape != expected or backward.shape != expected:
        raise ValueError("dense forward/backward shape must match sampled frames")
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("nonfinite flow cannot be evidence")
    queries = query_grid(height, width, config) if query_positions is None else np.asarray(query_positions, np.float32)
    if (queries.ndim not in (2, 3) or queries.shape[-1] != 2 or not np.isfinite(queries).all()
            or (queries.ndim == 3 and len(queries) != time - 1)):
        raise ValueError("queries must be finite N,2 or T-1,N,2")
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    grid = np.stack((xx, yy), axis=-1)
    gray = []
    for frame in frames:
        base = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        filtered = cv2.GaussianBlur(base, (0, 0), config.blur_sigma)
        if config.illumination_sigma:
            filtered -= cv2.GaussianBlur(base, (0, 0), config.illumination_sigma)
        gray.append(filtered)
    result = {key: [] for key in ("displacement", "moving_error", "stationary_error", "cycle_error",
                                  "visible_pair", "inside_pair", "flow_vectors", "warp_patch_support")}
    size = 2 * config.patch_radius + 1
    for t in range(time - 1):
        query = queries if queries.ndim == 2 else queries[t]
        endpoints = grid + forward[t]
        in_bounds = ((endpoints[..., 0] >= 0) & (endpoints[..., 0] <= width - 1)
                     & (endpoints[..., 1] >= 0) & (endpoints[..., 1] <= height - 1))
        warped = cv2.remap(gray[t + 1], endpoints[..., 0], endpoints[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        reverse_at_end = cv2.remap(backward[t], endpoints[..., 0], endpoints[..., 1], cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        cycle = np.linalg.norm(forward[t] + reverse_at_end, axis=-1)
        before = _patches(gray[t], query, config)
        moved = _patches(warped, query, config)
        stationary = _patches(gray[t + 1], query, config)
        support = cv2.boxFilter(in_bounds.astype(np.float32), -1, (size, size), normalize=True,
                                borderType=cv2.BORDER_CONSTANT)
        support_at_queries = sample_field(support, query)
        vectors = sample_field(forward[t], query)
        result["displacement"].append(np.linalg.norm(vectors, axis=-1))
        result["flow_vectors"].append(vectors)
        result["moving_error"].append(np.mean((before - moved) ** 2, axis=1))
        result["stationary_error"].append(np.mean((before - stationary) ** 2, axis=1))
        result["cycle_error"].append(sample_field(cycle, query))
        # This is a geometric in-bounds proxy, NOT learned model visibility.
        result["visible_pair"].append(sample_field(in_bounds.astype(np.float32), query) >= 1 - 1e-6)
        result["inside_pair"].append(support_at_queries >= 1 - 1e-6)
        result["warp_patch_support"].append(support_at_queries)
    return {key: np.stack(value) for key, value in result.items()}


def score_dense_evidence(evidence, timestamps, height, width, config):
    """A failed moving correspondence is NOT evidence of stationarity.

    This corrects the legacy static fallback's contradictory evidence path.
    The scalar remains a partial observed-motion diagnostic until missing
    motion and the structural/nuisance decomposition are explicitly handled.
    """
    displacement = np.asarray(evidence["displacement"], float)
    dt = np.diff(np.asarray(timestamps, float))
    if (displacement.ndim != 2 or len(dt) != len(displacement) or not np.isfinite(dt).all()
            or not (dt > 0).all() or not np.isfinite(displacement).all()
            or (displacement < 0).any() or min(height, width) <= 0):
        raise ValueError("invalid displacement, timestamps, or geometry")
    advantage = evidence["moving_error"] + config.improvement_absolute < config.improvement_ratio * evidence["stationary_error"]
    reliable = (evidence["inside_pair"] & evidence["visible_pair"]
                & (evidence["cycle_error"] <= config.cycle_error_max)
                & (evidence["moving_error"] <= config.match_error_max))
    moving = reliable & advantage
    stationary = evidence["inside_pair"] & (evidence["stationary_error"] <= config.stationary_error_max) & ~advantage
    valid = moving | stationary
    coverage = float(np.sum(valid * dt[:, None]) / (dt.sum() * valid.shape[1]))
    denominator = dt.sum() * displacement.shape[1] * min(height, width)
    verified = np.where(moving, displacement, 0)
    observed = float(np.where(moving & (displacement > config.noise_floor_pixels), displacement, 0).sum() / denominator)
    return {"status": "succeeded" if coverage >= config.min_coverage else "insufficient_evidence",
            "score": observed if coverage >= config.min_coverage else None,
            "coverage": coverage, "moving_fraction": float(moving.mean()), "static_fraction": float(stationary.mean()),
            "units": "short_side_lengths_per_second", "noise_floor_pixels": config.noise_floor_pixels,
            "valid_pairs": int(valid.sum()), "total_pairs": int(valid.size),
            "static_conflict_fraction": float(np.mean(advantage & ~reliable & evidence["inside_pair"]
                                                     & (evidence["stationary_error"] <= config.stationary_error_max))),
            "ablations": {"raw_tracks": float(displacement.sum() / denominator),
                          "verified_correspondence": float(verified.sum() / denominator), "full": observed}}


class DenseCorrespondenceEvaluator:
    def __init__(self, config: DenseCorrespondenceConfig, weight: Path, upstream: Path,
                 device: str, *, flow_model=None):
        from vbench_audit_models.raft import RaftFlowModel

        self.config = config
        self.flow_model = flow_model if flow_model is not None else RaftFlowModel(device, weight, upstream)

    def evaluate_video(self, video: Path, *, evidence_path: Path | None = None):
        frames, timestamps, sampling = decode_video(video, self.config)
        model_frames = (np.stack([cv2.GaussianBlur(f, (0, 0), self.config.tracker_blur_sigma) for f in frames])
                        if self.config.tracker_blur_sigma else frames)
        forward = np.stack([self.flow_model.compute_flow(a, b) for a, b in zip(model_frames[:-1], model_frames[1:])])
        backward = np.stack([self.flow_model.compute_flow(b, a) for a, b in zip(model_frames[:-1], model_frames[1:])])
        evidence = dense_evidence(frames, forward, backward, self.config)
        h, w = frames.shape[1:3]
        result = score_dense_evidence(evidence, timestamps, h, w, self.config)
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(evidence_path, **evidence, forward=forward, backward=backward,
                                timestamps=timestamps, shape=np.asarray(frames.shape), queries=query_grid(h, w, self.config))
        return {"video": str(video), "backend": "audit", "variant": "dense_correspondence_reliability_v1",
                **result, "sampling": sampling, "config": asdict(self.config),
                "visibility_mode": "in_bounds_proxy_not_a_learned_visibility_head",
                "interpretation": "partial_observed_motion_accumulator; correspondence_ablation_not_jitter_repair"}
