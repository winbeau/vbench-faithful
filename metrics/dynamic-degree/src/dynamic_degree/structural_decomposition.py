"""Experimental, video-only decomposition of observed point trajectories.

Temporal smoothing alone is a negative-control ablation. Suppression requires
repeated reversals AND evidence against spatially supported residual motion.
Insufficient neighbors retain the residual rather than silently declaring it
nuisance. No construction amplitude, phase, pairing or seed is an input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecompositionConfig:
    temporal_scale_seconds: float = 0.20
    spatial_radius_fraction: float = 0.25
    color_scale: float = 0.15
    minimum_neighbors: int = 6
    coherence_error_scale: float = 0.25
    minimum_reversal_fraction: float = 0.5

    def __post_init__(self):
        for value in (self.temporal_scale_seconds, self.spatial_radius_fraction,
                      self.color_scale, self.coherence_error_scale):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("positive finite decomposition scales required")
        if self.minimum_neighbors < 4 or not 0 <= self.minimum_reversal_fraction <= 1:
            raise ValueError("invalid neighborhood/reversal criterion")


def smooth_trajectories(xy, timestamps, scale_seconds):
    """Second-derivative regularization in physical time; linear motion exact.

    All frames participate. This is not an even-frame selector or a notch at a
    known intervention frequency. Quadrature handles irregular timestamps.
    """
    xy = np.asarray(xy, float)
    t = np.asarray(timestamps, float)
    if xy.ndim != 3 or xy.shape[-1] != 2 or len(xy) != len(t) or len(t) < 3:
        raise ValueError("expected at least three T,N,2 coordinates and T timestamps")
    if not np.isfinite(xy).all() or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("finite coordinates and strictly increasing timestamps required")
    dt = np.diff(t)
    quadrature = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    second = np.zeros((len(t) - 2, len(t)))
    for i, (a, b) in enumerate(zip(dt[:-1], dt[1:])):
        second[i, i:i + 3] = np.array([1 / a, -(1 / a + 1 / b), 1 / b]) * 2 / (a + b)
    weights = np.diag(quadrature)
    penalty = second.T @ (quadrature[1:-1, None] * second)
    flat = xy.reshape(len(t), -1)
    return np.linalg.solve(weights + scale_seconds ** 4 * penalty, weights @ flat).reshape(xy.shape)


def spatial_support(deltas, positions, colors, reliable, short_side, config):
    """Leave-one-point-out local affine support, with appearance weighting.

    The center is excluded from its own prediction. Insufficient or degenerate
    neighborhoods yield support=1 (do not suppress unsupported small regions).
    No learned semantic object/part segmentation is claimed.
    """
    steps, count, _ = deltas.shape
    radius = config.spatial_radius_fraction * short_side
    support = np.ones((steps, count))
    tested = np.zeros((steps, count), bool)
    for i in range(count):
        offset = (positions - positions[i]) / radius
        distance = np.sum(offset ** 2, axis=1)
        affinity = np.exp(-distance / 2)
        if colors is not None:
            affinity *= np.exp(-np.sum((colors - colors[i]) ** 2, axis=1) / (2 * config.color_scale ** 2))
        eligible = (distance <= 1) & (affinity >= .05)
        eligible[i] = False
        design = np.c_[np.ones(count), offset]
        for t in range(steps):
            keep = eligible & reliable[t]
            if not reliable[t, i] or keep.sum() < config.minimum_neighbors:
                continue
            x, w, values = design[keep], affinity[keep], deltas[t, keep]
            normal = x.T @ (w[:, None] * x)
            if np.linalg.cond(normal) > 1e6:
                continue
            beta = np.linalg.solve(normal, x.T @ (w[:, None] * values))
            predicted = beta[0]
            neighbor_error = np.average(np.sum((values - x @ beta) ** 2, axis=1), weights=w)
            center_error = np.sum((deltas[t, i] - predicted) ** 2)
            energy = np.average(np.sum(values ** 2, axis=1), weights=w) + np.sum(deltas[t, i] ** 2)
            if energy <= 1e-10:
                continue
            ratio = (center_error + neighbor_error) / energy
            support[t, i] = np.exp(-ratio / config.coherence_error_scale)
            tested[t, i] = True
    return support, tested


def decompose_tracks(xy, timestamps, short_side, config, *, colors=None, reliable=None):
    xy = np.asarray(xy, float)
    times = np.asarray(timestamps, float)
    smooth = smooth_trajectories(xy, times, config.temporal_scale_seconds)
    if not np.isfinite(short_side) or short_side <= 0:
        raise ValueError("positive short side required")
    raw = np.diff(xy, axis=0)
    trend = np.diff(smooth, axis=0)
    residual = raw - trend
    if reliable is None:
        reliable = np.ones(raw.shape[:2], bool)
    reliable = np.asarray(reliable, bool)
    if reliable.shape != raw.shape[:2]:
        raise ValueError("reliability must be T-1,N")
    if colors is not None:
        colors = np.asarray(colors, float)
        if colors.shape != (xy.shape[1], 3) or not np.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
            raise ValueError("appearance must be N,3 RGB means in [0,1]")
    # Reversals are measured on observed motion, not manufactured by the smoother.
    dots = np.sum(raw[:-1] * raw[1:], axis=-1)
    norms = np.linalg.norm(raw[:-1], axis=-1) * np.linalg.norm(raw[1:], axis=-1)
    reversals = (dots < -.25 * norms) & (norms > 1e-10)
    repeated = reversals.mean(axis=0) > config.minimum_reversal_fraction
    positions = np.median(xy, axis=0)
    residual_support, tested = spatial_support(residual, positions, colors, reliable, short_side, config)
    spatial_only, spatial_tested = spatial_support(raw, positions, colors, reliable, short_side, config)
    gate = np.where(repeated[None] & tested, residual_support, 1.)
    combined = trend + gate[..., None] * residual
    return {"raw_deltas": raw, "temporal_only_deltas": trend,
            "structure_only_deltas": spatial_only[..., None] * raw, "full_deltas": combined,
            "residual_gate": gate, "structural_tested": tested,
            "structure_only_tested": spatial_tested, "reversal_fraction": reversals.mean(axis=0),
            "smoothed_tracks": smooth,
            "lag_speeds": {str(lag): float(np.mean(np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1)
                                                    / (times[lag:] - times[:-lag])[:, None]) / short_side)
                           for lag in range(1, min(4, len(times) - 1) + 1)}}
