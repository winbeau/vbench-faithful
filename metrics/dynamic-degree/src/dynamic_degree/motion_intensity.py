"""Continuous Repair motion intensity; does not redefine official VBench.

For each adjacent frame pair take the mean of the largest 5% displacement
magnitudes, then sum these displacements and divide by observed time and image
short side. No moving/not-moving threshold, sigmoid, clipping, or scale fit.
All fixed-grid predictions contribute to the spatial population; visibility
does not turn missing/unreliable motion into zero or change the denominator.
"""
from __future__ import annotations

import numpy as np


def motion_intensity(displacements, timestamps, shape):
    delta = np.asarray(displacements, dtype=float)
    times = np.asarray(timestamps, dtype=float)
    if (delta.ndim != 3 or delta.shape[-1] != 2 or delta.shape[0] < 1 or delta.shape[1] < 20
            or times.shape != (len(delta) + 1,) or not np.isfinite(delta).all()
            or not np.isfinite(times).all() or not (np.diff(times) > 0).all()
            or len(shape) != 2 or not np.isfinite(shape).all() or min(shape) <= 0):
        raise ValueError('finite full-grid T-1,N,2 displacements, increasing timestamps and H,W required')
    selected = int(delta.shape[1] * .05)
    top = np.sort(np.linalg.norm(delta, axis=-1), axis=1)[:, -selected:].mean(axis=1)
    dt = np.diff(times)
    duration = float(dt.sum())
    intensity = float(top.sum() / (min(shape) * duration))
    return {'score': intensity, 'units': 'short_side_lengths_per_second',
            'statistic': 'time_weighted_top5_percent_speed',
            'top5_mean_pixels': top.tolist(),
            'top5_speed_short_side_per_second': (top / dt / min(shape)).tolist(),
            'observed_motion_duration_seconds': duration,
            'grid_points': delta.shape[1], 'top_points': selected,
            'sampled_frames': len(times), 'thresholding': 'none', 'clipping': 'none',
            'interpretation': 'continuous estimated motion intensity, not a probability or a binary dynamic fraction'}
