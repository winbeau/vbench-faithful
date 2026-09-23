"""Experimental decision head for fixed-grid motion, NOT the public default.

Reuse the official 6*short_side/256 displacement threshold and round(T/4)
moving-pair count, without fitting a scale to counterfactual score increases.
Grid quantiles are not dense RAFT quantiles: this is a new DEV estimator.
All grid predictions enter its fixed denominator, including low-confidence
predictions; confidence is reported separately, never used to fill zeros.
"""
from __future__ import annotations

import numpy as np

from .backends.vbench import official_check_move, official_parameters


def fixed_grid_decision(displacements, shape):
    displacement = np.asarray(displacements, dtype=float)
    if (displacement.ndim != 3 or displacement.shape[-1] != 2
            or displacement.shape[0] < 4 or displacement.shape[1] < 20
            or not np.isfinite(displacement).all() or len(shape) != 2 or min(shape) <= 0):
        raise ValueError('finite T-1,N,2 full-grid displacements and native H,W required')
    count = int(displacement.shape[1] * .05)
    values = np.linalg.norm(displacement, axis=-1)
    top = np.sort(values, axis=1)[:, -count:].mean(axis=1)
    threshold, count_num = official_parameters(tuple(shape), len(displacement) + 1)
    return {'score': float(official_check_move(top.tolist(), threshold, count_num)),
            'units': 'predicted_dynamic_video_boolean', 'top5_mean_pixels': top.tolist(),
            'threshold_pixels': threshold, 'count_num': count_num,
            'moving_pairs': int((top > threshold).sum()),
            'grid_points': displacement.shape[1], 'top_points': count,
            'sampled_frames': len(displacement) + 1,
            'warning': 'experimental grid estimator, not official RAFT parity or validated physical-motion accuracy'}
