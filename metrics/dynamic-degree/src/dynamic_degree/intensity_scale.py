"""A zero-preserving, monotone scale for continuous motion intensity.

This is a score-scale calibration, NOT a new motion estimator or probability.
Fit the sole half-saturation parameter on independent natural development
videos, then freeze it. Never fit to the evaluated base/CF batch or its drift.
"""
from __future__ import annotations

import numpy as np


def calibrated_intensity(intensity, scale):
    values = np.asarray(intensity, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('finite nonnegative intensity required; missing is not zero')
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('finite positive scale required')
    # Equivalent to I/(I+scale), without overflowing I+scale.
    result = np.zeros_like(values)
    positive = values > 0
    with np.errstate(over='ignore'):
        result[positive] = 1 / (1 + scale / values[positive])
    return result


def fit_scale(intensities, origin):
    """Match natural calibration-cohort means; no offset or fitted exponent.

    Original binary labels provide a numeric anchor, not physical-motion truth.
    Caller must enforce prompt/UID-disjoint fitting and record cohort identity.
    """
    values, labels = np.asarray(intensities, float), np.asarray(origin, float)
    if (values.ndim != 1 or values.size < 2 or values.shape != labels.shape
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.isin(labels, (0, 1)).all()):
        raise ValueError('complete paired intensity and original binary scores required')
    target = float(labels.mean())
    if not 0 < target < np.mean(values > 0):
        raise ValueError('no finite positive zero-preserving scale can match this mean')
    low, high = 0., float(values.max())
    while calibrated_intensity(values, high).mean() > target:
        high *= 2
        if not np.isfinite(high):
            raise ValueError('calibration scale overflow')
    for _ in range(100):
        middle = (low + high) / 2
        if calibrated_intensity(values, middle).mean() > target:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2
    actual = float(calibrated_intensity(values, scale).mean())
    if abs(actual-target) > 1e-12:
        raise ValueError('calibration mean did not converge')
    return {'formula': 'I / (I + scale)', 'scale': scale, 'scale_units': 'short_side_lengths_per_second',
            'units': 'dimensionless_calibrated_motion_intensity', 'offset': 0., 'exponent': 1.,
            'calibration_count': int(values.size), 'origin_mean': target,
            'intensity_mean': float(values.mean()), 'calibrated_mean': actual,
            'interpretation': 'continuous intensity on an Origin-anchored score scale; not a dynamic-video probability'}
