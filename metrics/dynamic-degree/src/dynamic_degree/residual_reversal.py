"""Experimental residual-first extension of the guarded reversal candidate.

A large camera/part motion can dominate the raw temporal basis and hide a
smaller nuisance. Analyze the residual AFTER orthogonally separating protected
regional affine and per-point mean motion. Retain the original branch and its
gates unchanged. This is still a video-only hypothesis, not artifact causality.
"""
from __future__ import annotations

import numpy as np

from .image_plane_modes import constrained_mode_residual, regional_reversal_ablation, spatial_design


def residual_first_reversal(values, times, xy, owners, reliable, shape):
    """Two specified branches; no search over parameters or desired scores.

    The only change from v1 is a second application of its same gates to the
    removable subspace of the remaining motion. No reference, seed, source UID,
    construction frequency, or score-scale input enters this estimator.
    """
    values, times, xy = (np.asarray(a, float) for a in (values, times, xy))
    owners, reliable = np.asarray(owners), np.asarray(reliable, bool)
    baseline, old = regional_reversal_ablation(values, times, xy, owners, reliable, shape)
    remaining = old['corrected_velocity']
    residual, separation = constrained_mode_residual(remaining, times, xy, owners, reliable, shape)
    extra = np.zeros_like(values)
    branch = {'status': 'not_run_projection_not_converged', 'subtracted': False}
    if separation['converged']:
        branch, candidate = regional_reversal_ablation(residual, times, xy, owners, reliable, shape)
        extra = candidate['removal']
    removal = old['removal'] + extra
    corrected = values-removal
    dt = np.diff(times)
    design = spatial_design(xy, shape, 0)
    affine_error = 0.
    for t in range(len(dt)):
        for owner in np.unique(owners):
            selected = reliable[t] & (owners == owner)
            affine_error = max(affine_error, float(np.max(np.abs(design[selected].T @ removal[t, selected]), initial=0)))
    mean_error = float(np.max(np.abs(np.sum(removal * dt[:, None, None], axis=0))))
    # This verifies algebraic protection, not correct masks or physical tracks.
    if np.any(removal[~reliable]) or max(affine_error, mean_error) > 1e-5:
        raise ValueError('residual branch changed protected affine/mean/unknown motion')
    return {'status': 'experimental_motion_decomposition', 'version': 'protected-residual-reversal-v1',
            'baseline': baseline, 'separation': separation, 'residual_branch': branch,
            'subtracted': bool(np.any(removal)), 'extra_subtracted': bool(np.any(extra)),
            'affine_constraint_error': affine_error, 'mean_velocity_constraint_error': mean_error,
            'physical_artifact_classification': 'NOT VERIFIED',
            'limitation': 'genuine synchronized nonaffine motion can resemble nuisance; natural-motion validation required'}, {
            'raw_velocity': values, 'baseline_corrected_velocity': remaining, 'analysis_residual': residual,
            'baseline_removal': old['removal'], 'extra_removal': extra, 'removal': removal,
            'corrected_velocity': corrected, 'timestamps': times, 'xy': xy, 'owners': owners,
            'reliable_pair': reliable}
