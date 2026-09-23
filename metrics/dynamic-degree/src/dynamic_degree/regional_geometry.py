"""Cross-fitted regional translation/similarity/affine motion proposals.

Sparse geometry supplies candidate transformations, not a final motion score or
a reliability certificate. Rotation and scale contribute pointwise motion even
when the region center is stationary. Missing or ambiguous correspondence is
never declared zero motion. No temporal smoothing or CF metadata is used.
"""
from __future__ import annotations

import numpy as np


MODELS = ("translation", "similarity", "affine")


def transform_points(matrix, points):
    matrix, points = np.asarray(matrix, float), np.asarray(points, float)
    if matrix.shape != (2, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("2x3 transform and Nx2 points required")
    if not np.isfinite(matrix).all() or not np.isfinite(points).all():
        raise ValueError("finite geometry required")
    return points @ matrix[:, :2].T + matrix[:, 2]


def fit_geometry(source, target, model, *, robust=False, max_iterations=20):
    """Weighted linear geometry, optionally with median-scaled Huber IRLS.

    The scale is recomputed from this fit's residuals, not a fixed pixel motion
    threshold. This is only a proposal estimator: a majority of false matches
    can still give a wrong model. Held-out error must be inspected separately.
    Duplicate SIFT orientations at the same source location share unit weight.
    """
    x, y = np.asarray(source, float), np.asarray(target, float)
    if (model not in MODELS or x.ndim != 2 or x.shape[1] != 2 or x.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not isinstance(max_iterations, int) or max_iterations < 1):
        raise ValueError("finite paired Nx2 coordinates and supported model required")
    if not len(x):
        return {"status": "insufficient", "matrix": None, "reason": "no_correspondence"}
    unique, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    center = np.average(x, weights=weights, axis=0)
    centered = x - center
    # Coordinate conditioning, not image rescaling or motion attenuation.
    radius = float(np.sqrt(np.average(np.sum(centered ** 2, axis=1), weights=weights)))
    radius = radius if radius > 0 else 1.
    q = centered / radius
    n, parameters = len(x), {"translation": 2, "similarity": 4, "affine": 6}[model]
    design = np.zeros((n, 2, parameters))
    if model == "translation":
        design[:, 0, 0] = 1; design[:, 1, 1] = 1
        response = y - x
    elif model == "similarity":
        design[:, 0] = np.c_[q[:, 0], -q[:, 1], np.ones(n), np.zeros(n)]
        design[:, 1] = np.c_[q[:, 1], q[:, 0], np.zeros(n), np.ones(n)]
        response = y - center
    else:
        design[:, 0, :3] = np.c_[q, np.ones(n)]
        design[:, 1, 3:] = np.c_[q, np.ones(n)]
        response = y - center
    active = weights.copy()
    floor = np.finfo(float).eps * max(1., float(np.max(abs(x))), float(np.max(abs(y)))) * 64
    previous = None
    for iteration in range(max_iterations if robust else 1):
        fitted_weights = active.copy()
        root = np.sqrt(active)[:, None, None]
        beta, _, rank, singular = np.linalg.lstsq((design * root).reshape(-1, parameters),
                                                 (response * root[:, :, 0]).ravel(), rcond=None)
        if rank < parameters:
            return {"status": "insufficient", "matrix": None, "reason": "rank_deficient",
                    "rank": int(rank), "parameters": parameters, "unique_source_locations": len(unique)}
        residual = np.linalg.norm(np.einsum('nip,p->ni', design, beta) - response, axis=1)
        if not robust:
            break
        # Weighted median: total unit mass per source location, including its
        # possibly conflicting orientations. Avoid O(N^2) grouped scans.
        order = np.argsort(residual, kind='stable')
        middle = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
        scale = max(floor, float(residual[order[middle]]))
        updated = weights * np.minimum(1., 1.345 * scale / np.maximum(residual, floor))
        if previous is not None and np.linalg.norm(beta - previous) <= floor:
            break
        previous, active = beta.copy(), updated
    if model == "translation":
        linear, offset = np.eye(2), beta
    elif model == "similarity":
        linear = np.array([[beta[0], -beta[1]], [beta[1], beta[0]]]) / radius
        offset = beta[2:] + center - linear @ center
    else:
        linear = np.array([beta[:2], beta[3:5]]) / radius
        offset = beta[[2, 5]] + center - linear @ center
    matrix = np.c_[linear, offset]
    return {"status": "proposal_only", "matrix": matrix.tolist(), "rank": int(rank),
            "parameters": parameters, "condition_number": float(singular[0] / singular[-1]),
            "unique_source_locations": len(unique), "iterations": iteration + 1,
            "residual_pixels": np.linalg.norm(transform_points(matrix, x) - y, axis=1).tolist(),
            "fit_weights": fitted_weights.tolist(), "determinant": float(np.linalg.det(linear))}


def regional_geometry(matches, source_keys):
    """All model/estimator ablations with held-out spatial-location folds.

    Keys at identical source coordinates stay in the same fold. The folds are
    interleaved sorted locations; they are not independent physical ground truth
    or confidence intervals. No model is automatically selected as reliable.
    """
    keys = set(source_keys)
    selected = [m for m in matches if m["source_key"] in keys]
    if len({m["source_key"] for m in selected}) != len(selected) or {m["source_key"] for m in selected} != keys:
        raise ValueError("every regional source key must have exactly one correspondence")
    x = np.array([m["source_xy"] for m in selected], float).reshape(-1, 2)
    y = np.array([m["target_xy"] for m in selected], float).reshape(-1, 2)
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    folds = inverse % 2
    proposals = []
    for model in MODELS:
        for robust in (False, True):
            full = fit_geometry(x, y, model, robust=robust)
            checked = []
            for fold in (0, 1):
                train, test = folds == fold, folds != fold
                fitted = fit_geometry(x[train], y[train], model, robust=robust)
                matrix = fitted["matrix"]
                checked.append({"training_fold": fold, "training_keys": [m["source_key"] for m, k in zip(selected, train) if k],
                                "heldout_keys": [m["source_key"] for m, k in zip(selected, test) if k],
                                "fit": fitted, "heldout_residual_pixels":
                                    np.linalg.norm(transform_points(matrix, x[test]) - y[test], axis=1).tolist()
                                    if matrix is not None and test.any() else None})
            # Per-point length preserves rotation/scale, unlike centroid drift.
            full["point_displacements_pixels"] = (transform_points(full["matrix"], x) - x).tolist() if full["matrix"] is not None else None
            proposals.append({"model": model, "estimator": "huber_irls" if robust else "least_squares",
                              "full": full, "spatial_folds": checked})
    return {"status": "diagnostic_only", "score": None, "correspondences": len(selected),
            "unique_source_locations": len(unique), "proposals": proposals}
