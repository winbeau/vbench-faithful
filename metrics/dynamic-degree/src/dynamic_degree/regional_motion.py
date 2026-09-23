"""Video-only region-supported translation hypotheses, not a calibrated score.

SAM regions propose shared motion support, never foreground truth. Joint
descriptor alignment retains multiple peaks and independently fits two spatial
folds. No temporal filtering: reversals, camera motion, and short paths remain.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionMotionConfig:
    hypotheses: int = 3
    min_overlap: float = .8
    refinement_steps: int = 8
    convergence: float = .001  # descriptor-grid precision, not a noise floor
    minimum_fold_mass: float = 2.0  # descriptor patch equivalents, not object area cutoff

    def __post_init__(self):
        if (self.hypotheses < 1 or self.refinement_steps < 1 or not 0 < self.min_overlap <= 1
                or not np.isfinite(self.convergence) or self.convergence <= 0
                or not np.isfinite(self.minimum_fold_mass) or self.minimum_fold_mass <= 0):
            raise ValueError("invalid region motion configuration")


def region_weights(masks, grid_shape):
    masks = np.asarray(masks)
    h, w = map(int, grid_shape)
    if masks.ndim != 3 or not np.isin(masks, (0, 1)).all() or min(h, w) < 2:
        raise ValueError("binary M,H,W masks and positive descriptor grid required")
    if len(masks) == 0:
        return np.zeros((0, h * w), np.float32)
    # Occupancy rather than a center hit retains thin/small region evidence.
    return np.stack([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).ravel()
                     for m in masks])


def displacement_table(similarity, grid_shape):
    """Full-image search with explicit visible mass, not a boundary penalty.

    A fixed high cost for every newly out-of-view point creates a discontinuous
    barrier at zero motion on a full-frame mask and can suppress camera motion.
    Use conditional visible-support loss and keep its denominator/overlap
    explicit. This loss alone is not a complete-image motion certificate.
    """
    h, w = map(int, grid_shape)
    s = np.asarray(similarity, np.float32)
    n = h * w
    if s.shape != (n, n) or not np.isfinite(s).all():
        raise ValueError("finite square descriptor similarity required")
    yy, xx = np.mgrid[:h, :w]
    dy, dx = np.mgrid[-h + 1:h, -w + 1:w]
    offsets = np.stack((dx.ravel(), dy.ravel()), axis=-1)
    tx = xx.ravel()[:, None] + offsets[None, :, 0]
    ty = yy.ravel()[:, None] + offsets[None, :, 1]
    valid = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    targets = np.clip(ty, 0, h - 1) * w + np.clip(tx, 0, w - 1)
    cost = np.where(valid, np.maximum(0, 2 - 2 * s[np.arange(n)[:, None], targets]), 0.)
    return offsets, cost.astype(np.float32), valid.astype(np.float32)


def _sample(features, xy):
    h, w, _ = features.shape
    xy = np.asarray(xy, float)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
    x = np.clip(xy[:, 0], 0, w - 1); y = np.clip(xy[:, 1], 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(int), w - 2); y0 = np.minimum(np.floor(y).astype(int), h - 2)
    ax, ay = (x - x0)[:, None], (y - y0)[:, None]
    a, b, c, d = features[y0, x0], features[y0, x0 + 1], features[y0 + 1, x0], features[y0 + 1, x0 + 1]
    value = (1 - ay) * ((1 - ax) * a + ax * b) + ay * ((1 - ax) * c + ax * d)
    gx = (1 - ay) * (b - a) + ay * (d - c)
    gy = (1 - ax) * (c - a) + ax * (d - b)
    return value, gx, gy, inside


def alignment_error(source, target, weights, displacement, target_weights=None):
    h, w, _ = source.shape
    yy, xx = np.mgrid[:h, :w]
    q = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    weights = np.asarray(weights, float).ravel()
    keep = weights > 0
    if not keep.any():
        return None, 0.
    warped, _, _, inside = _sample(target, q[keep] + displacement)
    residual = ((warped - source.reshape(h * w, -1)[keep]) ** 2).sum(-1)
    mass = weights[keep]
    active = mass * inside
    if target_weights is not None:
        target_mass, _, _, _ = _sample(np.asarray(target_weights, float)[..., None], q[keep] + displacement)
        active *= target_mass[:, 0]
    observed = np.sum(active)
    return (float(np.sum(active * residual) / observed) if observed > 0 else None,
            float(observed / mass.sum()))


def refine_translation(source, target, weights, initial, config, target_weights=None):
    """Direct bilinear descriptor alignment: identical input has zero residual.

    Unlike a one-sided parabola around a cosine peak, this does not introduce a
    self-motion offset from asymmetric neighboring correlations. No loss below
    a hand-tuned motion magnitude is zeroed out.
    """
    h, w, _ = source.shape
    yy, xx = np.mgrid[:h, :w]
    q = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    weights = np.asarray(weights, float).ravel()
    keep = weights > 0
    mass = weights[keep]
    if not len(mass):
        return {"displacement_grid": None, "loss": None, "overlap": 0., "rank": 0, "converged": False}
    source_values = source.reshape(h * w, -1)[keep]
    xy = q[keep]
    d = np.asarray(initial, float).copy()
    if target_weights is not None:
        target_weights = np.asarray(target_weights, float)
        if (target_weights.shape != (h, w) or not np.isfinite(target_weights).all()
                or (target_weights < 0).any() or (target_weights > 1).any()):
            raise ValueError("target visibility must be a finite H,W array in [0,1]")
    old, overlap = alignment_error(source, target, weights, d, target_weights)
    if old is None or overlap < config.min_overlap:
        return {"displacement_grid": None, "loss": None, "overlap": overlap,
                "rank": 0, "converged": False}
    rank, converged = 0, False
    for _ in range(config.refinement_steps):
        value, gx, gy, inside = _sample(target, xy + d)
        active = mass * inside
        if target_weights is not None:
            target_mass, _, _, _ = _sample(target_weights[..., None], xy + d)
            active *= target_mass[:, 0]
        residual = value - source_values
        jac = np.stack((gx, gy), axis=-1)
        matrix = np.einsum('n,nci,ncj->ij', active, jac, jac)
        rhs = np.einsum('n,nci,nc->i', active, jac, residual)
        eigen = np.linalg.eigvalsh(matrix)
        rank = int(np.sum(eigen > max(1e-10, eigen[-1] * 1e-6)))
        if rank < 2:
            break
        step = np.linalg.solve(matrix, rhs)
        step /= max(1., float(np.linalg.norm(step)))
        if np.linalg.norm(step) < config.convergence:
            converged = True
            break
        improved = False
        for scale in (1., .5, .25, .125):
            candidate = d - scale * step
            loss, candidate_overlap = alignment_error(source, target, weights, candidate, target_weights)
            if loss is not None and candidate_overlap >= config.min_overlap and loss < old - 1e-12:
                d, old, overlap = candidate, loss, candidate_overlap
                improved = True
                break
        if not improved:
            break
    return {"displacement_grid": d.tolist(), "loss": old, "overlap": overlap,
            "rank": rank, "converged": converged}


def _peaks(cost, overlap, offsets, config):
    available = np.isfinite(cost) & (overlap >= config.min_overlap)
    selected = []
    for _ in range(config.hypotheses):
        if not available.any():
            break
        i = int(np.argmin(np.where(available, cost, np.inf)))
        selected.append(i)
        # Keep distinct local modes; the complete curve remains available to
        # the ambiguity diagnostic, including an exactly flat/tied curve.
        available &= np.max(abs(offsets - offsets[i]), axis=1) > 1
    return selected


def fit_regions(source, target, weights, config=RegionMotionConfig()):
    """Multiple joint translations per region + held-out spatial agreement.

    Returns region diagnostics, never a video score. Each fold selects using
    only its own training cost; the complementary loss is reported afterwards.
    Region index is not an object ID. No propagation from a paired clean clip.
    """
    a, b, weights = np.asarray(source, np.float32), np.asarray(target, np.float32), np.asarray(weights, np.float32)
    if (a.shape != b.shape or a.ndim != 3 or min(a.shape[:2]) < 2
            or weights.ndim != 2 or weights.shape[1] != a.shape[0] * a.shape[1]
            or not all(np.isfinite(x).all() for x in (a, b, weights)) or (weights < 0).any()):
        raise ValueError("finite aligned descriptors and nonnegative region weights required")
    norms = [np.linalg.norm(x, axis=-1, keepdims=True) for x in (a, b)]
    if any((n <= 0).any() for n in norms):
        raise ValueError("zero descriptors are missing evidence, not stationary motion")
    a, b = a / norms[0], b / norms[1]
    h, w, c = a.shape
    offsets, table, in_bounds = displacement_table(a.reshape(-1, c) @ b.reshape(-1, c).T, (h, w))
    yy, xx = np.mgrid[:h, :w]
    parity = ((xx + yy) % 2).ravel()
    combined = np.concatenate((weights, weights * (parity == 0), weights * (parity == 1)))
    mass = combined.sum(axis=-1)
    observed_mass = combined @ in_bounds
    costs = combined @ table / np.maximum(observed_mass, 1e-12)
    overlap = observed_mass / np.maximum(mass[:, None], 1e-12)
    records = []
    for i in range(len(weights)):
        record = {"region": i, "patch_mass": float(mass[i]), "hypotheses": [], "folds": []}
        for fold in range(3):
            index = i + fold * len(weights)
            if mass[index] < config.minimum_fold_mass:
                result = {"displacement_grid": None, "loss": None, "overlap": 0., "rank": 0,
                          "reason": "insufficient_descriptor_support"}
                hypotheses = []
            else:
                peaks = _peaks(costs[index], overlap[index], offsets, config)
                hypotheses = [refine_translation(a, b, combined[index], offsets[p], config) for p in peaks]
                hypotheses = [x for x in hypotheses if x["loss"] is not None]
                if hypotheses:
                    best = min(range(len(hypotheses)), key=lambda j: hypotheses[j]["loss"])
                    result = dict(hypotheses[best])
                    losses = sorted(x["loss"] for x in hypotheses)
                    result["alternative_loss_gap"] = losses[1] - losses[0] if len(losses) > 1 else None
                else:
                    result = {"displacement_grid": None, "loss": None, "overlap": 0., "rank": 0,
                              "reason": "insufficient_overlap"}
            if fold == 0:
                record.update(full=result, hypotheses=hypotheses)
            else:
                d = result["displacement_grid"]
                held = weights[i] * (parity != fold - 1)
                error, held_overlap = alignment_error(a, b, held, d) if d is not None else (None, 0.)
                record["folds"].append({**result, "heldout_loss": error, "heldout_overlap": held_overlap})
        fold_vectors = [x["displacement_grid"] for x in record["folds"]]
        record["fold_disagreement_grid"] = float(np.linalg.norm(np.subtract(*fold_vectors))) if all(x is not None for x in fold_vectors) else None
        records.append(record)
    return records
