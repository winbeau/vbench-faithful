"""Uncalibrated descriptor correspondence diagnostics, NOT a motion Repair.

Mutuality and peak curvature are evidence, not certificates of true motion.
Repeated texture/parts can be mutually wrong; unresolved matching stays visible.
No construction labels, paired originals, prompt or intervention constants.
"""
from __future__ import annotations

import numpy as np


def peak_refinement(similarity, best, grid_shape):
    """2-D quadratic local maximum; return uncertainty flags, never forced zero.

    A non-concave/boundary peak retains its coarse match but is explicitly not
    subpixel resolved. No smoothing of time, spatial bin snapping, or temperature.
    """
    s = np.asarray(similarity, float)
    best = np.asarray(best)
    h, w = map(int, grid_shape)
    if s.ndim != 2 or s.shape[1] != h * w or best.shape != (len(s),) or min(h, w) < 1:
        raise ValueError("similarity/argmax/grid geometry mismatch")
    if not np.isfinite(s).all() or not np.issubdtype(best.dtype, np.integer) or (best < 0).any() or (best >= h * w).any():
        raise ValueError("finite similarity and in-range integer matches required")
    y, x = np.divmod(best, w)
    inside = (y > 0) & (y < h - 1) & (x > 0) & (x < w - 1)
    offsets = np.zeros((len(s), 2))
    resolved = np.zeros(len(s), bool)
    curvature = np.zeros(len(s))
    for i in np.flatnonzero(inside):
        j = best[i]
        gx = (s[i, j + 1] - s[i, j - 1]) / 2
        gy = (s[i, j + w] - s[i, j - w]) / 2
        xx = s[i, j + 1] + s[i, j - 1] - 2 * s[i, j]
        yy = s[i, j + w] + s[i, j - w] - 2 * s[i, j]
        cross = (s[i, j + w + 1] - s[i, j + w - 1] - s[i, j - w + 1] + s[i, j - w - 1]) / 4
        matrix = np.array([[xx, cross], [cross, yy]])
        eigenvalues = np.linalg.eigvalsh(matrix)
        curvature[i] = -eigenvalues[-1]
        if eigenvalues[-1] >= -1e-8:
            continue
        delta = -np.linalg.solve(matrix, [gx, gy])
        if (abs(delta) <= 1).all():
            offsets[i] = delta
            resolved[i] = True
    return offsets, resolved, curvature


def match_similarity(similarity, queries, grid_shape):
    s, q = np.asarray(similarity, float), np.asarray(queries, float)
    h, w = map(int, grid_shape)
    if s.shape != (len(q), len(q)) or q.shape != (h * w, 2) or h < 2 or w < 2 or not np.isfinite(q).all():
        raise ValueError("square similarity and rectangular regular query grid required")
    if not np.isfinite(s).all():
        raise ValueError("nonfinite similarity")
    step = np.array([q[1, 0] - q[0, 0], q[w, 1] - q[0, 1]])
    yy, xx = np.mgrid[:h, :w]
    expected = q[0] + np.stack((xx, yy), axis=-1).reshape(-1, 2) * step
    if (step <= 0).any() or not np.allclose(q, expected):
        raise ValueError("increasing row-major rectangular grid required")
    forward, reverse = s.argmax(axis=1), s.argmax(axis=0)
    top2 = np.partition(s, -2, axis=1)[:, -2:]
    offset, refined, curvature = peak_refinement(s, forward, grid_shape)
    reverse_offset, reverse_refined, reverse_curvature = peak_refinement(s.T, reverse, grid_shape)
    source = np.arange(len(q))
    # Exact ties are ambiguous even when deterministic argmax happens to cycle.
    forward_unique = (s == s[source, forward, None]).sum(axis=1) == 1
    reverse_unique = (s == s[reverse, source][None]).sum(axis=0) == 1
    mutual = (reverse[forward] == source) & forward_unique & reverse_unique[forward]
    # A one-sided parabola can shift an exactly self-matching descriptor when
    # its neighboring correlations are asymmetric. Reciprocal offsets cancel
    # this positional bias for identical inputs instead of inventing motion.
    resolved = mutual & refined & reverse_refined[forward]
    symmetric_offset = np.where(resolved[:, None], (offset - reverse_offset[forward]) / 2, 0.)
    return {"target_index": forward, "target_xy": q[forward] + symmetric_offset * step,
            "coarse_target_xy": q[forward], "mutual": mutual, "subpixel_resolved": refined,
            "reciprocal_subpixel_resolved": resolved,
            "oneway_target_xy": q[forward] + offset * step,
            "cosine": s[source, forward], "margin": top2[:, 1] - top2[:, 0],
            "stationary_cosine": np.diag(s), "peak_curvature": curvature,
            "reverse_peak_curvature": reverse_curvature[forward],
            "cycle_error_pixels": np.linalg.norm(q[reverse[forward]] - q, axis=-1)}


def pair_diagnostics(match, queries, attention, seconds, short_side):
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(short_side) or short_side <= 0:
        raise ValueError("positive time interval and short side required")
    q, attention = np.asarray(queries, float), np.asarray(attention, float)
    if attention.shape != (len(q),) or not np.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("finite nonnegative per-query attention required")
    distance = np.linalg.norm(match["target_xy"] - q, axis=-1)
    coarse = np.linalg.norm(match["coarse_target_xy"] - q, axis=-1)
    result = {"all_refined_or_coarse_speed": float(distance.mean() / seconds / short_side),
              "all_coarse_speed": float(coarse.mean() / seconds / short_side),
              "mutual_fraction": float(match["mutual"].mean()),
              "subpixel_resolved_fraction": float(match["reciprocal_subpixel_resolved"].mean()),
              "coarse_zero_fraction": float(np.mean(coarse == 0)),
              "median_margin": float(np.median(match["margin"])),
              "median_cosine": float(np.median(match["cosine"])),
              "median_cycle_pixels": float(np.median(match["cycle_error_pixels"]))}
    for name, keep in (("mutual", match["mutual"]),
                       ("mutual_resolved", match["reciprocal_subpixel_resolved"])):
        result[f"{name}_conditional_speed"] = float(distance[keep].mean() / seconds / short_side) if keep.any() else None
        weight = attention * keep
        result[f"{name}_attention_conditional_speed"] = float(np.sum(distance * weight) / weight.sum() / seconds / short_side) if weight.sum() > 0 else None
        result[f"{name}_fraction"] = float(keep.mean())
    return result


def transitivity_diagnostics(first, second, direct, queries):
    """Adjacent composition vs direct two-frame-interval matches, all phases.

    Disagreement is explicit; agreeing matches may still be wrong. This check
    is not a motion estimate made by dropping alternate frames.
    """
    q = np.asarray(queries, float)
    middle = first["target_index"]
    composed = second["target_index"][middle]
    error = np.linalg.norm(q[composed] - direct["coarse_target_xy"], axis=-1)
    mutual = first["mutual"] & second["mutual"][middle] & direct["mutual"]
    return {"all_mean_error_pixels": float(error.mean()), "mutual_triple_fraction": float(mutual.mean()),
            "mutual_conditional_mean_error_pixels": float(error[mutual].mean()) if mutual.any() else None}
