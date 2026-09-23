"""Disjoint spatial witnesses for local appearance motion proposals.

The seed template alone proposes displacements. Neither witness annulus enters
the search. This is not statistical independence (SAM/SIFT and target images
are shared), and a translation failing on context is NOT a motion veto: a real
articulated part or an occlusion can fail as well.
"""
from __future__ import annotations

import numpy as np

from .local_appearance import feature_support
from .native_region_motion import _mask


def context_masks(region, xy, size, factor):
    """Core plus two disjoint, feature-relative same-region annuli."""
    supports = [feature_support(region, xy, size, factor * multiple) for multiple in (1, 2, 4)]
    return supports[0], (supports[1] & ~supports[0], supports[2] & ~supports[1])


def evidence(source, target, support, displacements, *, min_overlap=.8, batch=16):
    """NCC and RGB MSE for each integer hypothesis vs its zero-warp control.

    Inputs are already decoded float64 RGB [0,1], not a resampled video. Each
    candidate/control comparison uses exactly the same visible source pixels.
    Different hypotheses can have different visible sets; fractions stay saved.
    Flat/insufficient support stays NaN, including NCC for a flat source frame.
    """
    a, b, d = np.asarray(source), np.asarray(target), np.asarray(displacements)
    if (a.dtype != np.float64 or b.dtype != np.float64 or a.ndim != 3 or a.shape[-1] != 3 or a.shape != b.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all() or a.min() < 0 or max(a.max(), b.max()) > 1 or b.min() < 0
            or d.ndim != 2 or d.shape[1] != 2 or not np.isfinite(d).all() or not np.array_equal(d, np.rint(d))
            or not 0 < min_overlap <= 1 or not isinstance(batch, int) or batch < 1):
        raise ValueError("aligned float64 RGB, integer hypotheses and valid visibility/batch required")
    support = _mask(support, a.shape[:2])
    yy, xx = np.nonzero(support)
    result = {name: np.full(len(d), np.nan) for name in ("ncc", "zero_ncc", "mse", "zero_mse")}
    result["overlap"] = np.zeros(len(d))
    if len(xx) < 3:
        return result
    d = d.astype(int)
    for start in range(0, len(d), batch):
        shifts = d[start:start + batch]
        n = len(shifts)
        mass = np.zeros(n); sums = np.zeros((n, 3, 3)); energies = np.zeros((n, 3)); products = np.zeros((n, 2))
        for offset in range(0, len(xx), 8192):
            x, y = xx[offset:offset + 8192], yy[offset:offset + 8192]
            tx, ty = x[None] + shifts[:, 0, None], y[None] + shifts[:, 1, None]
            visible = (tx >= 0) & (tx < a.shape[1]) & (ty >= 0) & (ty < a.shape[0])
            values = np.stack((np.broadcast_to(a[y, x], (n, len(x), 3)),
                               b[np.clip(ty, 0, a.shape[0] - 1), np.clip(tx, 0, a.shape[1] - 1)],
                               np.broadcast_to(b[y, x], (n, len(x), 3))), axis=1) * visible[:, None, :, None]
            mass += visible.sum(axis=1)
            sums += values.sum(axis=2)
            energies += (values ** 2).sum(axis=(2, 3))
            products += np.stack([(values[:, 0] * values[:, other]).sum(axis=(1, 2)) for other in (1, 2)], axis=1)
        safe = np.maximum(mass, 1.)
        fraction = mass / len(xx)
        observed = (mass >= 3) & (fraction >= min_overlap)
        variance = energies - (sums ** 2).sum(axis=2) / safe[:, None]
        floor = 128 * np.finfo(float).eps * np.maximum(1., energies)
        result["overlap"][start:start + n] = fraction
        for p, other in enumerate((1, 2)):
            prefix = "" if other == 1 else "zero_"
            mse = np.maximum(0., energies[:, 0] + energies[:, other] - 2 * products[:, p]) / (3 * safe)
            result[prefix + "mse"][start:start + n] = np.where(observed, mse, np.nan)
            valid = observed & (variance[:, 0] > floor[:, 0]) & (variance[:, other] > floor[:, other])
            cov = products[:, p] - (sums[:, 0] * sums[:, other]).sum(axis=1) / safe
            ncc = np.full(n, np.nan)
            ncc[valid] = np.clip(cov[valid] / np.sqrt(variance[valid, 0] * variance[valid, other]), -1., 1.)
            result[prefix + "ncc"][start:start + n] = ncc
    return result


def rankings(core, witnesses, *, top=3):
    """Keep core-only, each core+witness and both-witness comparisons explicit.

    An unobservable witness has no ranking; it never promotes the zero control.
    Scores here are correlation rankings only, not a Dynamic Degree metric.
    """
    if len(witnesses) != 2:
        raise ValueError("two disjoint witness annuli required")
    all_values = [core["ncc"], *(w["ncc"] for w in witnesses)]
    result = []
    for name, indices in (("core", [0]), ("core_inner", [0, 1]), ("core_outer", [0, 2]), ("core_both", [0, 1, 2])):
        merit = np.minimum.reduce([all_values[i] for i in indices])
        observed = np.flatnonzero(np.isfinite(merit))
        ordered = observed[np.argsort(-merit[observed], kind="stable")[:top]]
        result.append({"variant": name, "observable_hypotheses": len(observed), "ranked_indices": ordered.tolist(),
                       "ranked_ncc": merit[ordered].tolist(), "score": None})
    return result
