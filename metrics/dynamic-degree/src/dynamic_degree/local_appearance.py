"""Native masked local appearance relocation, without a temporal motion prior.

The local search is global in displacement. Correlation only supplies competing
appearance matches, never a physical-motion certificate or Dynamic score.
"""
from __future__ import annotations

import cv2
import numpy as np

from .native_region_motion import rgb_float, _mask


def feature_support(mask, xy, size, factor):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("2D source region required")
    mask = _mask(mask, mask.shape)
    xy = np.asarray(xy, float)
    if (xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(size) or size <= 0
            or not np.isfinite(factor) or factor <= 0):
        raise ValueError("finite center and positive feature-relative support required")
    yy, xx = np.indices(mask.shape)
    return mask & ((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= (factor * size) ** 2)


def correlate_full(target, kernel):
    """Float64 finite correlation, including every partially visible placement.

    OpenCV's double-precision filtering implements the finite sums; padding is
    later excluded by the explicit visible-support mass, never an image cost.
    """
    target, kernel = np.asarray(target, float), np.asarray(kernel, float)
    if target.ndim != 2 or kernel.ndim != 2 or min(*target.shape, *kernel.shape) < 1:
        raise ValueError("nonempty 2D arrays required")
    h, w = kernel.shape
    padded = np.pad(target, ((h - 1, h - 1), (w - 1, w - 1)))
    result = cv2.filter2D(padded, cv2.CV_64F, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    return result[:target.shape[0] + h - 1, :target.shape[1] + w - 1]


def ncc_surface(source, target, support):
    """Per-channel mean removal, then joint RGB normalized correlation.

    Missing/flat support is unobservable, not correlation=1. The variance floor
    is scaled floating-point roundoff, not a pixel displacement or CF amplitude.
    """
    a, b = rgb_float(source), rgb_float(target)
    if a.shape != b.shape:
        raise ValueError("matching native image geometry required")
    support = _mask(support, a.shape[:2])
    yy, xx = np.nonzero(support)
    if len(xx) < 3:
        return None
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    m, patch = support[y0:y1, x0:x1].astype(float), a[y0:y1, x0:x1]
    ones = np.ones(a.shape[:2])
    mass = np.clip(correlate_full(ones, m), 0, m.sum())
    safe_mass = np.maximum(mass, 1.)
    source_energy = correlate_full(ones, np.sum(patch * patch, axis=-1) * m)
    target_energy = correlate_full(np.sum(b * b, axis=-1), m)
    covariance = np.zeros_like(mass)
    source_var, target_var = source_energy.copy(), target_energy.copy()
    for channel in range(3):
        weighted_source = patch[..., channel] * m
        sum_a = correlate_full(ones, weighted_source)
        sum_b = correlate_full(b[..., channel], m)
        covariance += correlate_full(b[..., channel], weighted_source) - sum_a * sum_b / safe_mass
        source_var -= sum_a * sum_a / safe_mass
        target_var -= sum_b * sum_b / safe_mass
    floor = 128 * np.finfo(float).eps * np.maximum(1., np.maximum(source_energy, target_energy))
    valid = (mass >= 3 - 1e-9) & (source_var > floor) & (target_var > floor)
    denominator = np.sqrt(np.maximum(source_var * target_var, 0))
    correlation = np.full(mass.shape, np.nan)
    np.divide(covariance, denominator, out=correlation, where=valid)
    correlation[valid] = np.clip(correlation[valid], -1., 1.)
    dy, dx = np.mgrid[-m.shape[0] + 1:a.shape[0], -m.shape[1] + 1:a.shape[1]]
    return {"correlation": correlation, "overlap": mass / m.sum(), "dx": dx - x0, "dy": dy - y0,
            "source_variance": np.maximum(source_var, 0) / safe_mass,
            "target_variance": np.maximum(target_var, 0) / safe_mass}


def correlation_peaks(surface, hypotheses=3, min_overlap=.8):
    if not isinstance(hypotheses, int) or hypotheses < 1 or not np.isfinite(min_overlap) or not 0 < min_overlap <= 1:
        raise ValueError("positive number of hypotheses and valid visibility fraction required")
    if surface is None:
        return []
    available = np.isfinite(surface["correlation"]) & (surface["overlap"] >= min_overlap - 1e-10)
    result = []
    for _ in range(hypotheses):
        if not available.any():
            break
        y, x = np.unravel_index(np.argmax(np.where(available, surface["correlation"], -np.inf)), available.shape)
        dx, dy = int(surface["dx"][y, x]), int(surface["dy"][y, x])
        result.append({"displacement_pixels": [dx, dy], "correlation": float(surface["correlation"][y, x]),
                       "overlap": float(surface["overlap"][y, x]),
                       "source_variance": float(surface["source_variance"][y, x]),
                       "target_variance": float(surface["target_variance"][y, x])})
        # Numerical neighboring integer samples of one mode, not a motion floor.
        available &= np.maximum(abs(surface["dx"] - dx), abs(surface["dy"] - dy)) > 1
    return result


def relocate_support(source, target, support, *, hypotheses=3, min_overlap=.8, reverse=True):
    forward = correlation_peaks(ncc_surface(source, target, support), hypotheses, min_overlap)
    for proposal in forward:
        dx, dy = proposal["displacement_pixels"]
        shifted = cv2.warpAffine(np.asarray(support, np.uint8), np.array([[1., 0., dx], [0., 1., dy]]),
                                (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT).astype(bool)
        proposal["reverse_candidates"] = []
        if reverse:
            back = correlation_peaks(ncc_surface(target, source, shifted), hypotheses, min_overlap)
            for candidate in back:
                candidate["closure_error_pixels"] = float(np.linalg.norm(np.array(candidate["displacement_pixels"]) + [dx, dy]))
            proposal["reverse_candidates"] = back
    return {"status": "diagnostic_only", "score": None, "reliability": "NOT CERTIFIED",
            "support_pixels": int(np.asarray(support, bool).sum()), "hypotheses": forward}


def distinct_correlation_peaks(surface, hypotheses=12, min_overlap=.8, *, allowed=None):
    """Actual discrete local maxima, not adjacent shoulders of the same peak.

    The legacy greedy 1px suppression can spend all hypotheses on samples of a
    broad peak. Here a 3x3 maximum test precedes ranking; a connected plateau
    contributes one deterministic location. This is not a motion threshold or
    a guarantee that repeated visual identities have been distinguished.
    """
    if not isinstance(hypotheses, int) or hypotheses < 1 or not 0 < min_overlap <= 1:
        raise ValueError("positive hypothesis count and valid overlap required")
    if surface is None:
        return []
    valid = np.isfinite(surface["correlation"]) & (surface["overlap"] >= min_overlap - 1e-10)
    values = np.where(valid, surface["correlation"], -2.)
    local = cv2.dilate(values, np.ones((3, 3), np.uint8))
    maxima = valid & (values >= local - 1e-12)
    if allowed is not None:
        allowed = np.asarray(allowed, bool)
        if allowed.shape != valid.shape:
            raise ValueError("peak permission mask shape differs")
        # Exclusion must not manufacture peaks on a footprint boundary.
        maxima &= allowed
    count, labels = cv2.connectedComponents(maxima.astype(np.uint8), connectivity=8)
    candidates = []
    pixels = np.flatnonzero(maxima)
    if len(pixels):
        groups = labels.ravel()[pixels]
        order = np.lexsort((pixels, -values.ravel()[pixels], groups))
        _, first = np.unique(groups[order], return_index=True)
        counts = np.bincount(groups, minlength=count)
        for position in order[first]:
            y, x = np.unravel_index(pixels[position], values.shape)
            candidates.append((float(values[y, x]), int(y), int(x), int(counts[groups[position]])))
    candidates.sort(key=lambda v: (-v[0], v[1], v[2]))
    return [{"displacement_pixels": [int(surface["dx"][y, x]), int(surface["dy"][y, x])],
             "correlation": value, "overlap": float(surface["overlap"][y, x]),
             "source_variance": float(surface["source_variance"][y, x]),
             "target_variance": float(surface["target_variance"][y, x]),
             "plateau_pixels": size, "reverse_candidates": []}
            for value, y, x, size in candidates[:hypotheses]]


def appearance_ambiguity(source, target, support, *, hypotheses=12, min_overlap=.8):
    """Source self-similarity and distinct cross-frame alternatives.

    Support intersection thresholds .25/.5/.75 exclude the source's own
    immediate footprint, not a known pixel displacement. All three are reported.
    A cross-frame match weaker than a remote same-frame lookalike is ambiguous;
    the converse is not proof of physical identity. Reverse tracking is NOT RUN
    for this distinct-peak ablation; legacy reverse checks are not transferred.
    """
    support = _mask(support, np.asarray(source).shape[:2])
    cross = ncc_surface(source, target, support)
    auto = ncc_surface(source, source, support)
    proposals = distinct_correlation_peaks(cross, hypotheses, min_overlap)
    result = {"status": "diagnostic_only", "score": None, "reliability": "NOT CERTIFIED",
              "support_pixels": int(support.sum()), "hypotheses": proposals,
              "legacy_greedy_hypotheses": correlation_peaks(cross, 3, min_overlap),
              "self_identity_correlation": None, "self_alternatives": [], "reverse_check": "NOT RUN",
              "warning": "source uniqueness and distinct hypotheses are not physical-motion certification"}
    if auto is None:
        return result
    yy, xx = np.nonzero(support)
    kernel = support[yy.min():yy.max() + 1, xx.min():xx.max() + 1].astype(float)
    intersection = np.clip(correlate_full(support.astype(float), kernel) / support.sum(), 0., 1.)
    identity = (auto["dx"] == 0) & (auto["dy"] == 0)
    zero = auto["correlation"][identity]
    if len(zero) != 1:
        raise ValueError("self-correlation origin missing or duplicated")
    result["self_identity_correlation"] = float(zero[0]) if np.isfinite(zero[0]) else None
    for fraction in (.25, .5, .75):
        alternatives = distinct_correlation_peaks(auto, 3, min_overlap, allowed=intersection <= fraction)
        result["self_alternatives"].append({"maximum_support_intersection": fraction, "hypotheses": alternatives,
                                            "cross_top1_minus_self_alternative": proposals[0]["correlation"] - alternatives[0]["correlation"] if proposals and alternatives else None})
    return result


def consensus_hypotheses(entries, shape, *, intersection=.5, power=1, radius=1, hypotheses=3):
    """DEV regional translation proposals from self-ambiguity-weighted votes.

    Each distinct queried location can support multiple appearance hypotheses,
    but contributes at most its best margin at a tested integer displacement.
    Multiple feature scales at the same location are averaged, not extra votes.
    Margin = max(cross NCC - remote same-frame NCC, 0); linear and squared
    margins are explicit ablations, not calibrated likelihoods. A one-pixel
    integer-grid tolerance is not a motion floor. No velocity is thresholded.

    Returns proposals only; even agreeing folds can share an incorrect identity.
    """
    if (len(shape) != 2 or min(shape) < 2 or intersection not in {.25, .5, .75}
            or power not in {1, 2} or not isinstance(radius, int) or radius < 0
            or not isinstance(hypotheses, int) or hypotheses < 1):
        raise ValueError("valid native geometry and explicit consensus ablation required")
    grouped = {}
    for entry in entries:
        xy = tuple(map(float, entry["xy"]))
        if len(xy) != 2 or not np.isfinite(xy).all():
            raise ValueError("finite source point required")
        grouped.setdefault(xy, []).append(entry["appearance"])
    positions = np.array(sorted(grouped), float).reshape(-1, 2)
    h, w = map(int, shape)
    votes_shape = (2 * h - 1, 2 * w - 1)
    contributions = []
    oy, ox = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    offsets = np.c_[ox.ravel(), oy.ravel()]
    missing, zero_margin = 0, 0
    for xy in map(tuple, positions):
        records = grouped[xy]
        cells, values = [], []
        available = 0
        for appearance in records:
            alternatives = next((a["hypotheses"] for a in appearance["self_alternatives"]
                                 if a["maximum_support_intersection"] == intersection), [])
            if not alternatives or not appearance["hypotheses"]:
                continue
            available += 1
            competitor = alternatives[0]["correlation"]
            d = np.array([p["displacement_pixels"] for p in appearance["hypotheses"]], int)
            margin = np.maximum(np.array([p["correlation"] for p in appearance["hypotheses"]]) - competitor, 0.) ** power
            pixels = d[:, None, :] + offsets[None] + [w - 1, h - 1]
            weights = np.broadcast_to(margin[:, None], pixels.shape[:2]).ravel()
            pixels = pixels.reshape(-1, 2)
            inside = (pixels >= 0).all(axis=1) & (pixels < [votes_shape[1], votes_shape[0]]).all(axis=1) & (weights > 0)
            flat = pixels[inside, 1] * votes_shape[1] + pixels[inside, 0]
            unique, inverse = np.unique(flat, return_inverse=True)
            local = np.zeros(len(unique)); np.maximum.at(local, inverse, weights[inside])
            cells.extend(unique); values.extend(local / len(records))
        if not available:
            missing += 1
        unique, inverse = np.unique(cells, return_inverse=True)
        combined = np.zeros(len(unique)); np.add.at(combined, inverse, values)
        if available and not len(unique):
            zero_margin += 1
        contributions.append((unique.astype(int), combined))
    results = []
    yy, xx = np.mgrid[:votes_shape[0], :votes_shape[1]]
    for fold in ("all", "even_location", "odd_location"):
        selected = np.ones(len(positions), bool) if fold == "all" else np.arange(len(positions)) % 2 == (fold == "odd_location")
        votes = np.zeros(np.prod(votes_shape))
        for i in np.flatnonzero(selected):
            cells, weights = contributions[i]
            np.add.at(votes, cells, weights)
        # The peak routine also serves unnormalized vote surfaces; variance
        # placeholders are not evidence and are stripped from reported peaks.
        surface = {"correlation": np.where(votes.reshape(votes_shape) > 0, votes.reshape(votes_shape), np.nan),
                   "overlap": np.ones(votes_shape), "dx": xx - w + 1, "dy": yy - h + 1,
                   "source_variance": np.zeros(votes_shape), "target_variance": np.zeros(votes_shape)}
        peaks = []
        for peak in distinct_correlation_peaks(surface, hypotheses):
            dx, dy = peak["displacement_pixels"]
            flat = (dy + h - 1) * votes_shape[1] + dx + w - 1
            supporters = [int(i) for i in np.flatnonzero(selected) if flat in contributions[i][0]]
            points = positions[supporters]
            rank = int(np.linalg.matrix_rank(np.c_[np.ones(len(points)), points - points.mean(axis=0)])) if len(points) else 0
            peaks.append({"displacement_pixels": [dx, dy], "vote_weight": peak["correlation"],
                          "supporting_locations": len(supporters), "source_positions": points.tolist(), "spatial_design_rank": rank})
        results.append({"fold": fold, "source_locations": int(selected.sum()), "peaks": peaks})
    return {"status": "diagnostic_only", "score": None, "intersection": intersection, "power": power,
            "integer_localization_radius": radius, "distinct_source_locations": len(positions),
            "missing_appearance_locations": missing, "zero_positive_margin_locations": zero_margin, "folds": results,
            "warning": "query-location folds are disjoint but patches may overlap; consensus is not physical-motion truth"}


def bilinear_rgb_gradient(image, xy):
    """Exact piecewise-bilinear values/derivatives at in-image coordinates."""
    image, xy = np.asarray(image, float), np.asarray(xy, float)
    if (image.ndim != 3 or image.shape[-1] != 3 or xy.ndim != 2 or xy.shape[1] != 2
            or not np.isfinite(xy).all() or (xy < 0).any()
            or (xy[:, 0] >= image.shape[1] - 1).any() or (xy[:, 1] >= image.shape[0] - 1).any()):
        raise ValueError("finite interior coordinates and RGB image required")
    x, y = np.floor(xy).astype(int).T
    u, v = (xy - np.c_[x, y]).T[:, :, None]
    a, b, c, d = image[y, x], image[y, x + 1], image[y + 1, x], image[y + 1, x + 1]
    values = (1 - u) * (1 - v) * a + u * (1 - v) * b + (1 - u) * v * c + u * v * d
    dx = (1 - v) * (b - a) + v * (d - c)
    dy = (1 - u) * (c - a) + u * (d - b)
    return values, np.stack((dx, dy), axis=-1)


def refine_local_appearance(source, target, support, displacement, *, model="affine", iterations=30, min_overlap=.8):
    """Photometric local refinement; deformation is an estimated parameter.

    The initial in-view support is frozen, so shrinking the visible subset cannot
    reduce the objective. A positive-determinant affine map permits rotation,
    scale and shear; no known jitter amplitude/frequency enters the optimization.
    Convergence/low error still does not certify the physical correspondence.
    """
    a, b = rgb_float(source), rgb_float(target)
    support = _mask(support, a.shape[:2])
    d = np.asarray(displacement, float)
    if (a.shape != b.shape or d.shape != (2,) or not np.isfinite(d).all()
            or model not in {"translation", "affine"} or not isinstance(iterations, int) or iterations < 1
            or not np.isfinite(min_overlap) or not 0 < min_overlap <= 1):
        raise ValueError("aligned images, finite displacement and supported refinement model required")
    y, x = np.nonzero(support)
    xy = np.c_[x, y].astype(float)
    initial = xy + d
    visible = (initial >= 0).all(axis=1) & (initial < [b.shape[1] - 1, b.shape[0] - 1]).all(axis=1)
    result = {"status": "insufficient_evidence", "score": None, "reliability": "NOT CERTIFIED", "model": model,
              "initial_displacement_pixels": d.tolist(), "support_pixels": len(xy), "fixed_visible_pixels": int(visible.sum())}
    if visible.sum() < max(8, min_overlap * len(xy)):
        return result | {"reason": "insufficient_fixed_visible_support"}
    xy, values = xy[visible], a[y[visible], x[visible]]
    center = xy.mean(axis=0)
    radius = max(float(np.sqrt(np.mean(np.sum((xy - center) ** 2, axis=1)))), 1.)
    design = np.c_[np.ones(len(xy)), (xy - center) / radius] if model == "affine" else np.ones((len(xy), 1))
    ac = values - values.mean(axis=0)
    norm_a = float(np.linalg.norm(ac))
    if norm_a <= 1e-10:
        return result | {"reason": "flat_source"}
    ac /= norm_a
    beta = np.zeros((design.shape[1], 2))
    beta[0] = d

    def evaluate(parameters):
        positions = xy + design @ parameters
        if (positions < 0).any() or (positions >= [b.shape[1] - 1, b.shape[0] - 1]).any():
            return None
        linear = np.eye(2) + parameters[1:].T / radius if model == "affine" else np.eye(2)
        if np.linalg.det(linear) <= 0:
            return None
        predicted, gradients = bilinear_rgb_gradient(b, positions)
        centered = predicted - predicted.mean(axis=0)
        norm_b = float(np.linalg.norm(centered))
        if norm_b <= 1e-10:
            return None
        normalized = centered / norm_b
        residual = normalized - ac
        jac = np.einsum("ncd,nk->nckd", gradients, design).reshape(len(xy), 3, -1)
        jac -= jac.mean(axis=0)
        jac = (jac - normalized[..., None] * np.sum(normalized[..., None] * jac, axis=(0, 1))) / norm_b
        return float(np.sum(residual ** 2) / 2), residual.ravel(), jac.reshape(-1, jac.shape[-1]), linear

    current = evaluate(beta)
    if current is None:
        return result | {"reason": "unobservable_target"}
    initial_cost = current[0]
    converged, attempted = False, 0
    for attempted in range(1, iterations + 1):
        cost, residual, jac, _ = current
        if np.linalg.matrix_rank(jac) < jac.shape[1]:
            break
        step = np.linalg.lstsq(jac, -residual, rcond=None)[0].reshape(beta.shape)
        accepted = False
        for power in range(12):
            delta = step * 2. ** -power
            trial = evaluate(beta + delta)
            if trial is not None and trial[0] <= cost:
                beta += delta; current = trial; accepted = True
                converged = np.sqrt(np.mean(np.sum((design @ delta) ** 2, axis=1))) < .01
                break
        if not accepted or converged:
            break
    cost, _, jac, linear = current
    # q = A*p + b in the original native-pixel coordinate system.
    matrix = np.c_[linear, center + beta[0] - linear @ center]
    return result | {"status": "diagnostic_only", "affine_matrix": matrix.tolist(), "source_center_xy": center.tolist(),
                     "center_displacement_pixels": beta[0].tolist(), "initial_correlation": 1 - initial_cost,
                     "correlation": 1 - cost, "positive_determinant": float(np.linalg.det(linear)),
                     "linear_singular_values": np.linalg.svd(linear, compute_uv=False).tolist(),
                     "jacobian_rank": int(np.linalg.matrix_rank(jac)), "parameters": jac.shape[1],
                     "converged": bool(converged), "iterations": attempted,
                     "warning": "fitted on the same pixels; low loss/convergence is not held-out geometric truth"}
