"""Native line-profile and junction correspondence, not a motion score.

A junction is the intersection of two matched visible line segments. Its
location can move along a long boundary whose own endpoints are stationary;
no common translation is imposed on an entire region or neighboring parts.
"""
from __future__ import annotations

import cv2
import numpy as np

from .native_region_motion import rgb_float


def sample_strips(frame, lines, widths, *, along=32, across=9, width_factor=3.):
    image = rgb_float(frame)
    lines, widths = np.asarray(lines, float), np.asarray(widths, float)
    if (lines.ndim != 3 or lines.shape[1:] != (2, 2) or widths.shape != (len(lines),)
            or not np.isfinite(lines).all() or not np.isfinite(widths).all() or (widths <= 0).any()
            or not isinstance(along, int) or not isinstance(across, int) or min(along, across) < 2
            or not np.isfinite(width_factor) or width_factor <= 0):
        raise ValueError("finite segments, positive widths and valid strip geometry required")
    strips = np.empty((len(lines), along, across, 3), float)
    visible = np.empty((len(lines), along, across), bool)
    direction = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(direction, axis=1)
    if (length <= 0).any():
        raise ValueError("nonzero length segments required")
    normal = np.c_[-direction[:, 1], direction[:, 0]] / length[:, None]
    for i, line in enumerate(lines):
        coordinates = (line.mean(axis=0) + np.linspace(-.75, .75, along)[:, None, None] * direction[i]
                       + np.linspace(-1., 1., across)[None, :, None] * normal[i] * width_factor * widths[i])
        x, y = coordinates[..., 0], coordinates[..., 1]
        strips[i] = cv2.remap(image, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        visible[i] = (x >= 0) & (x <= image.shape[1] - 1) & (y >= 0) & (y <= image.shape[0] - 1)
    return strips, visible


def line_features(frame):
    rgb_float(frame)
    detected, widths, precision, _ = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    lines = np.empty((0, 2, 2)) if detected is None else detected.reshape(-1, 2, 2).astype(float)
    widths = np.empty(0) if widths is None else widths.ravel().astype(float)
    strips, visible = sample_strips(frame, lines, widths)
    return {"lines": lines, "widths": widths, "precision": np.empty(0) if precision is None else precision.ravel(),
            "strips": strips, "visible": visible}


def strip_correlations(a, ma, b, mb, *, min_overlap=.8, batch=64):
    """Joint RGB NCC with per-channel means on common canonical samples.

    Global line matching is tiled, not capped to a number of detections. Invalid
    image support is excluded explicitly, not correlated as padded black pixels.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = np.asarray(ma), np.asarray(mb)
    if (a.ndim != 4 or b.ndim != 4 or a.shape[1:] != b.shape[1:] or a.shape[-1] != 3
            or ma.shape != a.shape[:-1] or mb.shape != b.shape[:-1]
            or min(a.shape[1:3]) < 1 or ma.dtype != np.bool_ or mb.dtype != np.bool_
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not 0 < min_overlap <= 1 or not isinstance(batch, int) or batch < 1):
        raise ValueError("aligned RGB strips/visibility and valid batching required")
    pixels = int(np.prod(a.shape[1:3]))
    av, bv = a.reshape(len(a), pixels, 3), b.reshape(len(b), pixels, 3)
    am, bm = ma.reshape(len(a), pixels).astype(float), mb.reshape(len(b), pixels).astype(float)
    result = np.full((len(a), len(b)), np.nan)
    common = np.zeros_like(result)
    bmasked = bv * bm[..., None]
    be = (bv ** 2).sum(axis=2) * bm
    for start in range(0, len(a), batch):
        x, mask = av[start:start + batch], am[start:start + batch]
        mass = mask @ bm.T; safe = np.maximum(mass, 1.)
        xmasked = x * mask[..., None]
        ea = ((x ** 2).sum(axis=2) * mask) @ bm.T
        eb = mask @ be.T
        product = xmasked.reshape(len(x), -1) @ bmasked.reshape(len(b), pixels * 3).T
        va, vb, cov = ea.copy(), eb.copy(), product.copy()
        for channel in range(3):
            sa, sb = xmasked[:, :, channel] @ bm.T, mask @ bmasked[:, :, channel].T
            va -= sa ** 2 / safe; vb -= sb ** 2 / safe; cov -= sa * sb / safe
        floor = 128 * np.finfo(float).eps * np.maximum(1., np.maximum(ea, eb))
        fraction = mass / pixels
        valid = (mass >= 3) & (fraction >= min_overlap) & (va > floor) & (vb > floor)
        values = np.full(mass.shape, np.nan)
        values[valid] = np.clip(cov[valid] / np.sqrt(va[valid] * vb[valid]), -1., 1.)
        result[start:start + len(x)] = values; common[start:start + len(x)] = fraction
    return result, common


def reciprocal_lines(source, target, *, ratio=.75):
    if not 0 < ratio < 1:
        raise ValueError("ratio must be strictly between zero and one")
    plain, overlap = strip_correlations(source["strips"], source["visible"], target["strips"], target["visible"])
    reverse, ro = strip_correlations(source["strips"], source["visible"], target["strips"][:, ::-1, ::-1], target["visible"][:, ::-1, ::-1])
    flip = np.nan_to_num(reverse, nan=-2.) > np.nan_to_num(plain, nan=-2.)
    similarity = np.where(flip, reverse, plain)
    overlap = np.where(flip, ro, overlap)

    def neighbors(matrix):
        rows, columns = matrix.shape
        order = np.argsort(-np.nan_to_num(matrix, nan=-2.), axis=1, kind="stable")[:, :3]
        ratios = np.full(rows, np.nan)
        if columns >= 2:
            first, second = matrix[np.arange(rows), order[:, 0]], matrix[np.arange(rows), order[:, 1]]
            valid = np.isfinite(first) & np.isfinite(second)
            d1, d2 = np.maximum(2 - 2 * first, 0), np.maximum(2 - 2 * second, 0)
            ratios[valid] = np.sqrt(np.divide(d1[valid], d2[valid], out=np.ones(valid.sum()), where=d2[valid] > 1e-12))
        return order, ratios

    forward, fr = neighbors(similarity)
    backward, br = neighbors(similarity.T)
    candidates, matches = [], []
    for i in range(len(source["lines"])):
        choices = []
        for j in forward[i]:
            if not np.isfinite(similarity[i, j]):
                continue
            choices.append({"target_key": int(j), "correlation": float(similarity[i, j]), "reversed": bool(flip[i, j]), "common_overlap": float(overlap[i, j])})
        candidates.append({"source_key": i, "ratio": float(fr[i]) if np.isfinite(fr[i]) else None, "hypotheses": choices})
        if not choices:
            continue
        j = choices[0]["target_key"]
        if not (fr[i] < ratio and br[j] < ratio and backward[j, 0] == i):
            continue
        endpoints = target["lines"][j][::-1] if flip[i, j] else target["lines"][j]
        matches.append({"source_key": i, "target_key": j, "source_xy": source["lines"][i].mean(axis=0).tolist(),
                        "target_xy": endpoints.mean(axis=0).tolist(), "source_endpoints": source["lines"][i].tolist(), "target_endpoints": endpoints.tolist(),
                        "endpoint_displacements": (endpoints - source["lines"][i]).tolist(), "ratio": float(fr[i]), "reverse_ratio": float(br[j]), **choices[0]})
    return {"candidates": candidates, "matches": matches}


def segment_junctions(lines, widths, shape):
    """All nonparallel intersections within each segment + one measured width.

    No semantic object identity is inferred. Apparent crossings/occlusion edges
    remain possible. Near-parallel remote intersections fail finite extents,
    and rank degeneracy is handled as missing geometry rather than zero motion.
    """
    lines, widths = np.asarray(lines, float), np.asarray(widths, float)
    if (lines.ndim != 3 or lines.shape[1:] != (2, 2) or widths.shape != (len(lines),)
            or not np.isfinite(lines).all() or not np.isfinite(widths).all() or (widths <= 0).any()
            or len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)):
        raise ValueError("finite segments and positive widths required")
    direction = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(direction, axis=1)
    if (length <= 0).any():
        raise ValueError("nonzero segments required")
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    result = []
    for i in range(len(lines)):
        js = np.arange(i + 1, len(lines))
        delta = lines[js, 0] - lines[i, 0]
        denominator = cross(direction[i], direction[js])
        rank = abs(denominator) > 128 * np.finfo(float).eps * length[i] * length[js]
        t = np.divide(cross(delta, direction[js]), denominator, out=np.full(len(js), np.nan), where=rank)
        u = np.divide(cross(delta, direction[i]), denominator, out=np.full(len(js), np.nan), where=rank)
        xy = lines[i, 0] + t[:, None] * direction[i]
        visible = ((xy >= 0).all(axis=1) & (xy <= [shape[1] - 1, shape[0] - 1]).all(axis=1))
        accepted = (rank & visible & (t >= -widths[i] / length[i]) & (t <= 1 + widths[i] / length[i])
                    & (u >= -widths[js] / length[js]) & (u <= 1 + widths[js] / length[js]))
        for at in np.flatnonzero(accepted):
            result.append({"lines": [i, int(js[at])], "xy": xy[at].tolist(), "parameters": [float(t[at]), float(u[at])],
                           "sine_angle": float(abs(denominator[at]) / (length[i] * length[js[at]]))})
    return result


def junction_correspondences(source, target, matches):
    mapping = {m["source_key"]: m["target_key"] for m in matches}
    if len(mapping) != len(matches) or len(set(mapping.values())) != len(matches):
        raise ValueError("one-to-one line correspondences required")
    for junctions in (source, target):
        keys = [tuple(j["lines"]) for j in junctions]
        if len(set(keys)) != len(keys) or any(len(k) != 2 or k[0] >= k[1] for k in keys):
            raise ValueError("unique canonical line pairs required")
    targets = {tuple(j["lines"]): (i, j) for i, j in enumerate(target)}
    found, missing = [], []
    for i, junction in enumerate(source):
        a, b = junction["lines"]
        if a not in mapping or b not in mapping:
            missing.append({"source_key": i, "reason": "missing_line_correspondence"}); continue
        partner = targets.get(tuple(sorted((mapping[a], mapping[b]))))
        if partner is None:
            missing.append({"source_key": i, "reason": "target_intersection_not_visible"}); continue
        j, other = partner
        found.append({"source_key": i, "target_key": j, "source_xy": junction["xy"], "target_xy": other["xy"],
                      "displacement_pixels": np.subtract(other["xy"], junction["xy"]).tolist(),
                      "source_lines": junction["lines"], "target_lines": other["lines"]})
    return {"matches": found, "missing": missing}
