"""Multiple local appearance paths; no single-motion region assumption or score.

Every adjacent hypothesis stays eligible. Joining to a re-detected point is a
geometric proposal, not an identity certificate. Direct reference-template
checks inspect a new time relation without using net displacement as motion.
"""
from __future__ import annotations

import cv2
import numpy as np

from .native_region_motion import rgb_float, _mask
from .local_appearance import ncc_surface, distinct_correlation_peaks


def triplet_ncc(frames, support, first_offsets, total_offsets, *, min_overlap=.8, batch=32):
    """Compare the SAME source-index pixels at three integer placements.

    All three correlations use the intersection visible in all three frames.
    No re-detection, intermediate-template replacement, or velocity prior. This
    checks appearance consistency, not physical identity or nuisance type.
    """
    images = np.asarray([rgb_float(f) for f in frames])
    first, total = np.asarray(first_offsets), np.asarray(total_offsets)
    if (images.ndim != 4 or len(images) != 3 or first.ndim != 2 or first.shape[1] != 2
            or first.shape != total.shape or not np.isfinite(first).all() or not np.isfinite(total).all()
            or not np.array_equal(first, np.rint(first)) or not np.array_equal(total, np.rint(total))
            or not 0 < min_overlap <= 1 or not isinstance(batch, int) or batch < 1):
        raise ValueError("three aligned frames, integer path offsets and valid visibility/batch required")
    support = _mask(support, images.shape[1:3])
    yy, xx = np.nonzero(support)
    correlations = np.full((len(first), 3), np.nan)
    overlaps = np.zeros(len(first))
    if len(xx) < 3:
        return correlations, overlaps
    first, total = first.astype(int), total.astype(int)
    h, w = support.shape
    for start in range(0, len(first), batch):
        offsets = np.stack((np.zeros_like(first[start:start + batch]), first[start:start + batch], total[start:start + batch]), axis=1)
        count = len(offsets)
        mass = np.zeros(count)
        sums = np.zeros((count, 3, 3)); energies = np.zeros((count, 3)); products = np.zeros((count, 3))
        for pixel in range(0, len(xx), 8192):
            x = xx[None, None, pixel:pixel + 8192] + offsets[:, :, 0, None]
            y = yy[None, None, pixel:pixel + 8192] + offsets[:, :, 1, None]
            visible = ((x >= 0) & (x < w) & (y >= 0) & (y < h)).all(axis=1)
            values = images[np.arange(3)[None, :, None], np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
            values = values * visible[:, None, :, None]
            mass += visible.sum(axis=1)
            sums += values.sum(axis=2)
            energies += (values ** 2).sum(axis=(2, 3))
            for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                products[:, pair] += (values[:, a] * values[:, b]).sum(axis=(1, 2))
        safe = np.maximum(mass, 1.)
        variance = energies - (sums ** 2).sum(axis=2) / safe[:, None]
        floor = 128 * np.finfo(float).eps * np.maximum(1., energies)
        fraction = mass / len(xx)
        for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            valid = (mass >= 3) & (fraction >= min_overlap) & (variance[:, a] > floor[:, a]) & (variance[:, b] > floor[:, b])
            cov = products[:, pair] - (sums[:, a] * sums[:, b]).sum(axis=1) / safe
            values = np.full(count, np.nan)
            values[valid] = np.clip(cov[valid] / np.sqrt(variance[valid, a] * variance[valid, b]), -1., 1.)
            correlations[start:start + count, pair] = values
        overlaps[start:start + count] = fraction
    return correlations, overlaps


def reference_template_paths(frames, support, first_hypotheses, *, hypotheses=12, min_overlap=.8):
    """All adjacent x lag-two proposals using an unchanged source template.

    The first set comes from the bound adjacent global search. The second is
    searched globally against frame two, so absent middle SIFT detections do
    not delete a proposal. Both steps survive even for a returning trajectory.
    """
    reference = distinct_correlation_peaks(ncc_surface(frames[0], frames[2], support), hypotheses, min_overlap)
    first_rank, reference_rank = np.indices((len(first_hypotheses), len(reference))).reshape(2, -1)
    d1 = np.asarray([h["displacement_pixels"] for h in first_hypotheses], int).reshape(-1, 2)[first_rank]
    d2 = np.asarray([h["displacement_pixels"] for h in reference], int).reshape(-1, 2)[reference_rank]
    ncc, overlap = triplet_ncc(frames, support, d1, d2, min_overlap=min_overlap)
    arrays = {"first_rank": first_rank, "reference_rank": reference_rank, "first_offset": d1, "total_offset": d2,
              "common_first_ncc": ncc[:, 0], "common_reference_ncc": ncc[:, 1], "common_second_ncc": ncc[:, 2],
              "common_overlap": overlap, "bottleneck_ncc": np.min(ncc, axis=1)}
    observed = np.flatnonzero(np.isfinite(arrays["bottleneck_ncc"]))
    ranked = observed[np.argsort(-arrays["bottleneck_ncc"][observed], kind="stable")[:3]]
    result = {"reference_hypotheses": reference, "candidate_paths": len(first_rank), "observable_paths": len(observed),
              "ranked_path_indices": ranked.tolist(), "score": None}
    return result, arrays


def direct_ncc(source, target, support, offsets, *, min_overlap=.8, batch=64):
    """Evaluate declared translations, not a new global argmax search.

    RGB means are removed per channel on the common visible support; missing
    or flat support remains NaN. Offsets may be fractional, using OpenCV's
    documented quantized bilinear interpolator. No temporal filtering occurs.
    """
    a, b = rgb_float(source), rgb_float(target)
    support = _mask(support, a.shape[:2])
    offsets = np.asarray(offsets, float)
    if (a.shape != b.shape or offsets.ndim != 2 or offsets.shape[1] != 2
            or not np.isfinite(offsets).all() or not 0 < min_overlap <= 1
            or not isinstance(batch, int) or batch < 1):
        raise ValueError("aligned images, finite offsets and valid visibility/batch required")
    yy, xx = np.nonzero(support)
    correlation = np.full(len(offsets), np.nan)
    overlap = np.zeros(len(offsets))
    if len(xx) < 3:
        return correlation, overlap
    source_values = a[yy, xx]
    for start in range(0, len(offsets), batch):
        d = offsets[start:start + batch]
        mass = np.zeros(len(d)); ea = np.zeros(len(d)); eb = np.zeros(len(d)); product = np.zeros(len(d))
        sa = np.zeros((len(d), 3)); sb = np.zeros_like(sa)
        # Chunk pixels too, preserving ALL support and OpenCV's index limit.
        for pixel in range(0, len(xx), 8192):
            sl = slice(pixel, pixel + 8192)
            x, y = xx[None, sl] + d[:, 0, None], yy[None, sl] + d[:, 1, None]
            visible = (x >= 0) & (x <= b.shape[1] - 1) & (y >= 0) & (y <= b.shape[0] - 1)
            moved = cv2.remap(b, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
            original = source_values[None, sl]
            mass += visible.sum(axis=1)
            sa += np.sum(original * visible[..., None], axis=1)
            sb += np.sum(moved * visible[..., None], axis=1)
            ea += np.sum(original ** 2 * visible[..., None], axis=(1, 2))
            eb += np.sum(moved ** 2 * visible[..., None], axis=(1, 2))
            product += np.sum(original * moved * visible[..., None], axis=(1, 2))
        safe = np.maximum(mass, 1.)
        cov = product - np.sum(sa * sb, axis=1) / safe
        va, vb = ea - np.sum(sa ** 2, axis=1) / safe, eb - np.sum(sb ** 2, axis=1) / safe
        fraction = mass / len(xx)
        floor = 128 * np.finfo(float).eps * np.maximum(1., np.maximum(ea, eb))
        valid = (mass >= 3) & (fraction >= min_overlap) & (va > floor) & (vb > floor)
        values = np.full(len(d), np.nan)
        values[valid] = np.clip(cov[valid] / np.sqrt(va[valid] * vb[valid]), -1., 1.)
        correlation[start:start + len(d)] = values
        overlap[start:start + len(d)] = fraction
    return correlation, overlap


def two_hop_candidates(first, second, *, association_scale=.5):
    """Keep all two-hop paths via geometrically compatible automatic queries.

    Association radius = max(1px quantization, scale*sqrt(size_i*size_j)). No
    construction displacement or frequency enters. The intermediate node is a
    re-detection; record its error and re-evaluate the snapped first offset.
    All masks/hypotheses are retained, including conflicting and static paths.
    """
    if not np.isfinite(association_scale) or association_scale <= 0:
        raise ValueError("positive feature-relative association scale required")
    middle = second["source_points"]
    xy = np.array([p["xy"] for p in middle], float).reshape(-1, 2)
    sizes = np.array([p["size"] for p in middle])
    first_keys, second_keys = sorted(first["supports"]), sorted(second["supports"])
    first_index, second_index = {k: i for i, k in enumerate(first_keys)}, {k: i for i, k in enumerate(second_keys)}
    paths, counts = [], []
    for source_index, point in enumerate(first["source_points"]):
        position = np.asarray(point["xy"], float)
        radii = np.maximum(1., association_scale * np.sqrt(float(point["size"]) * sizes))
        linked, declared = 0, 0
        for source_key in dict.fromkeys(ref["support_sha256"] for ref in point["regional_supports"]):
            for rank1, hypothesis in enumerate(first["supports"][source_key]["hypotheses"]):
                declared += 1
                endpoint = position + hypothesis["displacement_pixels"]
                distances = np.linalg.norm(xy - endpoint, axis=1)
                targets = np.flatnonzero(distances <= radii)
                if len(targets):
                    linked += 1
                for target_index in targets:
                    next_point = middle[target_index]
                    snapped = xy[target_index] - position
                    for target_key in dict.fromkeys(ref["support_sha256"] for ref in next_point["regional_supports"]):
                        for rank2, next_hypothesis in enumerate(second["supports"][target_key]["hypotheses"]):
                            total = snapped + next_hypothesis["displacement_pixels"]
                            paths.append((source_index, first_index[source_key], rank1, int(target_index),
                                          second_index[target_key], rank2, *snapped, *total,
                                          distances[target_index], radii[target_index],
                                          hypothesis["correlation"], next_hypothesis["correlation"]))
        counts.append({"source_point": source_index, "declared_first_hypotheses": declared,
                       "with_middle_detection": linked, "without_middle_detection": declared - linked})
    matrix = np.asarray(paths, float).reshape(-1, 14)
    arrays = {"source_point": matrix[:, 0].astype(int), "source_support": matrix[:, 1].astype(int),
              "first_rank": matrix[:, 2].astype(int), "middle_point": matrix[:, 3].astype(int),
              "middle_support": matrix[:, 4].astype(int), "second_rank": matrix[:, 5].astype(int),
              "first_offset": matrix[:, 6:8], "total_offset": matrix[:, 8:10],
              "association_error": matrix[:, 10], "association_radius": matrix[:, 11],
              "cached_first_ncc": matrix[:, 12], "cached_second_ncc": matrix[:, 13],
              "source_support_keys": np.asarray(first_keys, dtype="U64"),
              "middle_support_keys": np.asarray(second_keys, dtype="U64")}
    return arrays, counts


def rank_reference_paths(arrays, *, hypotheses=3):
    """One stable ranking per source query AND support; all arrays stay saved.

    Bottleneck of the two adjacent photometric checks and direct reference
    check. This is NOT calibrated confidence or a motion score. Equal endpoint
    paths are not merged: their intermediate movements can differ materially.
    """
    first, second, reference = (arrays[k] for k in ("actual_first_ncc", "cached_second_ncc", "reference_ncc"))
    merit = np.minimum(np.minimum(first, second), reference)
    groups = sorted(set(zip(arrays["source_point"], arrays["source_support"])))
    result = []
    for point, support in groups:
        all_ids = np.flatnonzero((arrays["source_point"] == point) & (arrays["source_support"] == support))
        observed = all_ids[np.isfinite(merit[all_ids])]
        ranked = observed[np.argsort(-merit[observed], kind="stable")[:hypotheses]]
        result.append({"source_point": int(point), "source_support": int(support), "candidate_paths": len(all_ids),
                       "reference_observable_paths": len(observed), "ranked_path_indices": ranked.tolist()})
    return result, merit
