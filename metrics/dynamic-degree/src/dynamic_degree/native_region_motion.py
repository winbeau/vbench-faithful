"""Native-image checks of regional motion; not a calibrated Dynamic score.

RGB templates search the complete translation domain, independently of DINO.
The target SAM regions and sparse SIFT correspondences are separate diagnostics,
not labels, confidence certificates, or substitutions for missing motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import correlate

from .regional_motion import RegionMotionConfig, refine_translation


@dataclass(frozen=True)
class NativeMotionConfig:
    hypotheses: int = 3
    min_overlap: float = .8
    refinement_steps: int = 8
    convergence_pixels: float = .001
    sift_ratio: float = .75
    target_visibility: str = "source_only"

    def __post_init__(self):
        RegionMotionConfig(hypotheses=self.hypotheses, min_overlap=self.min_overlap,
                           refinement_steps=self.refinement_steps,
                           convergence=self.convergence_pixels)
        if not np.isfinite(self.sift_ratio) or not 0 < self.sift_ratio < 1:
            raise ValueError("SIFT ratio must lie strictly between zero and one")
        if self.target_visibility not in {"source_only", "associated_sam"}:
            raise ValueError("unknown target visibility model")


def rgb_float(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 2:
        raise ValueError("native H,W,3 uint8 RGB required")
    return frame.astype(np.float64) / 255.


def _mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != tuple(shape) or (mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all()):
        raise ValueError("aligned binary region required")
    return mask.astype(bool, copy=False)


def rgb_translation_surface(source, target, mask, target_mask=None):
    """Conditional RGB squared error for every integer translation.

    For d=(dx,dy), cost is mean |source(x)-target(x+d)|^2 over visible
    masked pixels/channels. FFT only accelerates these exact finite sums.
    Cropping the source support saves work, without shrinking the search.
    No zero-padding cost or circular wrap-around contributes to the loss.
    """
    a, b = rgb_float(source), rgb_float(target)
    if a.shape != b.shape:
        raise ValueError("aligned native frame geometry required")
    mask = _mask(mask, a.shape[:2])
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    m, s = mask[y0:y1, x0:x1].astype(float), a[y0:y1, x0:x1]
    visible = np.ones(a.shape[:2]) if target_mask is None else _mask(target_mask, a.shape[:2]).astype(float)
    corr = lambda x, y: correlate(x, y, mode="full", method="fft")
    mass = np.clip(corr(visible, m), 0., float(m.sum()))
    source_energy = corr(visible, (s * s).sum(-1) * m)
    target_energy = corr((b * b).sum(-1) * visible, m)
    cross = sum(corr(b[..., c] * visible, s[..., c] * m) for c in range(3))
    loss = np.maximum(0., source_energy + target_energy - 2 * cross) / (3 * np.maximum(mass, 1e-12))
    loss[mass < .5] = np.inf  # no pixel evidence; not a motion-magnitude floor
    dy, dx = np.mgrid[-m.shape[0] + 1:a.shape[0], -m.shape[1] + 1:a.shape[1]]
    return {"loss": loss, "overlap": mass / m.sum(), "dx": dx - x0, "dy": dy - y0}


def peaks(surface, config):
    available = np.isfinite(surface["loss"]) & (surface["overlap"] >= config.min_overlap - 1e-10)
    selected = []
    for _ in range(config.hypotheses):
        if not available.any():
            break
        y, x = np.unravel_index(np.argmin(np.where(available, surface["loss"], np.inf)), available.shape)
        dx, dy = int(surface["dx"][y, x]), int(surface["dy"][y, x])
        selected.append({"displacement_pixels": [dx, dy], "rgb_mse": float(surface["loss"][y, x]),
                         "overlap": float(surface["overlap"][y, x])})
        # Adjacent integer samples of one local mode are not separate hypotheses.
        available &= np.maximum(abs(surface["dx"] - dx), abs(surface["dy"] - dy)) > 1
    return selected


def mask_association(source_mask, target_masks, displacement):
    """Warp/overlap diagnostics; local mask indices are never persistent IDs."""
    source_mask = np.asarray(source_mask)
    target_masks = np.asarray(target_masks)
    if (source_mask.ndim != 2 or target_masks.ndim != 3
            or target_masks.shape[1:] != source_mask.shape):
        raise ValueError("aligned binary region sets required")
    source_mask = _mask(source_mask, source_mask.shape)
    _mask(target_masks, target_masks.shape)
    return _mask_association_validated(source_mask, target_masks.astype(np.float32, copy=False),
                                       target_masks.sum(axis=(1, 2)), displacement)


def _mask_association_validated(source_mask, target_masks, areas, displacement):
    """Reuse checked masks/areas within one frame pair; same overlap arithmetic."""
    d = np.asarray(displacement, float)
    if d.shape != (2,) or not np.isfinite(d).all():
        raise ValueError("finite native displacement required")
    h, w = source_mask.shape
    warped = cv2.warpAffine(source_mask.astype(np.float32), np.array([[1., 0., d[0]], [0., 1., d[1]]]),
                            (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    visible = float(warped.sum())
    if not len(target_masks) or visible == 0:
        return {"target_region": None, "iou": None, "source_visible_pixels": visible,
                "source_recall": None, "target_precision": None}
    intersections = np.einsum("mhw,hw->m", target_masks, warped, optimize=True)
    unions = areas + visible - intersections
    iou = np.divide(intersections, unions, out=np.zeros(len(areas)), where=unions > 0)
    best = int(np.argmax(iou))
    if areas[best] == 0 or intersections[best] <= 0:
        return {"target_region": None, "iou": None, "source_visible_pixels": visible,
                "source_recall": None, "target_precision": None}
    return {"target_region": best, "iou": float(iou[best]), "source_visible_pixels": visible,
            "source_recall": float(intersections[best] / visible),
            "target_precision": float(intersections[best] / areas[best])}


def sift_features(frame):
    rgb_float(frame)
    # OpenCV defaults, no region/foreground restriction or feature-count cap.
    keys, descriptors = cv2.SIFT_create().detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None)
    return {"xy": np.array([k.pt for k in keys], np.float32).reshape(-1, 2),
            "scale": np.array([k.size for k in keys], np.float32), "descriptors": descriptors}


def sift_correspondences(source, target, config=NativeMotionConfig()):
    a, b = source["descriptors"], target["descriptors"]
    if a is None or b is None or min(len(a), len(b)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, backward = matcher.knnMatch(a, b, k=2), matcher.knnMatch(b, a, k=2)
    reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < config.sift_ratio * n.distance}
    return [{"source_key": m.queryIdx, "target_key": m.trainIdx,
             "source_xy": source["xy"][m.queryIdx].tolist(), "target_xy": target["xy"][m.trainIdx].tolist(),
             "distance": float(m.distance), "ratio": float(m.distance / n.distance)}
            for m, n in forward if m.distance < config.sift_ratio * n.distance and reverse.get(m.trainIdx) == m.queryIdx]


def fit_native_regions(source, target, masks, target_masks, matches, config=NativeMotionConfig()):
    a, b = rgb_float(source), rgb_float(target)
    masks, target_masks = np.asarray(masks), np.asarray(target_masks)
    if a.shape != b.shape or masks.ndim != 3 or target_masks.ndim != 3:
        raise ValueError("aligned frame pair and two region sets required")
    masks = _mask(masks, (len(masks), *a.shape[:2]))
    target_masks = _mask(target_masks, (len(target_masks), *a.shape[:2]))
    # Profiling showed whole-stack binary validation/area summation inside every
    # hypothesis dominated runtime. Validate/cache once, never drop regions.
    association_masks = target_masks.astype(np.float32)
    association_areas = target_masks.sum(axis=(1, 2))
    associate = lambda mask, d: _mask_association_validated(mask, association_masks, association_areas, d)
    match_xy = np.array([m["source_xy"] for m in matches]).reshape(-1, 2)
    match_xy = np.rint(match_xy).astype(int)
    match_xy = np.clip(match_xy, [0, 0], [a.shape[1] - 1, a.shape[0] - 1])
    refine = RegionMotionConfig(hypotheses=config.hypotheses, min_overlap=config.min_overlap,
                               refinement_steps=config.refinement_steps, convergence=config.convergence_pixels)
    records = []
    for index, mask in enumerate(masks):
        surface = rgb_translation_surface(source, target, mask)
        hypotheses = []
        selected = peaks(surface, config) if surface is not None else []
        source_only_proposals = selected
        visibility = {None: None}
        if config.target_visibility == "associated_sam" and mask.any() and not mask.all():
            # Propose counterpart identities using source-only hypotheses AND
            # unshifted overlap. Zero is not selected as a motion prior: every
            # proposed mask gets a fresh complete-domain photometric search.
            ids = {associate(mask, d)["target_region"] for d in
                   ([p["displacement_pixels"] for p in selected] + [[0, 0]])}
            visibility = {i: target_masks[i] for i in sorted(i for i in ids if i is not None)}
            selected = [{**p, "alignment_target_region": i} for i, tm in visibility.items()
                        for p in peaks(rgb_translation_surface(source, target, mask, tm), config)]
        membership = mask[match_xy[:, 1], match_xy[:, 0]]
        local_matches = [m for m, keep in zip(matches, membership) if keep]
        vectors = np.array([np.subtract(m["target_xy"], m["source_xy"]) for m in local_matches]).reshape(-1, 2)
        for peak in selected:
            target_index = peak.get("alignment_target_region")
            result = refine_translation(a, b, mask.ravel(), peak["displacement_pixels"], refine,
                                        target_weights=visibility[target_index])
            d = result.pop("displacement_grid")
            result["displacement_pixels"] = d
            loss = result.pop("loss")
            result["rgb_mse"] = loss / 3 if loss is not None else None
            result["integer_peak"] = peak
            result["alignment_target_region"] = target_index
            result["mask_association"] = associate(mask, d) if d is not None else None
            result["sift_residual_pixels"] = np.linalg.norm(vectors - d, axis=1).tolist() if d is not None else []
            hypotheses.append(result)
        good = [i for i, x in enumerate(hypotheses) if x["rgb_mse"] is not None]
        best = min(good, key=lambda i: hypotheses[i]["rgb_mse"]) if good else None
        # A best template candidate is not a reliability certificate. Rank-zero
        # regions have no uniquely observable motion and must not become zeros.
        observable = best is not None and hypotheses[best]["rank"] == 2
        records.append({"region": index, "area_pixels": int(mask.sum()), "hypotheses": hypotheses,
                        "source_only_proposals": source_only_proposals,
                        "best_hypothesis": best, "observable_translation": observable,
                        "displacement_pixels": hypotheses[best]["displacement_pixels"] if observable else None,
                        "sift_match_source_keys": [m["source_key"] for m in local_matches],
                        "sift_displacements_pixels": vectors.tolist()})
    return records


def temporal_match_closure(pairs):
    """Record all available three-frame sparse identity checks, not a score.

    This tests feature-index correspondence, not constant speed or temporal
    smoothness. Real reversals/periodic motion are not rejected for reversing.
    Sparse agreement does not certify dense coverage or physical correctness.
    """
    maps = {(p["start"], p["lag"]): {m["source_key"]: m for m in p["sift_matches"]} for p in pairs}
    records = []
    for (start, total), direct in maps.items():
        for first in range(1, total):
            left, right = maps.get((start, first)), maps.get((start + first, total - first))
            if left is None or right is None:
                continue
            chains = [(key, a, right[a["target_key"]]) for key, a in left.items() if a["target_key"] in right]
            verified = [(key, a, b) for key, a, b in chains if key in direct and b["target_key"] == direct[key]["target_key"]]
            records.append({"start": start, "first_lag": first, "total_lag": total,
                            "chains": len(chains), "with_direct": sum(key in direct for key, _, _ in chains),
                            "consistent": len(verified), "source_keys": [key for key, _, _ in verified],
                            "two_step_path_pixels": [float(np.linalg.norm(np.subtract(a["target_xy"], a["source_xy"]))
                                + np.linalg.norm(np.subtract(b["target_xy"], b["source_xy"]))) for _, a, b in verified]})
    return records


def rank_with_common_sift(region, target_masks, matches):
    """Independent compatibility ranking, NOT a reliable motion certificate.

    The witnesses must land inside EVERY candidate target identity, so a
    candidate cannot improve its rank by masking out an inconvenient match.
    Same-location SIFT orientations share a single spatial vote. No velocity,
    reversal, known jitter amplitude, or fixed pixel-error gate is introduced.
    A single witness may order hypotheses but cannot certify a region/score.
    """
    target_masks=np.asarray(target_masks)
    if target_masks.ndim!=3:
        raise ValueError('aligned target mask stack required')
    target_masks=_mask(target_masks,target_masks.shape)
    h,w=target_masks.shape[1:]
    keys=set(region['sift_match_source_keys'])
    local=[m for m in matches if m['source_key'] in keys]
    if len(local)!=len(keys) or len({m['source_key'] for m in local})!=len(local):
        raise ValueError('every source witness must have exactly one correspondence')
    candidates=[(i,r) for i,r in enumerate(region['hypotheses']) if r['displacement_pixels'] is not None]
    ids={r.get('alignment_target_region') for _,r in candidates}
    if any(i is not None and not 0<=i<len(target_masks) for i in ids):
        raise ValueError('candidate target index outside supplied masks')
    common=np.ones((h,w),bool)
    for i in ids:
        if i is not None:common &= target_masks[i]
    groups={};used=[];excluded=[]
    for m in local:
        source,target=np.asarray(m['source_xy'],float),np.asarray(m['target_xy'],float)
        if source.shape!=(2,) or target.shape!=(2,) or not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError('finite paired source/target locations required')
        x,y=np.rint(target).astype(int)
        if 0<=x<w and 0<=y<h and common[y,x]:
            groups.setdefault(tuple(source),[]).append(target-source);used.append(m['source_key'])
        else:excluded.append(m['source_key'])
    stats=[]
    for i,r in candidates:
        residual=[float(np.median(np.linalg.norm(np.asarray(v)-r['displacement_pixels'],axis=1))) for v in groups.values()]
        stats.append({'hypothesis':i,'unique_location_residuals_pixels':residual,
                      'median_residual_pixels':float(np.median(residual)) if residual else None})
    order=sorted(stats,key=lambda x:x['median_residual_pixels']) if groups else []
    return {'status':'diagnostic_only','score':None,'reliability':'NOT CERTIFIED',
            'common_target_pixels':int(common.sum()),'common_source_keys':used,'excluded_source_keys':excluded,
            'unique_witness_locations':len(groups),'hypotheses':stats,
            'compatibility_order':[r['hypothesis'] for r in order],
            'best_compatible_hypothesis':order[0]['hypothesis'] if order else None}
