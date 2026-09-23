"""Video-only alternative spatial witnesses; no motion/score acceptance gate.

RootSIFT reweights the SAME SIFT descriptors, not an independent detector.
AKAZE uses OpenCV's default detector/MLDB descriptor. Every detected location is
kept; this module neither sees a segmentation mask nor a construction label.
"""
from __future__ import annotations

import cv2
import numpy as np

from .native_region_motion import rgb_float

METHODS = ("sift", "rootsift", "akaze")


def root_descriptors(descriptors):
    if descriptors is None:
        return None
    d = np.asarray(descriptors, np.float32)
    if d.ndim != 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("finite nonnegative descriptor matrix required")
    norm = d.sum(axis=1, keepdims=True)
    return np.sqrt(np.divide(d, norm, out=np.zeros_like(d), where=norm > 0))


def extract_features(frame, method):
    rgb_float(frame)
    if method not in METHODS:
        raise ValueError("unknown structural witness method")
    detector = cv2.AKAZE_create() if method == "akaze" else cv2.SIFT_create()
    keys, descriptors = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None)
    if method == "rootsift":
        descriptors = root_descriptors(descriptors)
    return {"method": method, "xy": np.array([k.pt for k in keys], np.float32).reshape(-1, 2),
            "scale": np.array([k.size for k in keys], np.float32), "descriptors": descriptors}


def reciprocal_matches(source, target, ratio=.75):
    method = source["method"]
    if method not in METHODS or target["method"] != method:
        raise ValueError("matching descriptor identities required")
    if not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("ratio must lie strictly between zero and one")
    a, b = source["descriptors"], target["descriptors"]
    if a is None or b is None or min(len(a), len(b)) < 2:
        return []
    norm = cv2.NORM_HAMMING if method == "akaze" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    forward, backward = matcher.knnMatch(a, b, k=2), matcher.knnMatch(b, a, k=2)
    reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < ratio * n.distance}
    return [{"source_key": m.queryIdx, "target_key": m.trainIdx,
             "source_xy": source["xy"][m.queryIdx].tolist(),
             "target_xy": target["xy"][m.trainIdx].tolist(),
             "source_scale": float(source["scale"][m.queryIdx]),
             "target_scale": float(target["scale"][m.trainIdx]),
             "distance": float(m.distance), "ratio": float(m.distance / n.distance)}
            for m, n in forward if m.distance < ratio * n.distance
            and reverse.get(m.trainIdx) == m.queryIdx]


def source_keys_in_mask(matches, mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("binary native mask required")
    result = []
    h, w = mask.shape
    for match in matches:
        xy = np.asarray(match["source_xy"], float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("finite source position required")
        x, y = np.rint(xy).astype(int)
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            result.append(match["source_key"])
    return result


def distinct_queries(features, frame_index):
    """All unique detected positions, not only matched/foreground positions."""
    xy = np.asarray(features["xy"], np.float32)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
            or not isinstance(frame_index, int) or frame_index < 0):
        raise ValueError("finite feature locations and nonnegative frame index required")
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    groups = [[] for _ in unique]
    for key, index in enumerate(inverse):
        groups[index].append(key)
    queries = np.column_stack((np.full(len(unique), frame_index, np.float32), unique))
    return queries, groups


def identity_triangles(pairs):
    """Cross-check feature identities through every available third frame.

    This checks composition of correspondence indices, not velocity smoothness,
    net displacement, even/odd phase, or a pixel-distance tolerance. A genuine
    returning/periodic point can pass. Unknown support remains unknown, and
    even a consistent identity is NOT certified physical motion.
    """
    edges = {}
    frames = set()
    for pair in pairs:
        start, end = pair["start"], pair["start"] + pair["lag"]
        if start < 0 or pair["lag"] <= 0 or (start, end) in edges:
            raise ValueError("unique ordered frame pairs required")
        direct = {m["source_key"]: m["target_key"] for m in pair["matches"]}
        reverse = {v: k for k, v in direct.items()}
        if len(direct) != len(pair["matches"]) or len(reverse) != len(direct):
            raise ValueError("one-to-one paired feature indices required")
        edges[start, end], edges[end, start] = direct, reverse
        frames.update((start, end))
    results = []
    for pair in pairs:
        start, end = pair["start"], pair["start"] + pair["lag"]
        support = []
        for match in pair["matches"]:
            confirmed, conflicting = [], []
            for third in sorted(frames - {start, end}):
                left, right = edges.get((start, third), {}), edges.get((end, third), {})
                a, b = match["source_key"], match["target_key"]
                if a in left and b in right:
                    (confirmed if left[a] == right[b] else conflicting).append(third)
            support.append({"source_key": match["source_key"], "target_key": match["target_key"],
                            "confirmed_frames": confirmed, "conflicting_frames": conflicting,
                            "supported_without_conflict": bool(confirmed) and not conflicting})
        results.append({"start": start, "lag": pair["lag"], "matches": support})
    return results
