"""Regional CoTracker input conditioning, with explicit native-coordinate maps.

Every source SAM region is tested, not a manually chosen moving object. The
reference-frame window is fixed over time: it cannot stabilize away motion.
This is a correspondence diagnostic, NOT a Dynamic Degree scoring function.
"""
from __future__ import annotations

import cv2
import numpy as np


def transform_xy(points, matrix):
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if (points.shape[-1:] != (2,) or matrix.shape != (2, 3)
            or not np.isfinite(points).all() or not np.isfinite(matrix).all()):
        raise ValueError("finite XY coordinates and 2x3 transform required")
    return points @ matrix[:, :2].T + matrix[:, 2]


def region_window(mask, context=1.5, shape=(384, 512)):
    mask = np.asarray(mask)
    if (mask.ndim != 2 or mask.dtype != np.bool_ or not np.isfinite(context) or context < 1
            or len(shape) != 2 or min(shape) < 2):
        raise ValueError("binary source mask, context>=1 and nondegenerate output shape required")
    y, x = np.nonzero(mask)
    if len(x) < 3:
        return None
    xy = np.column_stack((x, y)).astype(float)
    center = xy.mean(axis=0)
    covariance = (xy - center).T @ (xy - center) / len(xy)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, -1]
    if major[0] < 0 or (major[0] == 0 and major[1] < 0):
        major = -major
    axes = np.array([major, [-major[1], major[0]]])
    local = (xy - center) @ axes.T
    low, high = local.min(axis=0), local.max(axis=0)
    extent = (high - low + 1) * context
    scale = (np.array(shape[::-1], float) - 1) / extent
    linear = scale[:, None] * axes
    local_center = (low + high) / 2
    offset = (np.array(shape[::-1], float) - 1) / 2 - linear @ center - scale * local_center
    matrix = np.c_[linear, offset]
    return {"native_to_window": matrix.tolist(), "window_to_native": cv2.invertAffineTransform(matrix).tolist(),
            "shape": list(shape), "context": context, "eigenvalues": eigenvalues.tolist(),
            "native_extent_in_principal_axes": extent.tolist(),
            "note": "PCA orientation/anisotropic scaling is input conditioning, not a motion estimate"}


def track_regions(model, frames, queries, masks, context=1.5):
    frames, queries, masks = np.asarray(frames), np.asarray(queries), np.asarray(masks)
    if (frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3
            or queries.ndim != 2 or queries.shape[1] != 3 or not np.isfinite(queries).all()
            or masks.ndim != 3 or masks.shape[1:] != frames.shape[1:3] or masks.dtype != np.bool_):
        raise ValueError("native video, txy queries and aligned binary region stack required")
    indices = np.rint(queries[:, 1:]).astype(int)
    if ((indices < 0).any() or (indices[:, 0] >= frames.shape[2]).any()
            or (indices[:, 1] >= frames.shape[1]).any()):
        raise ValueError("source query outside native image")
    records, arrays = [], {}
    for i, mask in enumerate(masks):
        selected = np.flatnonzero(mask[indices[:, 1], indices[:, 0]])
        window = region_window(mask, context)
        record = {"region": i, "area_pixels": int(mask.sum()), "query_indices": selected.tolist(),
                  "window": window, "status": "diagnostic_only", "score": None}
        records.append(record)
        if not len(selected) or window is None:
            record.update(status="insufficient", reason="no_query_or_degenerate_region")
            continue
        try:
            matrix = np.asarray(window["native_to_window"])
            h, w = window["shape"]
            conditioned = np.stack([cv2.warpAffine(f, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT) for f in frames])
            q = queries[selected].copy()
            q[:, 1:] = transform_xy(q[:, 1:], matrix)
            prediction = model.track_queries(conditioned, q)
            native = transform_xy(prediction["tracks"], window["window_to_native"])
            in_window = ((prediction["tracks"] >= 0).all(axis=-1)
                         & (prediction["tracks"][..., 0] <= w - 1) & (prediction["tracks"][..., 1] <= h - 1))
            in_native = ((native >= 0).all(axis=-1) & (native[..., 0] <= frames.shape[2] - 1)
                         & (native[..., 1] <= frames.shape[1] - 1))
            arrays.update({f"region_{i}_tracks_native": native, f"region_{i}_tracks_window": prediction["tracks"],
                           f"region_{i}_model_visible": prediction["visible"],
                           f"region_{i}_inside_observed_window": in_window & in_native,
                           f"region_{i}_query_indices": selected})
            record["model_visible_fraction"] = float(np.mean(prediction["visible"]))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return records, arrays
