"""Background representations and aggregation, independent of model loading.

Only independent scoring masks may enter suppress_foreground. Construction
masks belong to the experiment generator, never this metric.
"""
from __future__ import annotations

import numpy as np


def suppress_foreground(frames, foreground, *, temporal_union=False, fill_rgb=(128, 128, 128)):
    images, masks = np.asarray(frames), np.asarray(foreground)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("frames must be uint8 [T,H,W,3]")
    if masks.shape != images.shape[:3] or not np.isin(masks, (0, 1)).all():
        raise ValueError("foreground must be binary [T,H,W] at native resolution")
    fill = np.asarray(fill_rgb)
    if fill.shape != (3,) or not np.isfinite(fill).all() or ((fill < 0) | (fill > 255)).any():
        raise ValueError("fill_rgb must be three values in [0,255]")
    masks = masks.astype(bool)
    if temporal_union:
        masks = np.broadcast_to(masks.any(axis=0), masks.shape)
    result = images.copy()
    result[masks] = fill.astype(np.uint8)
    return result, 1 - masks.mean(axis=(1, 2))


def validate_features(features):
    import torch
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("features require [T,D] with at least two frames")
    if not bool(torch.isfinite(features).all()) or bool((features.float().norm(dim=-1) == 0).any()):
        raise ValueError("features must be finite nonzero vectors")


def temporal_score(features, *, aggregation="official", valid=None):
    """Official scalar cosine loop, or mean over all unordered pairs.

    Preserve input dtype for the official path (upstream CUDA CLIP is fp16).
    Missing background contributes zero with the original fixed denominator.
    No detected foreground is valid whole-background evidence, not missing
    background; its coverage is reported separately by the localizer.
    """
    import torch
    from torch.nn import functional as F
    validate_features(features)
    count = len(features)
    if valid is None:
        valid = [True] * count
    if len(valid) != count:
        raise ValueError("valid frame count mismatch")
    valid = [bool(v) for v in valid]
    if aggregation == "official":
        total = 0.0
        for i in range(1, count):
            previous = max(0.0, F.cosine_similarity(features[i-1:i], features[i:i+1]).item())
            first = max(0.0, F.cosine_similarity(features[:1], features[i:i+1]).item())
            total += (previous * (valid[i-1] and valid[i]) + first * (valid[0] and valid[i])) / 2
        return total / (count - 1)
    if aggregation != "all_pairs":
        raise ValueError(f"unknown aggregation: {aggregation}")
    normalized = F.normalize(features.float(), dim=-1)
    similarities = (normalized @ normalized.T).clamp(0, 1)
    usable = torch.as_tensor(valid, device=features.device)
    pairs = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
    similarities = similarities * (usable[:, None] & usable[None, :])
    return float(similarities[pairs].double().mean().item())


METHODS = ("official", "aggregation", "frame_official", "frame_all_pairs", "union_official", "union_all_pairs")


def score_views(views, fractions, *, min_background_fraction=0.05):
    """Produce comparable ablations without selecting a winning method."""
    if not 0 < min_background_fraction <= 1:
        raise ValueError("min_background_fraction must be in (0,1]")
    result = {"official": temporal_score(views["global"]),
              "aggregation": temporal_score(views["global"], aggregation="all_pairs")}
    for view in ("frame", "union"):
        valid = np.asarray(fractions[view]) >= min_background_fraction
        for aggregation in ("official", "all_pairs"):
            result[f"{view}_{aggregation}"] = temporal_score(views[view], aggregation=aggregation, valid=valid)
    return result
