"""Recorded development candidates; not a validated default repair."""
from __future__ import annotations

import numpy as np

from .algorithms import suppress_foreground, temporal_score, validate_features


def all_pairs_same_precision(features, *, valid=None):
    """Mean every unordered pair with upstream's dtype, cosine and lower clamp.

    Pairwise scalar operations deliberately match the official path. In
    particular there is no float32 re-normalization or upper clamp at one.
    The number of usable observations never changes the pair denominator.
    """
    from torch.nn import functional as F
    validate_features(features)
    count = len(features)
    if valid is None:
        valid = [True] * count
    if len(valid) != count:
        raise ValueError('valid frame count mismatch')
    total = 0.0
    for i in range(count-1):
        # Each row has the same reduction dimension and dtype as upstream.
        # Avoid one host synchronization per pair without changing cosine.
        values = F.cosine_similarity(features[i:i+1], features[i+1:]).tolist()
        total += sum(max(0.0, value) for j, value in enumerate(values, i+1)
                     if valid[i] and valid[j])
    return total / (count * (count-1) / 2)


def box_masks(detections, shape):
    """Union of coarse detector rectangles, independently on every frame."""
    count, height, width = shape
    if len(detections) != count:
        raise ValueError('detector frame count mismatch')
    masks = np.zeros(shape, dtype=np.uint8)
    for i, row in enumerate(detections):
        for box in row['boxes']:
            values = np.asarray(box, dtype=float)
            if (values.shape != (4,) or not np.isfinite(values).all()
                    or (values < 0).any() or values[2] > width or values[3] > height
                    or values[0] >= values[2] or values[1] >= values[3]):
                raise ValueError('invalid detector rectangle')
            x0, y0 = np.floor(values[:2]).astype(int)
            x1, y1 = np.ceil(values[2:]).astype(int)
            masks[i, y0:y1, x0:x1] = 1
    return masks


def encode_candidate_views(encoder, frames, masks_by_kind):
    import torch
    if not torch.equal(frames, frames.to(torch.uint8).to(frames.dtype)):
        raise ValueError('candidate views require exact native RGB bytes')
    images = frames.to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    views, fractions = {}, {}
    for kind, masks in masks_by_kind.items():
        for support in ('frame', 'union'):
            key = kind+'_'+support
            edited, fractions[key] = suppress_foreground(images, masks, temporal_union=support == 'union')
            views[key] = encoder.features(torch.from_numpy(edited).permute(0, 3, 1, 2))
    return views, fractions


def candidate_scores(views, fractions, *, min_background_fraction=.05):
    result = {}
    for key, features in views.items():
        valid = np.asarray(fractions[key]) >= min_background_fraction
        result[key+'_official'] = temporal_score(features, valid=valid)
        result[key+'_all_pairs_fp16'] = all_pairs_same_precision(features, valid=valid)
    return result
