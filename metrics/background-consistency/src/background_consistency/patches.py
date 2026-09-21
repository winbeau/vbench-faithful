"""Development-only background pooling from the existing CLIP visual tokens."""
from __future__ import annotations

import numpy as np


def extract_clip_tokens(encoder, frames):
    """Run the pinned visual forward, retaining patch tokens before CLS selection.

    No weights or attention layers change. CLS follows the original operations;
    the same final LayerNorm/projection is additionally applied to patch tokens.
    Tokens remain contextualized: masking at pooling cannot guarantee isolation.
    """
    import torch
    from torch.nn import functional as F
    visual = encoder.model.visual
    images = encoder.transform(frames.float()).to(encoder.device, dtype=encoder.model.dtype)
    with torch.inference_mode():
        x = visual.conv1(images)
        grid = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = visual.ln_pre(torch.cat([cls, x], dim=1)+visual.positional_embedding.to(x.dtype))
        x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        cls, patches = visual.ln_post(x[:, 0, :]), visual.ln_post(x[:, 1:, :])
        if visual.proj is not None:
            cls, patches = cls @ visual.proj, patches @ visual.proj
        return F.normalize(cls, dim=-1), patches, grid


def background_patch_weights(encoder, foreground, grid, *, temporal_union=False):
    """Apply the exact image resize/crop geometry, then average mask coverage."""
    import torch
    from torch.nn import functional as F
    masks = np.asarray(foreground)
    if masks.ndim != 3 or not np.isin(masks, (0, 1)).all():
        raise ValueError('foreground masks must be binary [T,H,W]')
    if temporal_union:
        masks = np.broadcast_to(masks.any(0), masks.shape).copy()
    x = torch.from_numpy(masks.astype(np.float32))[:, None]
    # Pinned VBench clip_transform starts with Resize and CenterCrop. Do not
    # normalize masks as RGB or let a separate crop alter background geometry.
    for transform in encoder.transform.transforms[:2]:
        x = transform(x)
    weights = F.adaptive_avg_pool2d(1-x.clamp(0, 1), grid).flatten(1)
    return weights.to(encoder.device)


def pool_background_tokens(patches, weights, *, min_background_fraction=.05):
    """Soft area-weighted pooling; missing evidence retains the time denominator."""
    import torch
    from torch.nn import functional as F
    if patches.ndim != 3 or weights.shape != patches.shape[:2]:
        raise ValueError('patch features and weights must be [T,P,D] and [T,P]')
    if (not torch.isfinite(patches).all() or not torch.isfinite(weights).all()
            or torch.any(weights < 0) or torch.any(weights > 1)):
        raise ValueError('invalid patch features or background weights')
    fractions = weights.float().mean(-1)
    pooled = (patches.float()*weights.float()[..., None]).sum(1)
    norms = pooled.norm(dim=-1)
    valid = (fractions >= min_background_fraction) & (norms > 0)
    # A finite placeholder is required by cosine validation; it contributes
    # zero whenever either frame lacks background, via the valid-pair flags.
    pooled = torch.where(valid[:, None], pooled, torch.ones_like(pooled))
    features = F.normalize(pooled, dim=-1).to(patches.dtype)
    return features, fractions, valid


def patch_views(encoder, frames, foreground):
    import torch
    cls, patches, grid = extract_clip_tokens(encoder, frames)
    views, fractions, valid = {}, {}, {}
    for support in ('global', 'frame', 'union'):
        weights = (torch.ones(patches.shape[:2], device=patches.device) if support == 'global'
                   else background_patch_weights(encoder, foreground, grid, temporal_union=support == 'union'))
        views[support], fractions[support], valid[support] = pool_background_tokens(patches, weights)
    return cls, patches, views, fractions, valid
