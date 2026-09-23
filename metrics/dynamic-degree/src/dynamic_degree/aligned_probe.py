"""Model-training scale supervision; no inference-time score transformation."""
from __future__ import annotations

import torch

from .anchored_probe import anchored_losses


def aligned_losses(latent, pairs, labels, parent_loss, origin_train_mean, weight=1.):
    target = torch.as_tensor(origin_train_mean, dtype=latent.dtype, device=latent.device)
    if target.ndim != 0 or not torch.isfinite(target) or target < 0 or target > 1 or target.requires_grad:
        raise ValueError('one fixed finite TRAIN population mean in [0,1] required')
    if not 0 <= weight:
        raise ValueError('nonnegative mean-supervision weight required')
    total, parts = anchored_losses(latent, pairs, labels, parent_loss)
    mean_loss = (torch.sigmoid(latent[:, 0]).mean() - target).square()
    return total + weight * mean_loss, dict(parts, origin_train_mean=mean_loss)
