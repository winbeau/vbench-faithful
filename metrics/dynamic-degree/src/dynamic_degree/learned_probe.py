"""Experimental continuous readout of frozen video tokens; not a public default.

The sigmoid of the learned latent is only a relative display score. Pairwise
motion preference does not identify a calibrated absolute motion intensity.
"""
from __future__ import annotations

import torch
from torch import nn


class MotionProbe(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, mlp_dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.score = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.Tanh(), nn.Linear(mlp_dim, 1))

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.project.in_features:
            raise ValueError('expected batch x all spatiotemporal tokens x channels')
        hidden = torch.tanh(self.project(self.norm(tokens)))
        weights = torch.softmax(self.attention(hidden), dim=1)
        return self.score((weights * hidden).sum(dim=1)).squeeze(-1)


def motion_losses(latent, pairs, labels, consistency_weight, gauge_weight):
    """latent: source x [base,jitter1,jitter2]; labels 1=A, 0=B, .5=tie."""
    if latent.ndim != 2 or latent.shape[1] not in (1, 3):
        raise ValueError('one natural view or three views required')
    if not torch.all((labels == 0) | (labels == 0.5) | (labels == 1)):
        raise ValueError('invalid preference label')
    if consistency_weight and latent.shape[1] != 3:
        raise ValueError('consistency requires both counterfactual views')
    differences = latent[pairs[:, 0], 0] - latent[pairs[:, 1], 0]
    ordered = labels != 0.5
    if not ordered.any() or ordered.all():
        raise ValueError('both ordered and tied supervision required')
    rank_loss = (differences[ordered] - (2 * labels[ordered] - 1)).square().mean()
    tie_loss = differences[~ordered].square().mean()
    consistency = (latent[:, 1:] - latent[:, :1]).square().mean() if latent.shape[1] == 3 else latent.new_zeros(())
    gauge = latent[:, 0].mean().square()
    total = 0.5 * rank_loss + 0.5 * tie_loss + consistency_weight * consistency + gauge_weight * gauge
    return total, {'ordered': rank_loss, 'ties': tie_loss, 'consistency': consistency, 'gauge': gauge}
