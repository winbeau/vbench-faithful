"""Training objectives for a continuous video model with explicit still anchors.

No inference-time flow rule, score offset, rescaling, or calibration fit. The
same learned video head produces sigmoid scores; anchors supervise its weights.
"""
from __future__ import annotations

import torch


VIEWS = ('base', 'alternating', 'aperiodic', 'still', 'pan8', 'pan32', 'reversal32', 'still_jitter8')


def anchored_losses(latent, pairs, labels, config):
    if latent.ndim != 2 or latent.shape[1] != len(VIEWS):
        raise ValueError('all eight views required')
    if not torch.all((labels == 0) | (labels == .5) | (labels == 1)):
        raise ValueError('invalid motion preference label')
    scores = torch.sigmoid(latent)
    differences = scores[pairs[:, 0], 0] - scores[pairs[:, 1], 0]
    ordered = labels != .5
    if not ordered.any() or ordered.all():
        raise ValueError('ordered and tied supervision required')
    rank = (differences[ordered] - config['natural_margin'] * (2 * labels[ordered] - 1)).square().mean()
    ties = differences[~ordered].square().mean()
    invariance = (scores[:, 1:3] - scores[:, :1]).square().mean()
    zero = scores[:, [3, 7]].square().mean()
    weak = torch.relu(config['weak_pan_margin'] - (scores[:, 4] - scores[:, 3])).square().mean()
    strong = torch.relu(config['strong_pan_margin'] - (scores[:, 5] - scores[:, 4])).square().mean()
    reversal = torch.relu(config['reversal_margin'] - (scores[:, 6] - scores[:, 3])).square().mean()
    # Native video may really be static: no positive motion label is invented.
    floor = torch.relu(scores[:, 3] - scores[:, 0]).square().mean()
    parts = {'natural_ordered': rank, 'human_ties': ties, 'native_jitter': invariance,
             'still_zero': zero, 'weak_pan': weak, 'strong_pan': strong,
             'real_reversal': reversal, 'native_not_below_still': floor}
    total = (.5 * rank + .5 * ties + config['invariance_weight'] * invariance
             + config['zero_weight'] * zero + config['pan_weight'] * (weak + strong + reversal) / 3
             + config['native_floor_weight'] * floor)
    return total, parts
