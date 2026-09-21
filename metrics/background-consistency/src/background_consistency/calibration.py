"""Development-frozen score scale for the background patch representation."""
from __future__ import annotations

import math


PATCH_GAIN = 2.0
PATCH_OFFICIAL_WEIGHT = 0.5
REPAIR_GAIN = 1.75


def calibrate_patch_score(score, *, gain=PATCH_GAIN):
    """Expand deficits from one, preserving a bounded consistency score.

    Gain is fitted/frozen on development background responses, never test
    labels. It expands foreground variation too; it cannot improve the
    nuisance/sensitivity ratio when no clipping occurs. Raw scores are kept.
    """
    if not math.isfinite(score) or not math.isfinite(gain) or gain < 1:
        raise ValueError('calibration needs a finite score and gain >= 1')
    return min(1.0, max(0.0, 1-gain*(1-score)))


def patch_score_grid(cls, views, valid):
    """Origin, aggregation controls, raw patch ablations and calibrated scores."""
    from .algorithms import temporal_score
    from .candidates import all_pairs_same_precision
    scores = {'official': temporal_score(cls),
              'aggregation': temporal_score(cls, aggregation='all_pairs'),
              'aggregation_fp16': all_pairs_same_precision(cls)}
    for support, features in views.items():
        usable = valid[support].tolist()
        raw = {'official': temporal_score(features, valid=usable),
               'all_pairs': all_pairs_same_precision(features, valid=usable)}
        for aggregation, value in raw.items():
            name = f'patch_{support}_{aggregation}'
            scores[name] = value
            scores[name+'_gain2'] = calibrate_patch_score(value)
        scores[f'patch_{support}_balanced_gain2'] = balanced_patch_score(raw['official'], raw['all_pairs'])
    scores['patch_frame_all_pairs_calibrated'] = calibrate_patch_score(scores['patch_frame_all_pairs'], gain=REPAIR_GAIN)
    return scores


def balanced_patch_score(official, all_pairs):
    """One development-declared mixture; coefficients are not test-fitted."""
    return calibrate_patch_score(PATCH_OFFICIAL_WEIGHT*official+(1-PATCH_OFFICIAL_WEIGHT)*all_pairs)
