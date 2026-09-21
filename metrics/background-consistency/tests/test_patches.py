import pytest
import torch

from background_consistency.algorithms import temporal_score
from background_consistency.patches import pool_background_tokens


def test_excluding_foreground_tokens_removes_their_changes_without_a_gray_fill():
    # Fixed background token, strongly changing foreground token.
    patches = torch.tensor([[[1., 0.], [0., 5.]], [[1., 0.], [0., -5.]]])
    global_features, _, _ = pool_background_tokens(patches, torch.ones(2, 2))
    background, fractions, valid = pool_background_tokens(patches, torch.tensor([[1., 0.], [1., 0.]]))
    assert temporal_score(global_features) < .1
    assert temporal_score(background, valid=valid) == pytest.approx(1)
    assert fractions.tolist() == [.5, .5]


def test_missing_background_keeps_fixed_temporal_denominator():
    patches = torch.ones(3, 2, 4)
    features, _, valid = pool_background_tokens(patches, torch.tensor([[1., 1.], [0., 0.], [1., 1.]]))
    assert valid.tolist() == [True, False, True]
    assert torch.isfinite(features).all()
    assert temporal_score(features, valid=valid) == pytest.approx(.25)


def test_fractional_weights_and_all_background_match_weighted_direction():
    patches = torch.tensor([[[2., 0.], [0., 2.]], [[3., 0.], [0., 3.]]])
    pooled, fractions, valid = pool_background_tokens(patches, torch.tensor([[1., .5], [1., .5]]))
    assert torch.allclose(pooled[0], torch.tensor([2., 1.])/5**.5)
    assert fractions.tolist() == [.75, .75]
    assert valid.all()
