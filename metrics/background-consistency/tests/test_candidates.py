import numpy as np
import pytest
import torch
from torch.nn import functional as F

from background_consistency.candidates import all_pairs_same_precision, box_masks


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_same_precision_matches_literal_scalar_pairs_and_fixed_denominator(dtype):
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(9, 512, generator=generator).to(dtype)
    for valid in ([True]*9, [True, False, True, True, False, True, True, False, True]):
        expected = sum(max(0., F.cosine_similarity(features[i:i+1], features[j:j+1]).item())
            for i in range(8) for j in range(i+1, 9) if valid[i] and valid[j]) / 36
        assert all_pairs_same_precision(features, valid=valid) == expected


def test_coarse_boxes_round_outward_and_keep_missing_frames_empty():
    masks = box_masks([{'boxes': [[1.2, 2.8, 4.3, 4.1]]}, {'boxes': []}], (2, 6, 7))
    assert masks[0].sum() == 12
    assert np.all(masks[0, 2:5, 1:5] == 1)
    assert masks[1].sum() == 0
    with pytest.raises(ValueError, match='invalid'):
        box_masks([{'boxes': [[0, 0, 8, 3]]}], (1, 6, 7))
