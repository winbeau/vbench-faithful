from pathlib import Path

import numpy as np
import pytest
import torch

from subject_consistency.automatic_localizer import AutomaticMobileSamSubjectMaskProvider


class Detector:
    provenance = {'family': 'test_detector'}

    def boxes_for(self, frames, phrase):
        return [{'boxes': [[0, 0, 2, 2], [3, 3, 5, 5]], 'scores': [.9, .95]},
                {'boxes': [], 'scores': []},
                {'boxes': [[1, 2, 4, 5]], 'scores': [.99]}]


class Predictor:
    def __init__(self):
        self.calls = []

    def set_image(self, image, image_format):
        assert image_format == 'RGB'
        self.shape = image.shape[:2]
        self.calls.append(int(image[0, 0, 0]))

    def predict(self, box, multimask_output):
        assert multimask_output is False
        x0, y0, x1, y1 = map(int, box)
        mask = np.zeros((1, *self.shape), dtype=bool)
        mask[:, y0:y1, x0:x1] = True
        return mask, np.asarray([.9]), None


def test_per_frame_motion_union_and_misses_are_not_hidden():
    sam = Predictor()
    provider = AutomaticMobileSamSubjectMaskProvider(Detector(), sam, weights_sha256='test')
    frames = torch.arange(3, dtype=torch.uint8).reshape(3, 1, 1, 1).expand(3, 3, 6, 6)
    result = provider.masks_for(Path('video'), frames, 'person')
    assert result.instance_present.tolist() == [[True], [False], [True]]
    assert result.instance_masks.sum((1, 2, 3)).tolist() == [8, 0, 9]
    assert sam.calls == [0, 2]
    assert result.instance_masks[2, 0, 4, 3] == 1
    assert provider.last_diagnostics['num_multi_instance_frames'] == 1
    assert provider.provenance['human_confirmed'] is False
    assert provider.provenance['construction_masks_reused'] is False


def test_reject_invalid_boxes_and_unsupported_subjects():
    provider = AutomaticMobileSamSubjectMaskProvider(Detector(), Predictor(), weights_sha256='test')
    frames = torch.zeros((3, 3, 2, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match='invalid detector box'):
        provider.masks_for(Path('video'), frames, 'person')
    with pytest.raises(ValueError, match='unsupported official subject'):
        provider.masks_for(Path('video'), frames, 'made_up_category')
