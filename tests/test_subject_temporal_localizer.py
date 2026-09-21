from pathlib import Path

import numpy as np
import pytest
import torch

from subject_consistency.temporal_localizer import temporal_box_prompts, clip_anchor_prompts, direct_anchor_prompts, TemporalMobileSamSubjectMaskProvider


def detection(*boxes):
    return {'boxes': list(boxes), 'scores': [.9] * len(boxes)}


def test_bounded_gap_interpolates_prompts_without_extending_clip_boundaries():
    rows = [detection(), detection([1, 1, 5, 5]), detection(), detection([3, 1, 7, 5]), detection()]
    out = temporal_box_prompts(rows, smooth_radius=0)
    assert out[0] == out[4] == []
    assert out[2][0]['box'] == [2., 1., 6., 5.]
    assert out[2][0]['source'] == 'bounded_gap'


def test_separate_objects_are_both_kept_and_long_gaps_not_filled():
    rows = [detection([1, 1, 5, 5], [12, 1, 17, 6]), detection(), detection(), detection(),
            detection([1, 1, 5, 5], [12, 1, 17, 6])]
    out = temporal_box_prompts(rows)
    assert len(out[0]) == len(out[-1]) == 2
    assert out[1:4] == [[], [], []]


def test_detection_order_does_not_swap_tracks():
    out = temporal_box_prompts([detection([1, 1, 5, 5], [12, 1, 17, 6]),
                               detection([12, 1, 17, 6], [1, 1, 5, 5])])
    assert out[0] == out[1]


def test_empty_and_invalid_inputs():
    assert temporal_box_prompts([detection(), detection()]) == [[], []]
    with pytest.raises(ValueError):
        temporal_box_prompts([detection([1, 1, float('nan'), 4])])


def test_anchor_uses_highest_confidence_actual_frame_and_retains_its_instances():
    rows = [detection([1,1,3,3]), {'boxes':[[2,2,6,6],[7,2,10,6]],'scores':[.95,.6]}, detection()]
    out = clip_anchor_prompts(rows)
    assert all(len(x) == 2 and x[0]['anchor_frame'] == 1 for x in out)
    assert out[2][0]['box'] == [2,2,6,6]
    assert clip_anchor_prompts([detection(),detection()]) == [[],[]]
    out[0][0]['box'][0] = 100
    assert out[1][0]['box'][0] == 2


def test_anchor_rejects_confidences_without_matching_boxes():
    with pytest.raises(ValueError):
        clip_anchor_prompts([{'boxes':[],'scores':[.9]}])


def test_hybrid_keeps_moving_observations_and_fills_only_empty_detection_frames():
    rows = [detection([1,1,3,3]), detection(), detection([5,1,7,3])]
    out = direct_anchor_prompts(rows)
    assert out[0][0]['box'] == [1,1,3,3]
    assert out[2][0]['box'] == [5,1,7,3]
    assert out[1][0]['source'] == 'clip_anchor'
    assert out[2][0]['source'] == 'detected'
    assert direct_anchor_prompts([detection()]) == [[]]


@pytest.mark.parametrize('prompt_policy', ['bounded_tracks','clip_anchor','direct_anchor'])
def test_gap_uses_current_image_and_provider_resets_between_clips(prompt_policy):
    class Detector:
        provenance = {'test': True}
        rows = [detection([1, 1, 6, 6]), detection(), detection([1, 1, 6, 6])]
        def boxes_for(self, frames, phrase): return self.rows
    class Sam:
        def set_image(self, image, image_format): self.image = image
        def predict(self, box, multimask_output):
            # The middle frame is deliberately empty despite its interpolated
            # box: copied masks/features would falsely report a subject there.
            mask = np.zeros(self.image.shape[:2], bool)
            if self.image.any():
                x0,y0,x1,y1 = map(int, box); mask[y0:y1,x0:x1] = True
            return mask[None], np.array([.9]), None
    detector = Detector()
    provider = TemporalMobileSamSubjectMaskProvider(detector, Sam(), weights_sha256='test', prompt_policy=prompt_policy)
    frames = torch.ones((3,3,8,8),dtype=torch.uint8);frames[1] = 0
    masks = provider.masks_for(Path('a'), frames, 'person')
    assert masks.instance_present.flatten().tolist() == [True, False, True]
    detector.rows = [detection(), detection(), detection()]
    other = provider.masks_for(Path('b'), frames, 'person')
    assert not other.instance_present.any()
    assert not provider.last_direct_masks.instance_present.any()
