import json
from pathlib import Path

import numpy as np
import pytest

from vbench_audit_models.caption_union import caption_sam_union, select_caption_boxes


POLICY = json.loads((Path(__file__).parents[1] / 'configs/background-repair/scoring_coco80_caption_dev_v2.json').read_text())['caption_fallback']


class BoxPredictor:
    def __init__(self):
        self.images = []
        self.boxes = []

    def set_image(self, image, image_format):
        assert image_format == 'RGB'
        self.images.append(image.copy())

    def predict(self, *, box, multimask_output):
        assert not multimask_output
        self.boxes.append(box.tolist())
        height, width = self.images[-1].shape[:2]
        x0, y0, x1, y1 = map(int, box)
        mask = np.zeros((1, height, width), dtype=bool)
        mask[:, y0:y1, x0:x1] = True
        return mask, np.array([.9]), None


def instance(text, box):
    return {'text': text, 'box': box, 'score': .7}


def test_nonempty_tv_does_not_block_other_objects_and_scene_is_excluded():
    records = [instance('a wooden table', [1, 10, 18, 18]),
               instance('white curtain covering window', [1, 1, 6, 9]),
               instance('window covered by curtain', [7, 1, 15, 10])]
    predictor = BoxPredictor()
    image = np.zeros((20, 20, 3), np.uint8)
    base = np.zeros((20, 20), np.uint8)
    base[6:9, 12:17] = 1
    snapshot = base.copy()
    mask, trace = caption_sam_union(predictor, image, records, POLICY, base_mask=base)
    assert trace['selected_indices'] == [0, 1] and len(predictor.images) == 1
    assert mask[12:17, 3:17].all() and mask[2:8, 2:5].all()
    assert mask[6:9, 12:17].all() and not mask[1:5, 7:15].any()
    assert np.array_equal(base, snapshot)


def test_all_building_instances_are_kept_with_fixed_area_limits():
    records = [instance('a red house', [0, 0, 7, 9]),
               instance('a large building', [12, 0, 20, 10]),
               instance('the campus building', [0, 0, 20, 20])]
    boxes, trace = select_caption_boxes(records, (20, 20), POLICY)
    assert boxes == [[0, 0, 7, 9], [12, 0, 20, 10]]
    assert trace['selected_indices'] == [0, 1] and not trace['captions'][2]['eligible']


def test_empty_description_frame_does_not_reuse_another_images_mask():
    predictor = BoxPredictor()
    image = np.zeros((20, 20, 3), np.uint8)
    first, _ = caption_sam_union(predictor, image, [instance('a bed', [1, 2, 10, 12])], POLICY)
    second, trace = caption_sam_union(predictor, image + 1, [], POLICY)
    assert first.any() and not second.any() and trace['selected_indices'] == []


@pytest.mark.parametrize('bad', [np.zeros((10, 10)), np.full((20, 20), 2), np.full((20, 20), np.nan)])
def test_invalid_base_mask_rejected(bad):
    with pytest.raises(ValueError, match='base mask'):
        caption_sam_union(BoxPredictor(), np.zeros((20, 20, 3), np.uint8), [], POLICY, base_mask=bad)
