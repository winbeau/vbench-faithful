import numpy as np
import pytest

from vbench_audit_models.semantic_foreground import proposal_views, select_semantic_mask


def proposal(mask):
    return {'segmentation': mask, 'predicted_iou': .95, 'stability_score': .98, 'bbox': [0, 0, 20, 20]}


def test_semantics_rejects_larger_wall_without_rejecting_corner_object():
    wall = np.zeros((20, 20), np.uint8); wall[:15, 12:] = 1
    tub = np.zeros_like(wall); tub[12:, :10] = 1
    geometry = {'minimum_area': .01, 'maximum_area': .6, 'maximum_touched_edges': 4}
    mask, trace = select_semantic_mask([proposal(wall), proposal(tub)],
        [{'object_margin': -.03}, {'object_margin': .06}], wall.shape, geometry, {'minimum_object_margin': .02})
    np.testing.assert_array_equal(mask, tub)
    assert trace['candidates'][1]['touched_edges'] == 2 and trace['selected_index'] == 1
    empty, trace = select_semantic_mask([proposal(wall)], [{'object_margin': -.03}], wall.shape, geometry, {'minimum_object_margin': .02})
    assert not empty.any() and trace['selected_index'] is None


def test_crops_preserve_masked_pixels_and_do_not_reuse_other_frame_pixels():
    mask = np.zeros((10, 10), np.uint8); mask[2:6, 3:8] = 1
    original = np.full((10, 10, 3), 100, np.uint8)
    changed = original.copy(); changed[mask.astype(bool)] = 200
    a = proposal_views(original, mask, padding_fraction=.2, fill_rgb=[128]*3)
    b = proposal_views(changed, mask, padding_fraction=.2, fill_rgb=[128]*3)
    assert np.sum(np.asarray(a[1]) == 100) == 20*3
    assert np.sum(np.asarray(b[1]) == 200) == 20*3
    assert (np.asarray(a[1])[0, 0] == 128).all()
    with pytest.raises(ValueError, match='nonempty binary'):
        proposal_views(original, mask*0, padding_fraction=.2, fill_rgb=[128]*3)
