import numpy as np
import pytest

from vbench_audit_models.automatic_foreground import select_automatic_mask, SamFallbackForegroundProvider


POLICY = {'minimum_area': .01, 'maximum_area': .6, 'maximum_touched_edges': 1}


def record(mask, quality=.95):
    return {'segmentation': mask, 'predicted_iou': quality, 'stability_score': .98, 'bbox': [0, 0, 20, 20]}


def test_large_border_background_does_not_displace_interior_object():
    sky = np.zeros((20, 20), np.uint8); sky[:10] = 1
    object_mask = np.zeros_like(sky); object_mask[5:15, 6:14] = 1
    tiny = np.zeros_like(sky); tiny[8:10, 8:10] = 1
    selected, trace = select_automatic_mask([record(sky), record(object_mask), record(tiny)], sky.shape, POLICY)
    np.testing.assert_array_equal(selected, object_mask)
    assert trace['selected_index'] == 1
    assert not trace['candidates'][0]['eligible']


def test_automatic_selection_does_not_force_twenty_percent_or_invent_a_mask():
    larger = np.zeros((20, 20), np.uint8); larger[3:17, 5:15] = 1
    selected, _ = select_automatic_mask([record(larger)], larger.shape, POLICY)
    assert selected.mean() == .35
    empty, trace = select_automatic_mask([], larger.shape, POLICY)
    assert not empty.any() and trace['selected_index'] is None
    with pytest.raises(ValueError, match='native binary'):
        select_automatic_mask([record(np.full((20, 20), 2))], (20, 20), POLICY)


def test_fallback_uses_each_actual_variant_and_preserves_existing_detections(monkeypatch):
    import torch
    from types import SimpleNamespace
    seen = []

    class Generator:
        def generate(self, image):
            seen.append(image.copy())
            mask = np.zeros(image.shape[:2], np.uint8)
            start = 3 if image[0, 0, 0] == 0 else 9
            mask[4:10, start:start+5] = 1
            return [record(mask)]

    class Base:
        provenance = {'role': 'scoring', 'construction_masks_reused': False}
        predictor = SimpleNamespace(model=object())

        def masks_for(self, frames):
            masks = np.zeros((len(frames), 20, 20), np.uint8)
            masks[0, 2:4, 2:4] = 1
            self.last_diagnostics = {'num_empty_foreground_frames': 1, 'foreground_fraction': masks.mean((1, 2)).tolist()}
            return masks

    monkeypatch.setattr('vbench_audit_models.automatic_foreground.build_automatic_generator', lambda *args: Generator())
    provider = SamFallbackForegroundProvider(Base(), POLICY)
    clean = torch.zeros((2, 3, 20, 20), dtype=torch.uint8)
    edited = clean.clone(); edited[1, 0] = 255
    a = provider.masks_for(clean); b = provider.masks_for(edited)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[0].sum() == 4
    assert not np.array_equal(a[1], b[1])
    assert len(seen) == 2 and seen[0][0, 0, 0] == 0 and seen[1][0, 0, 0] == 255
    assert provider.last_diagnostics['num_empty_foreground_frames'] == 0
    assert provider.last_diagnostics['num_empty_before_fallback'] == 1
