from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from subject_consistency.isolation import isolate_subject_inputs
from subject_consistency.metric import evaluate_masked_batch
from subject_consistency.models import SubjectMasks


def fixture():
    frames = torch.zeros(3, 3, 16, 16, dtype=torch.uint8)
    frames[:, 0, 4:12, 4:12] = 220
    mask = torch.zeros(3, 1, 16, 16)
    mask[:, :, 4:12, 4:12] = 1
    present = torch.ones(3, 1, dtype=torch.bool)
    return frames, mask, present


@pytest.mark.parametrize("view", ["full", "crop"])
def test_background_pixels_cannot_reach_encoder_when_mask_is_fixed(view):
    frames, mask, present = fixture()
    changed = torch.where(mask.bool(), frames, torch.randint(0, 256, frames.shape, dtype=torch.uint8))
    before = frames.clone()
    a = isolate_subject_inputs(frames, mask, present, view=view)
    b = isolate_subject_inputs(changed, mask, present, view=view)
    assert torch.equal(a.frames, b.frames)
    assert torch.equal(a.masks, b.masks)
    assert torch.equal(frames, before)
    assert a.frames.dtype == torch.uint8


def test_crop_normalizes_translation_and_handles_the_image_boundary():
    frames, mask, present = fixture()
    shifted = torch.roll(frames, (-4, -4), (2, 3))
    shifted_masks = torch.roll(mask, (-4, -4), (2, 3))
    a = isolate_subject_inputs(frames, mask, present)
    b = isolate_subject_inputs(shifted, shifted_masks, present)
    assert torch.equal(a.frames, b.frames)
    assert torch.equal(a.masks, b.masks)
    assert b.boxes[0][0] < 0


class GloballyMixingEncoder:
    """Actual context mixing: background affects even foreground patch tokens."""
    def __init__(self, clips):
        self.module = SimpleNamespace(load_video=lambda path: clips[Path(path).name])

    def patches_from_frames(self, frames):
        rgb = frames.float() / 255
        local = torch.nn.functional.adaptive_avg_pool2d(rgb, (4, 4))
        mixed = local + 2 * rgb.mean((2, 3), keepdim=True)
        return mixed.flatten(2).transpose(1, 2), (4, 4)


def test_preencoding_fixes_real_context_mixing_without_erasing_subject_changes():
    frames, mask, present = fixture()
    background = frames.clone()
    background[1:, 2] = torch.where(mask[1:, 0].bool(), background[1:, 2], 255)
    subject = frames.clone()
    subject[1:, 0, 4:12, 4:12] = 0
    subject[1:, 2, 4:12, 4:12] = 220
    clips = {"clean": frames, "background": background, "subject": subject}
    provider = SimpleNamespace(masks_for=lambda *args: SubjectMasks(mask, present, "person", "independent_fixture"))
    metadata = {key: {"subject_en": "person"} for key in clips}
    def score(mode):
        values = evaluate_masked_batch([Path(key) for key in clips], metadata, "cpu", {}, provider,
                                      extractor=GloballyMixingEncoder(clips), encoding_mode=mode)
        assert all(v["status"] == "succeeded" for v in values)
        return [v["score"] for v in values]
    post = score("post_pool")
    assert post[0] - post[1] > .05
    for mode in ("preencode_full", "preencode_crop"):
        clean, background, subject = score(mode)
        assert clean == pytest.approx(background, abs=1e-7)
        assert clean - subject > .05


def test_absent_instances_cannot_leak_pixels_or_be_counted_as_evidence():
    frames, mask, present = fixture()
    present[:] = False
    result = isolate_subject_inputs(frames, mask, present)
    assert torch.all(result.frames == 128)
    assert not result.present.any()
    assert not result.masks.any()
    provider = SimpleNamespace(masks_for=lambda *args: SubjectMasks(mask, present, "person", "fixture"))
    value = evaluate_masked_batch([Path("clip")], {"clip": {"subject_en": "person"}}, "cpu", {}, provider,
                                 extractor=GloballyMixingEncoder({"clip": frames}), encoding_mode="preencode_crop")[0]
    assert value["status"] == "succeeded"
    assert value["score"] == 0


def test_wrong_resolution_or_nonfinite_masks_fail_explicitly():
    frames, mask, present = fixture()
    with pytest.raises(ValueError, match="native decoded"):
        isolate_subject_inputs(frames, mask[:, :, :4, :4], present)
    mask[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        isolate_subject_inputs(frames, mask, present)


def test_upstream_float_decode_preserves_exact_native_rgb():
    frames, mask, present = fixture()
    assert torch.equal(isolate_subject_inputs(frames.float(), mask, present).frames,
                       isolate_subject_inputs(frames, mask, present).frames)
    invalid = frames.float()
    invalid[0, 0, 0, 0] = .5
    with pytest.raises(ValueError, match='exact byte values'):
        isolate_subject_inputs(invalid, mask, present)
