"""Model-free scientific contracts; synthetic pixels are not research data."""
import json

import cv2
import numpy as np
import pytest

from scripts.counterfactual.build_region_discrimination import build, verify
from scripts.counterfactual.generate_subject_masks import generate
from scripts.counterfactual.region_discrimination import (
    RejectedBase, bounding_box, box_area, clean_components, corrupt_image,
    intersection_area, mirror_box, refine_grabcut, region_discrimination, window_indices,
)
from scripts.counterfactual.subject_artifacts import new_output, safe_output, tree_hashes


def pixels():
    frames = np.random.default_rng(17).integers(0, 256, (8, 80, 101, 3), dtype=np.uint8)
    masks = np.zeros(frames.shape[:3], np.uint8)
    masks[:, 15:55, 5:25] = 1
    masks[:, 15:25, 5:15] = 0  # Nonrectangular: equal boxes alone is insufficient.
    return frames, masks, np.zeros_like(masks)


@pytest.mark.parametrize("width,box", [(101, (5, 1, 25, 30)), (100, (0, 0, 20, 40))])
def test_mirror_has_exact_area_and_is_an_involution(width, box):
    mirrored = mirror_box(box, width)
    assert box_area(mirrored) == box_area(box)
    assert mirror_box(mirrored, width) == box
    assert intersection_area(box, mirrored) == 0


@pytest.mark.parametrize("position", ["start", "middle", "end"])
@pytest.mark.parametrize("operator", ["gaussian", "mosaic"])
def test_both_controls_have_equal_support_and_zero_change_outside(position, operator):
    frames, masks, other = pixels()
    before = frames.copy()
    family = region_discrimination(frames, masks, other, subject="person", position=position, operator=operator, background_mode="mirror")
    assert np.array_equal(frames, before)
    assert np.array_equal(family.frames["clean"], frames)
    indices = window_indices(len(frames), position)
    for level, support in (("subject_corrupt", masks), ("background_corrupt", masks[:, :, ::-1])):
        delta = np.any(family.frames[level] != frames, axis=-1)
        assert not delta[support == 0].any()
        assert delta.sum() > 0
        assert not delta[[i for i in range(len(frames)) if i not in indices]].any()
    for proof in family.proofs:
        assert proof["subject_mask_area"] == proof["background_mask_area"]
        assert proof["subject_mask_area"] < proof["box_area"]
        assert proof["subject_pixels_changed_by_background"] == 0
        assert proof["subject_corrupt"]["outside_mask_changed_pixels"] == 0


def test_reflection_overlap_is_rejected_not_clipped_or_shifted():
    frames, masks, other = pixels()
    masks[:] = 0
    masks[:, 20:50, 40:60] = 1
    with pytest.raises(RejectedBase, match="mirror_overlaps"):
        region_discrimination(frames, masks, other, subject="person", background_mode="mirror")


def test_other_instance_anywhere_in_mirror_box_rejects_even_outside_edit_mask():
    frames, masks, other = pixels()
    box = mirror_box(bounding_box(masks[3]), frames.shape[2])
    other[3, box[1], box[2] - 1] = 1
    assert masks[3, :, ::-1][box[1], box[2] - 1] == 0
    with pytest.raises(RejectedBase, match="mirror_contains_other_instance"):
        region_discrimination(frames, masks, other, subject="person", position="middle", background_mode="mirror")


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_window_rejected(count):
    with pytest.raises(RejectedBase, match="insufficient_frames"):
        window_indices(count, "middle")


def test_grabcut_and_png_replay_are_byte_deterministic(tmp_path):
    from scripts.counterfactual.subject_artifacts import write_npz, write_png_sequence
    cv2.setNumThreads(1)
    frames, masks, _ = pixels()
    image = np.full_like(frames[0], 15)
    image[masks[0] > 0] = [230, 30, 100]
    results = []
    for name in ("first", "second"):
        root = new_output(tmp_path / name)
        refined = refine_grabcut(image, masks[0], person=True)
        edited = corrupt_image(image, refined, operator="gaussian", scale=18)
        write_npz(root / "masks.npz", masks=refined, area=refined.sum())
        write_png_sequence(root, "frames", np.stack([image, edited]))
        results.append(tree_hashes(root))
    assert results[0] == results[1]


def test_component_cleanup_fills_only_small_internal_holes_and_keeps_largest():
    mask = np.zeros((60, 80), np.uint8)
    mask[5:50, 5:50] = 1
    mask[10:14, 10:14] = 0
    mask[25:40, 25:40] = 0  # 225 >= 200, not filled.
    mask[10:20, 60:70] = 1
    result = clean_components(mask, person=True)
    assert result[11, 11] == 1
    assert result[30, 30] == 0
    assert result[15, 65] == 0


@pytest.mark.parametrize('reference_short_side',[None,256])
def test_complete_dataset_replay_rechecks_pixels_and_every_artifact(tmp_path,reference_short_side):
    frames, masks, _ = pixels()
    frames[:] = 15
    frames[masks > 0] = [230, 30, 100]

    class FakeLocalizer:
        id2label = {0: "sky", 1: "person"}
        provenance = {"role": "construction", "family": "segformer", "test_fixture": True}

        def labels_for(self, frame):
            return masks[0]

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.bin").write_bytes(frames.tobytes())
    mapping = {"classes": {"person": ["person"]}, "unoccupied_stuff_labels": ["sky"], "absence_scope": "synthetic fixture"}
    base = {"base_id": "b0", "video_uid": "v0", "subject_en": "person", "prompt_en": "a person",
            "relative_video_path": "video.bin"}
    source = new_output(tmp_path / "construction")
    summary = generate([base, {**base, "base_id": "b1", "video_uid": "v1", "subject_en": "cat"}],
                       input_dir, source, mapping, FakeLocalizer(), decoder=lambda _: frames)
    assert summary["accepted"] == 1
    assert summary["base_rejection_counts"] == {"class_not_mapped": 1}
    outputs = [new_output(tmp_path / name) for name in ("a", "b")]
    kwargs={}
    if reference_short_side:
        protocol=tmp_path/'normalized-protocol.json'
        protocol.write_text(json.dumps({'gaussian_reference_short_side':reference_short_side}))
        kwargs={'protocol_path':protocol,'gaussian_reference_short_side':reference_short_side}
    for workers, output in enumerate(outputs, start=1):
        assert build(source, output, workers=workers,**kwargs)["accepted"] == 1
        proof = verify(output)
        assert proof["outside_mask_changed_pixels"] == 0
        assert proof["verified_corrupted_frames"] == 64
    assert tree_hashes(outputs[0]) == tree_hashes(outputs[1])
    png = next((outputs[0] / "clips").rglob("*.png"))
    png.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(outputs[0])


def test_resolution_normalization_requires_its_own_protocol_and_preserves_subject(tmp_path):
    with pytest.raises(ValueError,match='separately frozen'):
        build(tmp_path,tmp_path,gaussian_reference_short_side=256)
    frames,masks,other=pixels()
    legacy=region_discrimination(frames,masks,other,subject='person',position='start')
    normalized=region_discrimination(frames,masks,other,subject='person',position='start',gaussian_reference_short_side=256)
    assert normalized.parameters['scale']/80 == legacy.parameters['scale']/256
    assert not np.array_equal(normalized.frames['background_corrupt'],legacy.frames['background_corrupt'])
    assert np.array_equal(normalized.frames['background_corrupt'][masks>0],frames[masks>0])


def test_protected_trees_cannot_be_outputs():
    from scripts.counterfactual.common import ROOT
    for name in ("data", "results", "splits", "runs"):
        with pytest.raises(ValueError, match="protected"):
            safe_output(ROOT / name / "new")


@pytest.mark.parametrize("position", ["full", "start", "middle", "end"])
def test_full_background_complement_preserves_every_subject_pixel(position):
    frames, masks, others = pixels()
    # A central subject and extra background objects remain eligible in v2.
    masks[:] = 0
    masks[:, 20:55, 35:65] = 1
    others[:, 10:20, 75:90] = 1
    family = region_discrimination(frames, masks, others, subject="person", position=position)
    output = family.frames["background_corrupt"]
    assert np.array_equal(output[masks > 0], frames[masks > 0])
    for t in window_indices(len(frames), position):
        assert np.any(output[t, 10:20, 75:90] != frames[t, 10:20, 75:90])
    for proof in family.proofs:
        assert proof["subject_mask_area"] + proof["background_mask_area"] == frames.shape[1] * frames.shape[2]
        assert proof["subject_pixels_changed_by_background"] == 0
    assert family.parameters["background_support"] == "subject_mask_complement"
