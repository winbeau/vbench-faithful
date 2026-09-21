import cv2
import numpy as np
import pytest

from scripts.counterfactual.region_discrimination import recover_grabcut_extent, RejectedBase
from scripts.counterfactual.recover_subject_construction import recover_entry
from scripts.counterfactual.common import sha256_file
from scripts.counterfactual.subject_artifacts import write_json, write_npz, write_png_sequence


def test_recovery_can_reach_beyond_old_five_pixel_extent_and_preserves_background():
    image=np.full((120,160,3),(20,130,20),np.uint8)
    image[25:90,35:115]=(190,40,40)
    mask=np.zeros((120,160),np.uint8);mask[35:75,45:105]=1
    out=recover_grabcut_extent(image,mask,person=False,padding_fraction=.25)
    assert out[85,80] == 1 and mask[85,80] == 0
    assert out[0,0] == 0 and out[-1,-1] == 0
    assert np.array_equal(out,recover_grabcut_extent(image,mask,person=False,padding_fraction=.25))


def test_empty_recovery_is_failure_not_fabricated_foreground():
    image=np.zeros((40,40,3),np.uint8);mask=np.zeros((40,40),np.uint8)
    with pytest.raises(RejectedBase):recover_grabcut_extent(image,mask,person=False,padding_fraction=.25)
    with pytest.raises(ValueError):recover_grabcut_extent(image,mask,person=False,padding_fraction=0)


def test_recovery_uses_construction_pixels_without_modifying_original_artifacts(tmp_path):
    import json
    source=tmp_path/'in';output=tmp_path/'out';output.mkdir()
    frames=np.full((2,120,160,3),(20,130,20),np.uint8);frames[:,25:90,35:115]=(190,40,40)
    masks=np.zeros((2,120,160),np.uint8);masks[:,35:75,45:105]=1
    path=source/'masks/a.npz';write_npz(path,masks=masks,other_instances=np.zeros_like(masks))
    original_hash=sha256_file(path)
    base={'video_uid':'a','base_id':'a','subject_en':'cow'}
    entry={'role':'construction','localizer':{'family':'segformer'},'base':base,
           'status':'accepted','rejection_reasons':[],
           'mask_file':{'path':'masks/a.npz','sha256':original_hash},
           'frames':write_png_sequence(source,'frames/a',frames)}
    write_json(source/'manifests/a.json',entry)
    row={'video_uid':'a','base_id':'a','manifest':'manifests/a.json','status':'accepted'}
    result=recover_entry((row,source,output,tmp_path,{'recovery_padding_fraction':.25}))
    assert result['status']=='accepted' and sha256_file(path)==original_hash
    new=json.loads((output/'manifests/a.json').read_text())
    assert new['area_pixels'][0]>int(masks[0].sum())
    assert new['frames'][0]['sha256']==entry['frames'][0]['sha256']
    entry['role']='scoring';write_json(source/'manifests/a.json',entry)
    with pytest.raises(ValueError,match='independent'):
        recover_entry((row,source,output,tmp_path,{'recovery_padding_fraction':.25}))


def test_failed_optional_recovery_does_not_erase_an_existing_valid_mask(tmp_path):
    import json
    source=tmp_path/'in';output=tmp_path/'out';output.mkdir()
    frames=np.zeros((2,100,100,3),np.uint8)
    masks=np.zeros((2,100,100),np.uint8);masks[:,30:35,20:80]=1
    path=source/'masks/a.npz';write_npz(path,masks=masks,other_instances=np.zeros_like(masks))
    entry={'role':'construction','localizer':{'family':'segformer'},
           'base':{'video_uid':'a','base_id':'a','subject_en':'cow'},'status':'accepted','rejection_reasons':[],
           'mask_file':{'path':'masks/a.npz','sha256':sha256_file(path)},
           'frames':write_png_sequence(source,'frames/a',frames)}
    write_json(source/'manifests/a.json',entry)
    row={'video_uid':'a','base_id':'a','manifest':'manifests/a.json','status':'accepted'}
    result=recover_entry((row,source,output,tmp_path,{'recovery_padding_fraction':.25,'recovery_keep_valid_initial':True}))
    assert result['status']=='accepted'
    new=json.loads((output/'manifests/a.json').read_text())
    assert new['frame_recovery_fallbacks']==['recovery_grabcut_unusable_seeds']*2
    with np.load(output/new['mask_file']['path']) as arrays:assert np.array_equal(arrays['masks'],masks)


@pytest.mark.parametrize('failure', ['seeds', 'area'])
def test_fresh_generation_honors_the_same_fallback_and_records_each_frame(tmp_path, monkeypatch, failure):
    import json
    from scripts.counterfactual.generate_subject_masks import generate, localize_frames
    mask = np.zeros((100, 100), np.uint8); mask[30:35, 20:80] = 1
    frames = np.zeros((8, 100, 100, 3), np.uint8)
    # An admissible initial refinement can lack the certain core required by
    # the optional second pass; the regression is at that handoff.
    monkeypatch.setattr('scripts.counterfactual.generate_subject_masks.refine_grabcut', lambda *a, **k: mask.copy())
    if failure == 'area':
        monkeypatch.setattr('scripts.counterfactual.region_discrimination.recover_grabcut_extent',
                            lambda *a, **k: np.ones_like(mask))
    class Localizer:
        id2label = {0: 'sky', 1: 'person'}
        provenance = {'role': 'construction', 'family': 'segformer', 'test_fixture': True}
        def labels_for(self, frame): return mask
    mapping = {'classes': {'person': ['person']}, 'unoccupied_stuff_labels': ['sky'],
               'absence_scope': 'synthetic fixture', 'recovery_padding_fraction': .25,
               'recovery_keep_valid_initial': True}
    (tmp_path / 'video.bin').write_bytes(frames.tobytes()); out = tmp_path / 'output'
    base = {'base_id': 'b0', 'video_uid': 'v0', 'subject_en': 'person',
            'prompt_en': 'a person', 'relative_video_path': 'video.bin'}
    assert generate([base], tmp_path, out, mapping, Localizer(), decoder=lambda _: frames)['accepted'] == 1
    saved = json.loads((out / 'manifests/b0.json').read_text())
    reason = 'recovery_grabcut_unusable_seeds' if failure == 'seeds' else 'recovered_area_outside_frozen_gates'
    assert saved['frame_recovery_fallbacks'] == [reason] * 8
    with np.load(out / saved['mask_file']['path']) as data:
        assert np.array_equal(data['masks'], np.repeat(mask[None], 8, axis=0))
    _, _, rejected = localize_frames(frames, 'person', Localizer(), {**mapping, 'recovery_keep_valid_initial': False})
    assert all(rejected)
