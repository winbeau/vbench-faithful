import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vbench_audit_models.sam_regions import load_local_sam, pack_proposals, unpack_masks, SamRegionModel
from vbench_audit_models.sam_regions import SamPromptRegionModel


def proposal(mask):
    return {"segmentation": mask, "area": int(mask.sum()), "predicted_iou": .99,
            "stability_score": .98, "bbox": [0, 0, mask.shape[1], mask.shape[0]],
            "point_coords": [[2, 3]]}


def test_lossless_odd_width_and_overlapping_regions_no_postfilter():
    mask = np.zeros((7, 13), bool); mask[1:5, 2:9] = True
    small = np.zeros_like(mask); small[3, 3] = True
    r = pack_proposals([proposal(mask), proposal(small)], mask.shape)
    assert np.array_equal(unpack_masks(r["masks_packed"], mask.shape), [mask, small])
    assert r["areas"].tolist() == [28, 1]
    assert float(r["union_fraction"]) == pytest.approx(28 / 91)


def test_empty_regions_explicit_and_pack_geometry_checked():
    r = pack_proposals([], (7, 13))
    assert unpack_masks(r["masks_packed"], (7, 13)).shape == (0, 7, 13)
    assert r["quality"].shape == (0, 2)
    assert float(r["union_fraction"]) == 0
    with pytest.raises(ValueError): unpack_masks(r["masks_packed"], (7, 17))


@pytest.mark.parametrize("field,value", [("area", 0), ("predicted_iou", float('nan')),
                                          ("segmentation", np.full((7, 13), 2))])
def test_invalid_proposals_rejected(field, value):
    r = proposal(np.ones((7, 13), bool)); r[field] = value
    with pytest.raises(ValueError): pack_proposals([r], (7, 13))


def test_missing_model_assets_fail_without_model_import(tmp_path):
    with pytest.raises(FileNotFoundError): SamRegionModel(tmp_path, tmp_path / 'absent', 'cpu', {})
    with pytest.raises(FileNotFoundError): load_local_sam(tmp_path)


def test_reject_other_imported_sam_source(tmp_path, monkeypatch):
    package = tmp_path / 'segment_anything'; package.mkdir()
    (package / 'build_sam.py').write_text('')
    monkeypatch.setitem(sys.modules, 'segment_anything', SimpleNamespace(__file__='/elsewhere/__init__.py'))
    with pytest.raises(RuntimeError, match="already imported"): load_local_sam(tmp_path)


def test_propose_checks_rgb_and_preserves_all_generator_output():
    model = object.__new__(SamRegionModel)
    expected = np.ones((7, 13), bool)
    model.generator = SimpleNamespace(generate=lambda frame: [proposal(expected)])
    result = model.propose(np.ones((7, 13, 3), np.uint8))
    assert np.array_equal(unpack_masks(result['masks_packed'], (7, 13))[0], expected)
    with pytest.raises(ValueError): model.propose(np.ones((7, 13, 3), float))


def test_prompted_sam_keeps_all_modes_hypotheses_and_empty_predictions():
    model = object.__new__(SamPromptRegionModel)
    calls=[]
    def predict(**kw):
        calls.append(kw)
        n=3 if kw['multimask_output'] else 1
        return np.zeros((n,7,13),bool),np.linspace(.1,.9,n),None
    model.predictor=SimpleNamespace(set_image=lambda frame,**kw:None,predict=predict)
    result=model.propose(np.ones((7,13,3),np.uint8),[dict(box_xyxy=[1,1,8,6],point_xy=[3,3])])
    assert len(result)==8 and len(calls)==4
    assert all(not r['segmentation'].any() for r in result)
    assert {r['uses_positive_point'] for r in result}=={False,True}
    assert model.propose(np.ones((7,13,3),np.uint8),[])==[]
    with pytest.raises(ValueError): model.propose(np.ones((7,13,3),np.uint8),[dict(box_xyxy=[1,1,80,6],point_xy=[3,3])])
