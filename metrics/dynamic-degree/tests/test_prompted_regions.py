import numpy as np
import pytest

from dynamic_degree.prompted_regions import relocation_prompts


def test_native_xy_translation_clip_and_visible_interior_point():
    mask=np.zeros((20,30),bool);mask[3:10,4:13]=True
    proposals=[dict(displacement_pixels=d,origin='math_unit_test') for d in ([5,-1],[-10,0],[40,0])]
    a,b,c=relocation_prompts(mask,proposals)
    assert a['box_xyxy']==[9,2,18,9] and a['visible_prompt_pixels']==mask.sum()
    assert b['box_xyxy']==[0,3,3,10] and 0<=b['point_xy'][0]<3
    assert c['status']=='out_of_view_or_empty' and c['point_xy'] is None


def test_whole_frame_and_empty_masks_never_create_fake_support():
    h=[dict(displacement_pixels=[0,0],origin='unshifted_identity_proposal')]
    r=relocation_prompts(np.ones((7,13),bool),h)[0]
    assert r['box_xyxy']==[0,0,13,7] and 0<r['point_xy'][1]<6
    assert relocation_prompts(np.zeros((7,13),bool),h)[0]['status']=='out_of_view_or_empty'
    with pytest.raises(ValueError): relocation_prompts(np.full((7,13),.5),h)
    with pytest.raises(ValueError): relocation_prompts(np.ones((7,13),bool),[dict(displacement_pixels=[float('nan'),0],origin='bad')])
