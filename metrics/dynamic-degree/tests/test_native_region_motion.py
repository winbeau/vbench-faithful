import cv2
import numpy as np
import pytest

from dynamic_degree.native_region_motion import (
    NativeMotionConfig, fit_native_regions, mask_association, peaks,
    rgb_translation_surface, sift_correspondences, temporal_match_closure,
)


def fixture():
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (24, 30, 3), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], bool); mask[7:17, 9:21] = True
    return frame, mask


def shifted(frame, dx, dy):
    return cv2.warpAffine(frame, np.array([[1., 0., dx], [0., 1., dy]]),
                          frame.shape[1::-1], borderMode=cv2.BORDER_CONSTANT)


@pytest.mark.parametrize('delta', [(5, -3), (-6, 4), (0, 0)])
def test_full_domain_native_translation_sign_and_self(delta):
    a, m = fixture(); b = shifted(a, *delta)
    s = rgb_translation_surface(a, b, m)
    best = peaks(s, NativeMotionConfig())[0]
    assert best['displacement_pixels'] == list(delta)
    assert best['rgb_mse'] < 1e-12


def test_fft_matches_visible_pixel_sum_every_offset_with_crop():
    a, m = fixture(); b = shifted(a, 5, -3)
    s = rgb_translation_surface(a, b, m)
    sy, sx = np.nonzero(m)
    for dx, dy in ((0, 0), (5, -3), (-13, -12), (18, 2)):
        keep = (sx+dx >= 0) & (sx+dx < a.shape[1]) & (sy+dy >= 0) & (sy+dy < a.shape[0])
        select = (s['dx'] == dx) & (s['dy'] == dy)
        assert select.sum() == 1
        assert s['overlap'][select][0] == pytest.approx(keep.mean(), abs=1e-12)
        direct = np.mean((a[sy[keep], sx[keep]].astype(float)/255-b[sy[keep]+dy, sx[keep]+dx]/255)**2)
        assert s['loss'][select][0] == pytest.approx(direct, abs=1e-12)


def test_two_region_visibility_fft_matches_direct_sum():
    a,m=fixture();b=shifted(a,5,-3)
    tm=np.ones(m.shape,bool);tm[8:12,10:15]=False
    s=rgb_translation_surface(a,b,m,tm)
    sy,sx=np.nonzero(m)
    for dx,dy in ((0,0),(5,-3),(-13,-12)):
        inside=(sx+dx>=0)&(sx+dx<a.shape[1])&(sy+dy>=0)&(sy+dy<a.shape[0])
        sy1,sx1=sy[inside],sx[inside]
        keep=tm[sy1+dy,sx1+dx];sy1,sx1=sy1[keep],sx1[keep]
        select=(s['dx']==dx)&(s['dy']==dy)
        assert s['overlap'][select][0]==pytest.approx(len(sy1)/m.sum(),abs=1e-12)
        loss=np.mean((a[sy1,sx1].astype(float)/255-b[sy1+dy,sx1+dx]/255)**2)
        assert s['loss'][select][0]==pytest.approx(loss,abs=1e-12)


def test_new_foreground_occlusion_must_not_drag_static_background():
    rng=np.random.default_rng(8)
    background=rng.integers(177,184,(64,96,3),dtype=np.uint8)
    m=np.ones((64,96),bool);m[20:45,40:62]=False
    tm=np.ones_like(m);tm[20:45,26:48]=False
    a=background.copy();b=background.copy();a[~m]=10;b[~tm]=10
    # This synthetic unit case diagnoses correspondence geometry only; it is
    # not a replacement for the official-source counterfactual cohort.
    wrong=peaks(rgb_translation_surface(a,b,m),NativeMotionConfig())[0]
    assert abs(wrong['displacement_pixels'][0])>2
    fixed=fit_native_regions(a,b,m[None],tm[None],[],NativeMotionConfig(target_visibility='associated_sam'))[0]
    assert fixed['observable_translation'] and fixed['displacement_pixels']==[0.,0.]
    assert fixed['source_only_proposals'][0]['displacement_pixels']!=[0,0]


def test_target_region_visibility_preserves_actual_moving_object():
    a,m=fixture();b=shifted(a,5,-3);tm=shifted(m.astype(np.uint8),5,-3).astype(bool)
    r=fit_native_regions(a,b,m[None],np.stack([np.zeros_like(tm),tm]),[],
                         NativeMotionConfig(target_visibility='associated_sam'))[0]
    assert np.allclose(r['displacement_pixels'],[5,-3])
    assert r['hypotheses'][r['best_hypothesis']]['alignment_target_region']==1
    missing=fit_native_regions(a,b,m[None],np.zeros((0,*m.shape),bool),[],
                               NativeMotionConfig(target_visibility='associated_sam'))[0]
    assert missing['displacement_pixels'] is None and missing['hypotheses']==[]


def test_identical_and_fractional_full_frame_do_not_lock_camera_to_zero():
    rng=np.random.default_rng(9)
    # The exact subpixel construction is mathematical test evidence only.
    a=cv2.GaussianBlur(rng.integers(0,256,(24,30,3),dtype=np.uint8),(0,0),1)
    m=np.ones(a.shape[:2], bool)
    self_row=fit_native_regions(a,a,m[None],m[None],[])[0]
    assert self_row['displacement_pixels'] == [0.,0.]
    b=shifted(a,.25,-.25)
    row=fit_native_regions(a,b,m[None],m[None],[])[0]
    assert row['observable_translation']
    assert row['displacement_pixels'][0] > .1 and row['displacement_pixels'][1] < -.1


def test_flat_region_is_not_an_observed_zero():
    a=np.full((16,16,3),127,np.uint8); m=np.ones((16,16),bool)
    r=fit_native_regions(a,a,m[None],m[None],[])[0]
    assert r['displacement_pixels'] is None and not r['observable_translation']
    assert r['hypotheses']


def test_empty_and_small_regions_retained_not_zero_filled():
    a,m=fixture(); small=np.zeros_like(m); small[11,14]=True
    rows=fit_native_regions(a,a,np.stack([np.zeros_like(m),small]),m[None],[])
    assert len(rows)==2 and rows[0]['displacement_pixels'] is None
    assert rows[1]['area_pixels']==1
    assert rgb_translation_surface(a,a,np.zeros_like(m)) is None


def test_target_mask_index_not_assumed_equal_and_out_of_view_missing():
    _,m=fixture(); target=shifted(m.astype(np.uint8),5,-3).astype(bool)
    assoc=mask_association(m,np.stack([np.zeros_like(m),target]),[5,-3])
    assert assoc['target_region']==1 and assoc['iou']==pytest.approx(1)
    out=mask_association(m,target[None],[100,0])
    assert out['target_region'] is None and out['iou'] is None


@pytest.mark.parametrize('dtype', [bool, np.uint8, np.float32])
def test_association_cached_pair_path_equals_checked_public_path(dtype):
    from dynamic_degree.native_region_motion import _mask_association_validated
    _, m = fixture()
    targets = np.stack([m, shifted(m.astype(np.uint8), 5, -3).astype(bool)]).astype(dtype)
    for delta in ([0, 0], [5.2, -3.1], [100, 0]):
        expected = mask_association(m, targets, delta)
        actual = _mask_association_validated(m, targets.astype(np.float32), targets.sum((1, 2)), delta)
        assert actual == expected


def test_invalid_binary_masks_are_not_silently_coerced():
    a, m = fixture(); bad = m.astype(float); bad[1, 1] = .5
    with pytest.raises(ValueError): mask_association(bad, m[None], [0, 0])
    with pytest.raises(ValueError): mask_association(m, bad[None], [0, 0])
    with pytest.raises(ValueError): fit_native_regions(a, a, m[None], bad[None], [])
    with pytest.raises(ValueError): fit_native_regions(a, a, np.zeros((0, 3, 3)), m[None], [])


def test_vectorized_region_membership_preserves_round_clip_order():
    a, m = fixture()
    points = [[8.5, 6.5], [9.5, 7.5], [-2., -2.], [40., 25.], [12., 13.]]
    matches = [dict(source_key=i, target_key=i, source_xy=p, target_xy=p) for i, p in enumerate(points)]
    row = fit_native_regions(a, a, m[None], m[None], matches)[0]
    expected = [i for i, p in enumerate(points) if m[int(np.clip(round(p[1]), 0, m.shape[0]-1)),
                                                    int(np.clip(round(p[0]), 0, m.shape[1]-1))]]
    assert row['sift_match_source_keys'] == expected


def test_periodic_return_keeps_two_step_path_not_just_net_displacement():
    def match(q,t,a,b): return dict(source_key=q,target_key=t,source_xy=a,target_xy=b)
    pairs=[dict(start=0,lag=1,sift_matches=[match(0,1,[5,5],[8,5])]),
           dict(start=1,lag=1,sift_matches=[match(1,2,[8,5],[5,5])]),
           dict(start=0,lag=2,sift_matches=[match(0,2,[5,5],[5,5])])]
    c=temporal_match_closure(pairs)[0]
    assert c['consistent']==1 and c['two_step_path_pixels']==[6.]
    pairs[-1]['sift_matches'][0]['target_key']=3
    assert temporal_match_closure(pairs)[0]['consistent']==0


def test_sift_missing_and_reciprocal_ratio_matches():
    assert sift_correspondences({'descriptors':None},{'descriptors':None})==[]
    f={'descriptors':np.eye(3,dtype=np.float32),'xy':np.array([[0.,0.],[1.,2.],[3.,4.]])}
    g={**f,'xy':f['xy']+np.array([2.,-1.])}
    matches=sift_correspondences(f,g)
    assert len(matches)==3
    assert all(np.allclose(np.subtract(m['target_xy'],m['source_xy']),[2,-1]) for m in matches)


@pytest.mark.parametrize('bad', [0,1,float('nan')])
def test_invalid_sift_ratio(bad):
    with pytest.raises(ValueError): NativeMotionConfig(sift_ratio=bad)


def test_target_region_excludes_spurious_background_match_without_motion_floor():
    from dynamic_degree.native_region_motion import rank_with_common_sift
    masks=np.zeros((2,20,30),bool);masks[:,2:8,3:24]=True
    matches=[dict(source_key=0,source_xy=[10.,4.],target_xy=[7.,4.]),
             dict(source_key=1,source_xy=[12.,5.],target_xy=[12.,17.])]
    region=dict(sift_match_source_keys=[0,1],hypotheses=[
        dict(displacement_pixels=[0.,0.],alignment_target_region=0),
        dict(displacement_pixels=[-3.,0.],alignment_target_region=1)])
    r=rank_with_common_sift(region,masks,matches)
    assert r['common_source_keys']==[0] and r['excluded_source_keys']==[1]
    assert r['best_compatible_hypothesis']==1 and r['unique_witness_locations']==1
    assert r['score'] is None and r['reliability']=='NOT CERTIFIED'


def test_compatibility_cannot_cherry_pick_support_and_duplicate_orientations():
    from dynamic_degree.native_region_motion import rank_with_common_sift
    masks=np.ones((2,20,30),bool);masks[1,:10]=False
    matches=[dict(source_key=i,source_xy=[8.,4.],target_xy=[7.,4.]) for i in range(2)]
    region=dict(sift_match_source_keys=[0,1],hypotheses=[
        dict(displacement_pixels=[0.,0.],alignment_target_region=0),
        dict(displacement_pixels=[-1.,0.],alignment_target_region=1)])
    assert rank_with_common_sift(region,masks,matches)['best_compatible_hypothesis'] is None
    masks[:]=True;r=rank_with_common_sift(region,masks,matches)
    assert r['unique_witness_locations']==1 and len(r['common_source_keys'])==2
