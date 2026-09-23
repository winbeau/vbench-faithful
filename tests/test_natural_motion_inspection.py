import numpy as np
import pytest

from scripts.counterfactual.inspect_natural_motion import contact_sheet, display_source, select_development


def test_reserved_sources_are_never_opened_or_selected():
    a=dict(split='dev',video_uid='a',relative_video_path='generator/a.mp4',dimension='dynamics_degree')
    # The reserved record deliberately has no media path at all.
    assert select_development([a,dict(split='test',video_uid='heldout')])==[a]
    with pytest.raises(ValueError): select_development([a,a])
    with pytest.raises(ValueError): select_development([{**a,'relative_video_path':'../escape'}])


def test_every_frame_has_its_own_display_cell():
    frames=np.stack([np.full((8,8,3),i*40,np.uint8) for i in range(5)])
    image=contact_sheet(frames,np.arange(5)/8,'unit display')
    for i in range(5):
        assert np.all(image[64+(i//4)*284+28:64+(i//4)*284+284,(i%4)*256:(i%4+1)*256]==i*40)


def test_unknown_gif_timing_only_displays_order_never_fabricates_fps(tmp_path):
    from PIL import Image
    frames=[Image.fromarray(np.full((8,8,3),i*40,np.uint8)) for i in range(3)]
    p=tmp_path/'untimed.gif';frames[0].save(p,save_all=True,append_images=frames[1:],loop=0)
    rgb,times,fps,info=display_source(p,allow_unknown_gif_timing=True)
    assert rgb.shape==(3,8,8,3) and times==[None]*3 and fps is None
    assert info['speed_calibration_eligible'] is False
    assert info['timing_status']=='unknown_missing_GIF_delays'
    assert contact_sheet(rgb,times,'untimed test').ndim==3
