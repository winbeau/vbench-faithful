import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.counterfactual.probe_native_region_motion import main, resolve_media, load_masks
from scripts.counterfactual.static_jitter import digest
from vbench_audit_models.sam_regions import pack_proposals


def test_native_probe_rejects_bad_shard_before_input_access():
    with pytest.raises(SystemExit) as e:
        main(['--manifest','absent','--review','absent','--region-run','absent',
              '--config','absent','--output','absent','--shard','2','--shards','2'])
    assert e.value.code==2


def test_relocation_requires_byte_identity_not_similar_filename(tmp_path):
    p=tmp_path/'official clip.mp4'; p.write_bytes(b'native')
    row={'video':'/missing/source/official clip.mp4','sha256':digest(p)}
    assert resolve_media(row,[tmp_path])==p
    p.write_bytes(b'reencoded')
    with pytest.raises(ValueError): resolve_media(row,[tmp_path])


def test_bound_cache_geometry_and_times_required(tmp_path):
    p=tmp_path/'regions.npz'; packed=pack_proposals([], (8,8))
    frames=np.zeros((3,8,8,3),np.uint8); t=np.array([0.,.1,.2])
    values={f'frame_{i}_{k}':v for i in range(3) for k,v in packed.items()}
    np.savez_compressed(p,**values,timestamps=t,input_shape=frames.shape)
    assert len(load_masks(p,digest(p),frames,t))==3
    with pytest.raises(ValueError): load_masks(p,digest(p),frames,t+.1)
    with pytest.raises(ValueError): load_masks(p,'wrong',frames,t)
