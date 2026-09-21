import json

import pytest

from scripts.counterfactual.generate_subject_masks_parallel import merge_shards
from scripts.counterfactual.subject_artifacts import write_json, write_jsonl


def make_shard(root, name, uid, status='accepted'):
    base = {'base_id':uid,'video_uid':uid}
    entry = {'base':base,'status':status,'rejection_reasons':[],
             'mask_file':{'path':f'masks/{uid}.npz','sha256':'original-mask'},
             'frames':[{'path':f'frames/{uid}/000000.png','sha256':'original-frame'}]}
    write_json(root/name/'manifests'/f'{uid}.json',entry)
    write_jsonl(root/name/'index.jsonl',[{'video_uid':uid,'status':status,
                'rejection_reasons':[],'manifest':f'manifests/{uid}.json'}])
    return base


def test_merge_retains_rejects_and_preserves_original_artifact_hashes(tmp_path):
    a=make_shard(tmp_path,'shard0','a');b=make_shard(tmp_path,'shard1','b','rejected')
    summary=merge_shards(tmp_path,{'a':a,'b':b})
    assert summary['total_bases']==2 and summary['accepted']==1 and summary['rejected']==1
    entry=json.loads((tmp_path/'manifests/a.json').read_text())
    assert entry['mask_file']=={'path':'shard0/masks/a.npz','sha256':'original-mask'}
    assert entry['frames'][0]['path']=='shard0/frames/a/000000.png'
    assert entry['frames'][0]['sha256']=='original-frame'
    assert json.loads((tmp_path/'shard0/manifests/a.json').read_text())['frames'][0]['path']=='frames/a/000000.png'


def test_merge_refuses_duplicate_or_incomplete_cohorts(tmp_path):
    a=make_shard(tmp_path,'shard0','a')
    with pytest.raises(ValueError,match='incomplete'):
        merge_shards(tmp_path,{'a':a,'b':{'base_id':'b','video_uid':'b'}})
    make_shard(tmp_path,'shard1','a')
    with pytest.raises(ValueError,match='duplicate'):
        merge_shards(tmp_path,{'a':a})
