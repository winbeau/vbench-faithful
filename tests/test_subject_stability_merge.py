import json

import pytest

from scripts.counterfactual.common import sha256_file
from scripts.counterfactual.merge_subject_stability import merge
from scripts.counterfactual.subject_artifacts import write_json, write_jsonl


def shard(root, index, uid, code='same'):
    path=root/f'shard{index}'
    write_jsonl(path/'scores.jsonl',[{'base':{'video_uid':uid},'status':'completed'}])
    write_json(path/'protocol.json',{'cohort_candidates':2})
    write_json(path/'run.json',{'completed':True,'scores_sha256':sha256_file(path/'scores.jsonl'),
        'base_count':1,'num_shards':2,'shard_index':index,'protocol_sha256':'p',
        'dataset_index_sha256':'d','source_sha256':code,'origin_reference':'upstream',
        'completed_bases':1,'runtime_failures':0,'maximum_origin_reference_error':1e-8,
        'physical_gpu':str(index),'started_unix':10+index,'finished_unix':20+index})


def test_merge_keeps_full_population_provenance_and_parallel_wall_time(tmp_path):
    shard(tmp_path/'in',0,'a');shard(tmp_path/'in',1,'b')
    run=merge(tmp_path/'in',tmp_path/'out')
    assert run['completed_bases']==2 and run['wall_seconds']==11
    rows=[json.loads(x) for x in (tmp_path/'out/scores.jsonl').read_text().splitlines()]
    assert [r['source_shard'] for r in rows]==['shard0','shard1']
    assert len(run['shards'])==2


@pytest.mark.parametrize('fault',['missing','duplicate','code'])
def test_merge_rejects_incomplete_duplicate_and_mixed_code(tmp_path,fault):
    shard(tmp_path/'in',0,'a')
    if fault!='missing':shard(tmp_path/'in',1,'a' if fault=='duplicate' else 'b',code='different' if fault=='code' else 'same')
    with pytest.raises(ValueError):merge(tmp_path/'in',tmp_path/'out')
    assert not (tmp_path/'out').exists()
