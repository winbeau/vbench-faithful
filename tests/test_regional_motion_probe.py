import json

import numpy as np
import pytest

from scripts.counterfactual.probe_regional_motion import frame_weights, main
from scripts.counterfactual.merge_regional_motion import merge_runs
from vbench_audit_models.sam_regions import pack_proposals


def test_no_regions_keeps_explicit_whole_frame_control_not_fake_foreground():
    packed = pack_proposals([], (16, 16))
    cache = {'frame_0_' + k: v for k, v in packed.items()}
    weights = frame_weights(cache, 0, (4, 4))
    assert weights.shape == (1, 16) and np.all(weights == 1)


def test_invalid_shard_rejected_before_source_access():
    with pytest.raises(SystemExit) as error:
        main(['--feature-run','absent','--region-run','absent','--config','absent',
              '--output','absent','--shard','2','--shards','2'])
    assert error.value.code == 2


def fixture_shards(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / str(i); p.mkdir(); paths.append(p)
        identity = {k: {} for k in ('code_files','input_sha256')}
        identity.update({k:'bound' for k in ('config_sha256','script_sha256',
                                           'feature_provenance_sha256','region_provenance_sha256')})
        identity.update(config={'lags':[1,2]}, sharding={'shard':i,'shards':2,
            'full_cohort':['a','b'],'assigned':[['a'],['b']][i]})
        row = {'candidate_id':['a','b'][i], 'status':'diagnostic_only', 'score':None,
               'sampling':{'timestamps':[0,.1,.2]},
               'pairs':[{'start':start,'lag':lag,'regions':[{'region':0,'whole_frame_control':True}]}
                        for lag in (1,2) for start in range(3-lag)]}
        (p/'diagnostics.jsonl').write_text(json.dumps(row)+'\n')
        (p/'provenance.json').write_text(json.dumps(identity))
        (p/'runtime.json').write_text(json.dumps({'status':'finished','completed':1,'expected':1,'failed':0}))
    return paths


def test_complete_merge_restores_cohort_order(tmp_path):
    paths = fixture_shards(tmp_path)
    rows, identity = merge_runs(paths[::-1])
    assert [r['candidate_id'] for r in rows] == ['a','b']
    assert identity['score'] is None and identity['failed'] == 0


@pytest.mark.parametrize('defect', ['missing_shard','duplicate_shard','running','missing_phase','missing_control','changed_code'])
def test_merge_refuses_incomplete_or_incompatible_evidence(tmp_path, defect):
    paths = fixture_shards(tmp_path)
    if defect == 'missing_shard': paths = paths[:1]
    elif defect == 'duplicate_shard': paths = paths[:1] * 2
    elif defect == 'running':
        f=paths[0]/'runtime.json'; r=json.loads(f.read_text()); r['status']='running'; f.write_text(json.dumps(r))
    elif defect == 'changed_code':
        f=paths[0]/'provenance.json'; r=json.loads(f.read_text()); r['code_files']={'x':'changed'}; f.write_text(json.dumps(r))
    else:
        f=paths[0]/'diagnostics.jsonl'; r=json.loads(f.read_text())
        if defect == 'missing_phase': r['pairs'].pop()
        else: r['pairs'][0]['regions']=[]
        f.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError): merge_runs(paths)
