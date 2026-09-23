import json

import pytest
import numpy as np

from scripts.counterfactual.probe_regional_geometry import main, replay_pair, validate_source
from scripts.counterfactual.static_jitter import digest


def test_region_without_sparse_matches_remains_present_and_insufficient():
    pair = {'start': 2, 'lag': 3, 'seconds': .375, 'sift_matches': [],
            'regions': [{'region': 0, 'area_pixels': 17, 'whole_frame_control': False,
                         'sift_match_source_keys': []}]}
    r = replay_pair(pair)['regions'][0]
    assert r['area_pixels'] == 17 and r['score'] is None
    assert len(r['proposals']) == 6 and all(p['full']['matrix'] is None for p in r['proposals'])


def test_common_sift_mode_never_fills_missing_motion_with_zero():
    pair={'start':0,'lag':1,'seconds':.125,'sift_matches':[],
          'regions':[{'region':0,'area_pixels':10,'whole_frame_control':False,
                      'sift_match_source_keys':[],'hypotheses':[],'displacement_pixels':None}]}
    r=replay_pair(pair,np.zeros((0,8,8),bool))['regions'][0]
    assert r['compatible_displacement_pixels'] is None and r['score'] is None
    assert r['reliability']=='NOT CERTIFIED'


def test_source_hash_and_cohort_are_required(tmp_path):
    p = tmp_path / 'diagnostics.jsonl'; p.write_text('{}\n')
    identity = dict(status='diagnostic_only', score=None, completed=1,
                    common_identity={'full_cohort':['x']}, diagnostics_sha256=digest(p))
    (tmp_path / 'provenance.json').write_text(json.dumps(identity))
    assert validate_source(tmp_path) == identity
    p.write_text('{}\n{}\n')
    with pytest.raises(ValueError): validate_source(tmp_path)


def test_probe_preserves_source_failures_no_zero_score(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    row = dict(candidate_id='x', base_id='base', prompt_id='prompt', family='original', seed=None,
               status='failed', score=None)
    p = source/'diagnostics.jsonl'; p.write_text(json.dumps(row)+'\n')
    (source/'provenance.json').write_text(json.dumps(dict(status='diagnostic_only', score=None,
        completed=1, common_identity={'full_cohort':['x']}, diagnostics_sha256=digest(p))))
    output = tmp_path/'output'
    assert main(['--source-run',str(source),'--output',str(output)]) == 1
    r = json.loads((output/'diagnostics.jsonl').read_text())
    assert r['status'] == 'failed' and r['score'] is None and r['pairs'] == []
    with pytest.raises(FileExistsError): main(['--source-run',str(source),'--output',str(output)])
