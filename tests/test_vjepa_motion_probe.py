import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.counterfactual.vjepa_motion_probe import augment, feasibility, fresh_output, select_pairs, select_sources, statistics
from vbench_audit_models.vjepa import clean_state, check_namespace


CONFIG = json.loads((Path(__file__).parents[1] / 'configs/dynamic-static-jitter/vjepa-probe-v1.json').read_text())


def source(uid, prompt, split='dev', dimension='dynamics_degree', role='train'):
    return {'video_uid': uid, 'prompt_id': prompt, 'split': split, 'dimension': dimension,
            'relative_video_path': uid+'.mp4', 'role': role}


def test_score_blind_prompt_split_and_exclusions():
    config = copy.deepcopy(CONFIG)
    config['expected_sources'] = 6
    config['prompt_split_counts'] = {'train': 1, 'validation': 1, 'calibration_reserved': 1}
    pool = [source(str(i), f'p{i//2}') for i in range(6)]
    pool += [source('x', 'excluded'), source('y', 'p3', split='test'), source('z', 'p4', dimension='subject_consistency')]
    rows = select_sources(pool, [source('unused', 'excluded')], config)
    assert len(rows) == 6
    assert rows == select_sources(list(reversed(pool)), [source('unused', 'excluded')], config)
    assert all(len({r['role'] for r in rows if r['prompt_id'] == p}) == 1 for p in ['p0','p1','p2'])


def test_pair_filter_precedes_label_access():
    rows = [{'dimension': 'dynamics_degree', 'split': 'test'}, {'dimension': 'subject_consistency', 'split': 'dev'},
            {'dimension': 'dynamics_degree', 'split': 'dev', 'video_a_uid': 'x', 'video_b_uid': 'y'}]
    assert select_pairs(rows, []) == []
    valid = dict(dimension='dynamics_degree', split='dev', video_a_uid='a', video_b_uid='b',
                 prompt_id='p', human_label='1', label_source='official_human_anno')
    assert select_pairs([valid], [source('a','p'), source('b','p')])[0]['label'] == 1
    with pytest.raises(ValueError, match='crosses'):
        select_pairs([valid], [source('a','p'), source('b','p',role='validation')])
    bad = dict(valid)
    del bad['human_label']
    assert select_pairs([bad], [source('a','p',role='calibration_reserved'), source('b','p',role='calibration_reserved')]) == []


@pytest.mark.parametrize('view', ['alternating', 'aperiodic'])
def test_local_warp_is_reproducible_native_time_and_bounded(view):
    frames = np.random.default_rng(3).integers(0,256,(16,64,64,3),dtype=np.uint8)
    output, info = augment(frames, 'example', view, CONFIG)
    repeated, again = augment(frames, 'example', view, CONFIG)
    assert np.array_equal(output, repeated) and info == again
    assert output.shape == frames.shape and output.dtype == frames.dtype
    assert not np.array_equal(output, frames)
    assert info['max_displacement_pixels'] == pytest.approx(8, abs=1e-5)
    assert abs(np.mean(info['phase'])) < 1e-6
    assert np.array_equal(output[:,0], frames[:,0])
    assert np.array_equal(output[:,-1], frames[:,-1])
    assert not info['quality_excluded'] and not info['human_reviewed']


def test_head_continuous_loss_direction_and_constant_rejection():
    torch = pytest.importorskip('torch')
    from dynamic_degree.learned_probe import MotionProbe, motion_losses
    model = MotionProbe(input_dim=8,hidden_dim=4,mlp_dim=3)
    x = torch.randn(4,6,8)
    values = model(x)
    assert values.shape == (4,) and torch.isfinite(values).all()
    pairs = torch.tensor([[0,1],[2,3]])
    labels = torch.tensor([1.,.5])
    correct = torch.tensor([[.5,.5,.5],[-.5,-.5,-.5],[0.,0.,0.],[0.,0.,0.]])
    zero_loss, _ = motion_losses(correct,pairs,labels,1.,.01)
    assert zero_loss == 0
    wrong_loss, _ = motion_losses(-correct,pairs,labels,1.,.01)
    assert wrong_loss > 0
    collapsed_loss, _ = motion_losses(torch.zeros(4,3),pairs,labels,1.,.01)
    assert collapsed_loss == .5
    shifted = correct.clone()
    shifted[:,1:] += 1
    loss, parts = motion_losses(shifted,pairs,labels,1.,.01)
    assert loss > zero_loss and parts['consistency'] == 1
    loss, _ = motion_losses(values[:,None],pairs,labels,0.,.01)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_statistics_are_source_equal_weight_and_label_oriented():
    sources = [source('a','p'), source('b','p'), source('c','p')]
    pairs = [dict(a='a',b='b',label=1.),dict(a='a',b='c',label=.5)]
    q = np.array([[1,2,0],[0,1,-1],[1,2,0]],float)
    summary = statistics(sources,pairs,q)
    assert summary['ordered_correct_by_view'] == [1,1,1]
    assert summary['latent']['signed_change'] == 0
    assert summary['latent']['mean_absolute_change'] == 1
    collapsed = statistics(sources,pairs,np.zeros((3,3)))
    gate = feasibility(summary,collapsed,CONFIG['feasibility_gate'])
    assert not gate['passed'] and not gate['checks']['not_constant']


def test_checkpoint_cleaning_rejects_collisions():
    assert clean_state({'module.backbone.x': 1}) == {'x': 1}
    with pytest.raises(ValueError,match='colliding'):
        clean_state({'module.x':1, 'x':2})


def test_wrong_namespace_and_frozen_output(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'src',SimpleNamespace(__file__='/unrelated/src/__init__.py'))
    with pytest.raises(RuntimeError,match='foreign'):
        check_namespace(tmp_path)
    from scripts.counterfactual.static_jitter import ROOT
    with pytest.raises(ValueError,match='frozen'):
        fresh_output(ROOT / 'results' / 'no-write')
    assert not (ROOT / 'results' / 'no-write').exists()


def test_confound_diagnostic_fits_training_labels_only():
    from scripts.counterfactual.audit_vjepa_motion_probe import generator_confound
    sources=[dict(source('a','p'),generator='g1'),dict(source('b','p'),generator='g2'),
             dict(source('c','p'),generator='g3')]
    pairs=[dict(a='a',b='b',label=1.,role='train'),dict(a='b',b='c',label=.5,role='train'),
           dict(a='a',b='b',label=1.,role='validation')]
    first=generator_confound(sources,pairs)
    pairs[-1]['label']=0.
    second=generator_confound(sources,pairs)
    assert first['coefficients']==second['coefficients']
    assert first['counts']['validation']['correct']==1
    assert second['counts']['validation']['correct']==0
