import json
from pathlib import Path

import pytest
import torch

from dynamic_degree.aligned_probe import aligned_losses
from dynamic_degree.anchored_probe import anchored_losses
from scripts.counterfactual.train_vjepa_aligned import train_mean, acceptance
from scripts.counterfactual.audit_vjepa_aligned import check_origin_record


ROOT = Path(__file__).parents[1]
PARENT = json.loads((ROOT / 'configs/dynamic-static-jitter/vjepa-anchored-v1.json').read_text())


def inputs():
    return torch.zeros(2, 8, requires_grad=True), torch.tensor([[0, 1], [0, 0]]), torch.tensor([1., .5])


def test_scale_term_trains_native_model_output_without_static_offset():
    q, pairs, labels = inputs()
    full, _ = aligned_losses(q, pairs, labels, PARENT['loss'], .8)
    old, _ = anchored_losses(q, pairs, labels, PARENT['loss'])
    difference = full - old
    assert difference.item() == pytest.approx(.09)
    difference.backward()
    assert q.grad[:, 0].lt(0).all()
    assert q.grad[:, 1:].eq(0).all()


def test_zero_weight_exactly_preserves_parent_training_objective():
    q, pairs, labels = inputs()
    old, _ = anchored_losses(q, pairs, labels, PARENT['loss'])
    new, _ = aligned_losses(q, pairs, labels, PARENT['loss'], .8, weight=0)
    assert torch.equal(new, old)


@pytest.mark.parametrize('target', [[0., 1.], float('nan'), -1, 2])
def test_per_video_teacher_or_invalid_mean_rejected(target):
    q, pairs, labels = inputs()
    with pytest.raises(ValueError, match='TRAIN population mean'):
        aligned_losses(q, pairs, labels, PARENT['loss'], target)


def test_validation_origin_scores_cannot_change_training_target():
    sources = [dict(video_uid='a', role='train'), dict(video_uid='b', role='train'), dict(video_uid='c', role='validation')]
    teacher = {'a': {'score': 0}, 'b': {'score': 1}, 'c': {'score': 0}}
    assert train_mean(sources, teacher) == .5
    teacher['c']['score'] = 1
    assert train_mean(sources, teacher) == .5


def test_scale_gate_does_not_erase_parent_natural_preference_failure():
    stats = dict(ordered_correct_base_alternating_aperiodic=[18, 19, 19],
        mean_by_view={'still': .01, 'still_jitter8': .01, 'base': .7},
        p95_by_view={'still': .01, 'still_jitter8': .01},
        weak_pan_above_still_fraction=1., strong_pan_above_weak_fraction=1., reversal_above_still_fraction=1.,
        native_jitter_mae=.01, native_base_std=.1)
    config = {'acceptance': {'validation_native_mean_abs_error_vs_origin_max': .05}}
    result = acceptance(stats, PARENT['acceptance'], .7, config)
    assert result['checks']['natural_scale_alignment']
    assert not result['checks']['natural_ordering']
    assert not result['all_passed']


@pytest.mark.parametrize('moving,expected', [(3, 0.), (4, 1.), (15, 1.)])
def test_origin_audit_keeps_strict_threshold_and_four_frame_rule(moving, expected):
    flags = [True] * moving + [False] * (15 - moving)
    row = dict(status='ok', score=expected, diagnostics=dict(
        sampling=dict(source_fps=8, sampled_frame_count=16, sampled_source_frame_indices=list(range(16)),
                      sampling_interval=1, timestamps=[i / 8 for i in range(16)], frame_shape=[256, 256]),
        official=dict(raw_flow_top5_mean=[7.] * moving + [6.] * (15 - moving), official_threshold=6.,
                      official_count_num=4, official_moving_flags=flags, official_moving_count=moving,
                      official_video_boolean=bool(expected))))
    assert check_origin_record(row) == expected
    row['score'] = 1 - expected
    with pytest.raises(AssertionError):
        check_origin_record(row)
