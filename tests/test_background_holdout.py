import hashlib
import json
from pathlib import Path

import pytest

from scripts.counterfactual import run_background_holdout as runner
from scripts.counterfactual.analyze_background_holdout import (
    assess_gates, evaluate_natural, localization_summary, verify_construction,
)


def test_holdout_rejects_an_unfrozen_or_changed_scoring_method(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    source = tmp_path/'scoring.py'
    source.write_text('fixed method')
    path = tmp_path/'protocol.json'
    protocol = {'status': 'draft', 'source_sha256': {'scoring.py': hashlib.sha256(source.read_bytes()).hexdigest()},
                'configuration_sha256': {}}
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='not been frozen'):
        runner.frozen_protocol(path)
    protocol['status'] = 'frozen_before_test_scoring'
    path.write_text(json.dumps(protocol))
    assert runner.frozen_protocol(path) == protocol
    source.write_text('changed after freezing')
    with pytest.raises(ValueError, match='scoring source changed'):
        runner.frozen_protocol(path)


def test_natural_paired_reference_is_official_even_for_sorted_method_names():
    videos, pairs = {}, []
    for prompt in ('a', 'b'):
        pairs.append(dict(prompt_id=prompt, video_a_uid=prompt+'1',
                          video_b_uid=prompt+'2', human_label='1'))
        for index in (1, 2):
            videos[prompt+str(index)] = {'scores': {
                m: {'status': 'succeeded', 'score': (index if m == 'aggregation' else -index)}
                for m in ('aggregation', 'official', 'repair')}}
    protocol = {'methods': ['aggregation', 'official', 'repair'], 'analysis': {
        'primary_tie_margin': 0, 'bootstrap_resamples': 100, 'bootstrap_seed': 42}}
    result = evaluate_natural(pairs, videos, protocol)['methods']
    assert result['repair']['accuracy_full_denominator'] == 1
    assert result['repair']['paired_delta_vs_official_full_denominator'] == 0
    assert result['repair']['paired_delta_ci95'] == [0, 0]
    assert result['aggregation']['paired_delta_vs_official_full_denominator'] == -1


def test_pixel_verification_requires_every_candidate_and_unchanged_background():
    protocol = {'test_dataset_index_sha256': 'index', 'construction_protocol_sha256': 'construction'}
    index = [{'video_uid': 'a', 'status': 'accepted'}]
    verification = {'dataset_index_sha256': 'index', 'protocol_sha256': 'construction', 'results': []}
    with pytest.raises(ValueError, match='coverage mismatch'):
        verify_construction(index, verification, protocol, {'a': {}})
    verification['results'] = [{'video_uid': 'a', 'status': 'accepted_replay_verified',
        'subject_blur_background_changed_pixels': 1, 'background_switch_subject_changed_pixels': 0,
        'variants': 8}]
    with pytest.raises(ValueError, match='invariant failed'):
        verify_construction(index, verification, protocol, {'a': {}})


def test_localization_distinguishes_missing_foreground_from_missing_background():
    rows = [{'status': 'completed', 'localizer_diagnostics': {
        'num_empty_foreground_frames': 4, 'num_frames': 4},
        'background_fraction': {'frame': [1, 1, 1, 1]}},
        {'status': 'completed', 'localizer_diagnostics': {
        'num_empty_foreground_frames': 0, 'num_frames': 2},
        'background_fraction': {'frame': [.01, .5]}}]
    result = localization_summary(rows)['natural']
    assert result['videos_without_foreground'] == 1
    assert result['frames_with_insufficient_background'] == 1
    assert result['mean_foreground_frame_presence'] == .5
    assert result['total_frames'] == 6


@pytest.mark.parametrize('failure', ['missing_score', 'partial_position', 'undefined_interval'])
def test_joint_gates_reject_incomplete_or_partial_success(failure):
    protocol = json.loads((Path(__file__).parents[1]/'configs/background-repair/holdout_protocol_v1.json').read_text())
    primary = protocol['primary_method']
    natural = {'methods': {primary: {'coverage': 1, 'paired_delta_ci95': [0, .02]}}}
    native = {'construction_status_counts': {'accepted': 2}, 'status_counts': {'completed': 2},
        'by_position': {}, 'scale_adjusted_response': {primary: {
            'background_response_retention': .8, 'paired_ratio_improvement_ci95': [.01, .1]}}}
    for position in ('full', 'start', 'middle', 'end'):
        native['by_position'][position] = {primary: {
            'foreground_absolute_change': {'mean': .005, 'n_bases': 2},
            'paired_improvement_vs_official': {'mean_ci95': [.001, .01]},
            'background_temporal_drop': {'mean_ci95': [.01, .02]},
            'background_drop_gt_foreground_abs': {'mean': .9}}}
    kwargs = {'max_parity_error': 0, 'clean_replay_error': 0}
    assert all(assess_gates(natural, native, protocol, **kwargs).values())
    if failure == 'missing_score':
        native['status_counts']['completed'] = 1
        key = 'counterfactual_scoring_coverage'
    elif failure == 'partial_position':
        native['by_position']['start'][primary]['foreground_absolute_change']['mean'] = .011
        key = 'foreground_absolute_start'
    else:
        native['scale_adjusted_response'][primary]['paired_ratio_improvement_ci95'] = None
        key = 'scale_adjusted_selectivity'
    gates = assess_gates(natural, native, protocol, **kwargs)
    assert not gates[key]
    assert not all(gates.values())
