import numpy as np
import pytest

from scripts.counterfactual.audit_subject_score_evidence import condition_evidence, encoder_presence


def test_all_edited_frames_omitted_is_not_supported_stability():
    clean = np.ones(8, dtype=bool)
    background = clean.copy()
    background[:2] = False
    result = condition_evidence(clean, background, clean, [0, 1])
    assert result['background_edited_evidence_frames'] == 0
    assert not result['background_intervention_observed']
    assert result['subject_intervention_observed']
    assert not result['background_all_frames_have_evidence']


def test_missing_score_is_distinct_from_a_scored_clip_that_omits_the_window():
    clean = np.ones(8, dtype=bool)
    result = condition_evidence(clean, clean, None, [0, 1])
    assert result['subject_edited_evidence_frames'] is None
    assert result['background_intervention_observed']
    assert not result['subject_intervention_observed']
    with pytest.raises(ValueError, match='indices'):
        condition_evidence(clean, clean, clean, [8])


def test_encoder_presence_checks_pair_denominator_and_actual_mask_evidence():
    raw = np.array([True, True, True, False])
    source = np.array([.2, .2, .2, 0])
    score = {'status': 'succeeded', 'score': .9, 'diagnostics': {
        'coverage': [.3, .3, 0, 0], 'max_frames': None,
        'isolation': {'source_coverage': source.tolist()}, 'missing_policy': 'exclude',
        'num_frames': 4, 'num_present_frames': 2, 'num_missing_frames': 2,
        'pair_count': 1, 'pair_denominator': 6, 'score': .9}}
    assert encoder_presence(score, raw, source).tolist() == [True, True, False, False]
    score['diagnostics']['pair_count'] = 6
    with pytest.raises(ValueError, match='pair denominator'):
        encoder_presence(score, raw, source)
    score['diagnostics']['pair_count'] = 1
    raw[0] = False
    with pytest.raises(ValueError, match='without an actual scoring mask'):
        encoder_presence(score, raw, source)
