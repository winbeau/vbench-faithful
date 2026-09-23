import copy

import pytest

from scripts.counterfactual.audit_natural_motion_response import evaluate_pairs


def pair(a='a', b='b', label='1', prompt='p'):
    return dict(dimension='dynamics_degree', split='dev', video_a_uid=a, video_b_uid=b,
                human_label=label, prompt_id=prompt, video_a_path=a+'.mp4', video_b_path=b+'.mp4',
                label_source='official_human_anno')


def test_strict_ties_and_ordering_are_separate_no_margin_fit():
    scores = {'a': {'origin': 0., 'repair': .3}, 'b': {'origin': 0., 'repair': .2},
              'c': {'origin': 1., 'repair': .5}}
    result = evaluate_pairs([pair(), pair('b', 'c', '.5', 'q')], scores)
    assert result['ordered_pairs'] == result['human_tie_pairs'] == 1
    assert result['methods']['origin']['ordered_metric_ties'] == 1
    assert result['methods']['origin']['ordered_strict_agreement'] == 0
    assert result['methods']['repair']['ordered_strict_agreement'] == 1
    assert result['methods']['repair']['human_tie_absolute_gap_mean'] == pytest.approx(.3)
    assert not result['tie_margin_fitted']


def test_other_dimension_and_test_labels_never_consulted():
    good = pair()
    other, holdout = copy.copy(good), copy.copy(good)
    other.update(dimension='subject_consistency', human_label='not-a-label')
    holdout.update(split='test', human_label='not-a-label')
    result = evaluate_pairs([good, other, holdout], {'a': {'origin': 1., 'repair': .6}, 'b': {'origin': 0., 'repair': .1}})
    assert result['total_dynamic_dev_pairs'] == result['covered_pairs'] == 1


def test_partial_overlap_and_empty_ordered_stratum_are_explicit():
    result = evaluate_pairs([pair('a','b','.5'), pair('c','d')], {'a': {'origin': 0., 'repair': .1}, 'b': {'origin': 0., 'repair': .1}})
    assert result['pair_coverage_all_dev'] == .5
    assert result['ordered_pairs'] == 0
    assert result['ordered_delta_ci95_prompt_cluster'] is None
    assert result['methods']['repair']['ordered_strict_agreement'] is None


def test_duplicate_pairs_and_nonfinite_scores_fail():
    scores = {'a': {'origin': 0., 'repair': .1}, 'b': {'origin': 0., 'repair': .1}}
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_pairs([pair(),pair('b','a')], scores)
    scores['a']['repair'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        evaluate_pairs([pair()], scores)
