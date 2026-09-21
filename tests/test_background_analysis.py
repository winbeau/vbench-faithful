import pytest

from scripts.counterfactual.analyze_background_native import dependency_components, native_cases, normalized_response


def test_shared_donor_and_reversed_roles_stay_in_one_cluster():
    result = dependency_components({'1': ('a', 'b'), '2': ('c', 'b'),
        '3': ('c', 'd'), '4': ('e', 'f'), '5': ('f', 'e')})
    assert result['1'] == result['2'] == result['3']
    assert result['4'] == result['5'] != result['1']


def test_failure_keeps_accepted_denominator_and_null_contrasts():
    rows = [{'video_uid': 'x', 'construction_status': 'accepted', 'status': 'failed',
             'base': {'prompt_id': 'a', 'generator': 'g'}, 'variants': {}}]
    cases = native_cases(rows, {'x': {'donor': {'prompt_id': 'b'}}}, methods=('official',))
    assert len(cases) == 4
    assert all(r['foreground_absolute_change'] is None for r in cases)
    assert all(r['background_drop_gt_foreground_abs'] is None for r in cases)


def test_uniform_score_compression_does_not_improve_normalized_invariance():
    cases = []
    for uid in ('a', 'b', 'c'):
        for method, fg, bg in (('official', .02, .1), ('scaled', .01, .05), ('region', .005, .08)):
            for position in ('full', 'start', 'middle', 'end'):
                cases.append(dict(base_id=uid, prompt_id=uid, donor_prompt_id=uid+'-donor',
                    method=method, position=position, foreground_absolute_change=fg,
                    background_temporal_drop=None if position == 'full' else bg))
    scaled = normalized_response(cases, 'scaled', resamples=100)
    assert scaled['paired_ratio_improvement'] == pytest.approx(0)
    assert scaled['paired_ratio_improvement_ci95'] == pytest.approx([0, 0])
    assert scaled['background_response_retention'] == pytest.approx(.5)
    region = normalized_response(cases, 'region', resamples=100)
    assert region['paired_ratio_improvement_ci95'][0] > 0
    assert region['background_response_retention'] == pytest.approx(.8)
