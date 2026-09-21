import pytest

from scripts.counterfactual.analyze_subject_natural import evaluate_population


def video(**values):
    return {'scores': {k: {'status': 'succeeded' if v is not None else 'failed', 'score': v}
                       for k, v in values.items()}}


def test_failed_pairs_keep_fixed_denominator_and_are_not_predicted_ties():
    videos = {'a':video(official=.9,repair=.9),'b':video(official=.8,repair=.8),
              'c':video(official=.8,repair=None)}
    pairs = [{'video_a_uid':'a','video_b_uid':'b','prompt_id':'p1','human_label':'1'},
             {'video_a_uid':'b','video_b_uid':'c','prompt_id':'p2','human_label':'0.5'}]
    r=evaluate_population(pairs,videos,['official','repair'],{'official':0,'repair':0},resamples=100)
    assert r['methods']['official']['accuracy_full_denominator']==1
    assert r['methods']['repair']['accuracy_full_denominator']==.5
    assert r['methods']['repair']['accuracy_scored_pairs']==1
    assert r['methods']['repair']['coverage']==.5


def test_repeated_pairs_from_one_prompt_do_not_create_independent_precision():
    videos={'a':video(official=.1,repair=.9),'b':video(official=.8,repair=.8)}
    pairs=[{'video_a_uid':'a','video_b_uid':'b','prompt_id':f'p{group}','human_label':str(1-group)}
           for group in (0,1) for repeat in range(20)]
    r=evaluate_population(pairs,videos,['official','repair'],{'official':0,'repair':0},resamples=1000)
    assert r['source_prompt_clusters']==2
    assert r['methods']['repair']['paired_delta_vs_official_full_denominator']==pytest.approx(0)
    assert r['methods']['repair']['paired_delta_ci95']==[-1.,1.]
