import pytest

from scripts.counterfactual.analyze_subject_stability import summarize, validate_integrity_receipt, review_qualified_records


def row(uid, clean, bg, fg):
    def variant(origin, repair):
        return {'scores': {'official': {'status':'succeeded','score':origin},
                           'tracked_exclude': {'status':'succeeded' if repair is not None else 'undefined','score':repair}}}
    return {'base':{'video_uid':uid,'prompt_id':uid},'construction_status':'accepted','status':'completed',
            'variants':{'clean':variant(.9,clean), 'start/background_corrupt':variant(.7,bg),
                        'start/subject_corrupt':variant(.8,fg)}}


def test_missing_case_stays_in_denominator_and_prevents_full_cohort_pass():
    report,_ = summarize([row('a',.9,.905,.7),row('b',None,None,None)],['tracked_exclude'],resamples=100)
    x = report['positions']['start']['tracked_exclude']
    assert x['paired_scored'] == 1 and x['unscored'] == 1
    assert x['mean_target_on_scored_subset'] and not x['mean_target_on_complete_cohort']
    assert x['joint_success_fraction_full_constructed_denominator'] == .5


def test_zero_to_zero_is_not_success_and_opposite_changes_do_not_cancel():
    report,_ = summarize([row('a',0,0,0),row('b',.9,.92,.7),row('c',.9,.88,.7)],
                         ['tracked_exclude'],resamples=100)
    x = report['positions']['start']['tracked_exclude']
    assert x['joint_success_count'] == 0 and x['zero_to_zero'] == 1
    assert x['repair']['mean'] > .013
    assert not x['mean_target_on_complete_cohort']


def test_final_analysis_cannot_accept_wrong_or_incomplete_pixel_replay():
    p={'cohort_candidates':160,'dataset_index_sha256':'frozen'}
    receipt={'index_sha256':'frozen','bases':160,'outside_mask_changed_pixels':0,'verified_corrupted_frames':9600}
    validate_integrity_receipt(receipt,p,p)
    for field,bad in [('index_sha256','another'),('bases',159),('outside_mask_changed_pixels',1),('verified_corrupted_frames',0)]:
        with pytest.raises(ValueError):validate_integrity_receipt({**receipt,field:bad},p,p)


def test_pre_score_semantic_review_keeps_original_data_and_full_candidate_denominator():
    rows=[row('a',.9,.905,.7),row('b',.9,.2,.7)]
    qualified=review_qualified_records(rows,{'b':'Most of the actual subject was edited; reviewed before scoring'})
    report,_=summarize(qualified,['tracked_exclude'],resamples=20)
    assert report['total_candidates']==2 and report['constructed_denominator']==1
    assert report['status_counts']['input_review_rejected']==1
    assert rows[1]['construction_status']=='accepted' and rows[1]['status']=='completed'
    with pytest.raises(ValueError):review_qualified_records(rows,{'unknown':'bad mask'})
