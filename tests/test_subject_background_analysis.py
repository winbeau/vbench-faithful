from scripts.counterfactual.analyze_subject_background_auto import cluster_summary, make_cases, summarize_cases


def test_prompt_bootstrap_preserves_duplicate_members():
    result = cluster_summary({'a': 0., 'b': 0., 'c': 1.}, {'a': 'p', 'b': 'p', 'c': 'q'}, resamples=1000)
    assert result['n_bases'] == 3 and result['n_prompt_clusters'] == 2
    assert result['mean'] == 1/3
    assert result['mean_ci95'] == [0., 1.]
    assert cluster_summary({'a': .5}, {'a': 'p'})['mean_ci95'] is None


def test_rejections_missing_scores_and_empty_masks_are_not_hidden():
    def variant(value):
        return {'scores': {'repair': {'status': 'succeeded' if value is not None else 'failed', 'score': value},
                           'official': {'status': 'succeeded', 'score': .9}},
                'localizer_diagnostics': {'num_frames': 4, 'num_missing_frames': 4}}
    base = {'base_id': 'a', 'prompt_id': 'p', 'subject_en': 'person'}
    records = [{'base': base, 'construction_status': 'accepted', 'status': 'completed',
                'variants': {'clean': variant(0.), 'full/background_corrupt': variant(0.),
                             'full/subject_corrupt': variant(None)}},
               {'base': {'base_id': 'b'}, 'construction_status': 'rejected', 'variants': {}}]
    cases = make_cases(records, 'full', 'repair')
    assert len(cases) == 1 and cases[0]['subject_drop'] is None
    stats = summarize_cases(cases)
    assert stats['subject_response_counts']['scored'] == 0
    assert stats['background_zero_scores_both_versions'] == 1
    assert stats['clean_and_background_masks_all_empty'] == 1
    assert stats['background_absolute_change']['n_bases'] == 1
