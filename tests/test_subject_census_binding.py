import copy
import json

import pytest

from scripts.counterfactual.run_subject_official_census import bind_scoring


def test_census_binding_retains_rejected_candidates_and_does_not_change_policy(tmp_path):
    rows = [{'video_uid': 'accepted', 'status': 'accepted'}, {'video_uid': 'rejected', 'status': 'rejected'}]
    (tmp_path/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'protocol.json').write_text('{}\n')
    plan = {'videos': 2, 'scoring_template': {'input_review_exclusions': {'rejected': 'known input failure'},
        'cohort_candidates': 2, 'detector': {'threshold': .5}, 'prompt_policy': 'direct_anchor'}}
    before = copy.deepcopy(plan); config = bind_scoring(plan, tmp_path)
    assert plan == before
    assert config['cohort_candidates'] == 2 and config['expected_constructed'] == config['expected_review_qualified'] == 1
    assert config['input_review_exclusions'] == {}
    assert config['detector'] == before['scoring_template']['detector']
    assert config['prompt_policy'] == 'direct_anchor'
    (tmp_path/'index.jsonl').write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError, match='registered candidate population'):
        bind_scoring(plan, tmp_path)
