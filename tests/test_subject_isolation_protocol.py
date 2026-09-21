import copy
import json

import pytest

from scripts.counterfactual.common import ROOT, sha256_file
from scripts.counterfactual.diagnose_subject_isolation import validate_protocol
from scripts.counterfactual.subject_artifacts import write_json, write_jsonl


def frozen_inputs(tmp_path):
    dataset, reference = tmp_path / 'dataset', tmp_path / 'reference'
    cohort = [{'base_id': 'b1', 'video_uid': 'v1', 'source_prompt_id': 'new person prompt'}]
    write_jsonl(dataset / 'index.jsonl', cohort)
    protocol = json.loads((ROOT / 'configs/subject-repair/isolation_development_protocol.json').read_text())
    protocol.update(cohort=cohort, sample_size=1, development_prompt_ids=['old prompt'],
                    pinned_inputs={'dataset_index_sha256': sha256_file(dataset / 'index.jsonl'),
                                   'localizer_sha256': 'human_prompts_hash'})
    run = dict(protocol['pinned_inputs'])
    write_json(reference / 'run.json', run)
    return dataset, reference, protocol, run


def test_valid_frozen_inputs_and_reference(tmp_path):
    dataset, reference, protocol, run = frozen_inputs(tmp_path)
    assert validate_protocol(dataset, reference, protocol) == run


@pytest.mark.parametrize('mutation,reason', [
    ('reference_dataset', 'different dataset'),
    ('protocol_dataset', 'preregistered cohort'),
    ('localizer', 'different human'),
    ('cohort', 'preregistered identities'),
    ('overlap', 'overlaps scored development'),
    ('parameter', 'representation settings'),
])
def test_reject_drift_and_prompt_leakage(tmp_path, mutation, reason):
    dataset, reference, protocol, run = frozen_inputs(tmp_path)
    protocol = copy.deepcopy(protocol)
    if mutation == 'reference_dataset':
        run['dataset_index_sha256'] = 'other_dataset'
    elif mutation == 'protocol_dataset':
        protocol['pinned_inputs']['dataset_index_sha256'] = 'other_dataset'
    elif mutation == 'localizer':
        run['localizer_sha256'] = 'other_prompts'
    elif mutation == 'cohort':
        protocol['cohort'][0]['video_uid'] = 'other_video'
    elif mutation == 'overlap':
        protocol['development_prompt_ids'] = ['new person prompt']
    else:
        protocol['fixed_parameters']['crop_margin'] = .2
    write_json(reference / 'run.json', run)
    with pytest.raises(ValueError, match=reason):
        validate_protocol(dataset, reference, protocol)
