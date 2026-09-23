import json
from pathlib import Path

import numpy as np
import pytest

from scripts.counterfactual.score_vjepa_anchored450 import numpy_sha, previous_data
from scripts.counterfactual.static_jitter import digest
from scripts.counterfactual.summarize_vjepa_anchored450 import paired_statistics, preference_statistics


def test_in_memory_feature_hash_matches_saved_numpy_cache(tmp_path):
    array = np.arange(48, dtype=np.float16).reshape(8, 6)
    path = tmp_path / 'feature.npy'
    np.save(path, array, allow_pickle=False)
    assert numpy_sha(array) == digest(path)
    assert numpy_sha(array + 1) != digest(path)


def test_new_head_and_old_head_are_distinct_and_no_retuning_allowed():
    root = Path(__file__).parents[1]
    config = json.loads((root / 'configs/dynamic-static-jitter/vjepa-anchored450-v1.json').read_text())
    assert config['head_sha256'] != config['old_head_sha256']
    assert config['coverage']['inputs'] == 450 * 4
    assert config['scope']['previous_development_gate_failed']
    assert config['scope']['no_retuning']
    assert config['inference']['training_updates'] == 0


def test_previous_data_rejects_changed_manifest_before_inference(tmp_path):
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs/inputs.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='identity mismatch'):
        previous_data(tmp_path, {'inputs_sha256': '0' * 64})


def test_three_versions_remain_separate_and_signed_delta_not_absolute():
    config = {'analysis': {'bootstrap_seed': 1, 'bootstrap_replicates': 100}}
    pairs = [dict(prompt_id='a', origin=[0, 1, 1], old_joint=[.5, .5, .5],
                  anchored=[.1, .2, .2], anchored_latent=[-1, -.9, -.9]),
             dict(prompt_id='b', origin=[1, 1, 1], old_joint=[.6, .6, .6],
                  anchored=[.4, .2, .2], anchored_latent=[1, .8, .8])]
    result = paired_statistics(pairs, config)
    assert result['origin']['delta'] == .5
    assert result['old_joint']['delta'] == 0
    assert result['anchored']['delta'] == pytest.approx(-.05)
    assert result['anchored']['mae'] == pytest.approx(.15)
    assert result['relative_scale_invariance_check']['point_estimate_pass']
    assert not result['relative_scale_invariance_check']['overall_goal_complete']


def test_preference_uses_half_credit_only_for_predicted_ties_and_preserves_losses():
    config = {'analysis': {'bootstrap_seed': 1, 'bootstrap_replicates': 100}}
    pairs = [dict(a='a', b='b', label=1, prompt_id='p'), dict(a='a', b='b', label=.5, prompt_id='p')]
    scores = {'a': dict(origin=0, old_joint=.8, anchored=.1), 'b': dict(origin=0, old_joint=.2, anchored=.3)}
    result, credits = preference_statistics(pairs, scores, config)
    assert result['origin']['predicted_ties'] == 1
    assert result['origin']['concordance'] == .5
    assert result['old_joint']['strict_correct'] == 1
    assert result['anchored']['incorrect'] == 1
    assert result['anchored']['human_ties'] == 1
    assert result['anchored_minus_old_joint']['worse_pairs'] == 1
    assert credits['anchored'][0] == 0
