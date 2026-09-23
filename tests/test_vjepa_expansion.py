import json
from pathlib import Path

import pytest

from scripts.counterfactual.expand_vjepa_validation import expand_sources, load_config, validate_combinations


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / 'configs/dynamic-static-jitter/vjepa-expansion450-v1.json'


def fixture_sources():
    natural = [dict(video_uid=f'{p}-{g}-{r}', prompt_id=f'p{p}', split='test',
                    relative_video_path=f'{p}-{g}-{r}.mp4')
               for p in range(30) for g in range(3) for r in range(5)]
    previous = [r for r in natural if r['video_uid'].endswith('-0')]
    return natural, previous


def test_all_existing_natural_sources_selected_without_scores():
    natural, previous = fixture_sources()
    counts = load_config(CONFIG)['extension']['cohorts']
    full, additional = expand_sources(natural, previous, counts)
    assert (len(full), len(additional)) == (450, 360)
    assert set(r['video_uid'] for r in additional).isdisjoint(r['video_uid'] for r in previous)
    assert expand_sources(natural[::-1], previous[::-1], counts) == (full, additional)


def test_duplicate_or_out_of_scope_sources_fail_closed():
    natural, previous = fixture_sources()
    counts = load_config(CONFIG)['extension']['cohorts']
    with pytest.raises(ValueError, match='coverage|duplicate'):
        expand_sources(natural + [natural[0]], previous, counts)
    natural[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='TEST MP4'):
        expand_sources(natural, previous, counts)


def test_all_quality_flags_retained_and_exact_combinations_required():
    rows = [dict(base_id='u', family=f, seed=s, status='rejected') for f, s in
            [('original', 0), ('encoding_control', 0), ('local_texture_alternating', 1701),
             ('local_texture_alternating', 2904)]]
    validate_combinations(rows, {'u'})
    with pytest.raises(ValueError, match='missing, extra or duplicate'):
        validate_combinations(rows[:-1] + [rows[0]], {'u'})


def test_frozen_model_mapping_and_analysis_are_not_changed():
    parent = json.loads((ROOT / 'configs/dynamic-static-jitter/vjepa-validation-v1.json').read_text())
    expanded = load_config(CONFIG)
    for key in ('heads', 'implementation_sha256', 'score', 'analysis', 'raft_sha256', 'origin_upstream_commit'):
        assert expanded[key] == parent[key]
    for key, value in parent['construction'].items():
        if key not in ('selection_rule', 'analysis_intent'):
            assert expanded['construction'][key] == value
    assert expanded['extension']['scope']['training_updates'] == 0
    assert not expanded['extension']['scope']['mapping_changes']
