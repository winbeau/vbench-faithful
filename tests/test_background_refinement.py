import numpy as np
import json
import pytest

from scripts.counterfactual.refine_background_construction import choose_target, eligibility


def test_persistent_foreground_wins_over_single_large_false_positive():
    labels = np.zeros((5, 20, 20), dtype=np.uint8)
    labels[:, :4] = 1
    labels[0, 4:18] = 2
    target, target_id, _ = choose_target(labels, {0: 'sky', 1: 'person', 2: 'car'}, ['car', 'person'])
    assert (target, target_id) == ('person', 1)


def test_zero_median_tie_cannot_select_an_unobserved_class():
    labels = np.zeros((5, 20, 20), dtype=np.uint8)
    labels[1:3, :8] = 2
    names = {0: 'sky', 1: 'airplane', 2: 'person'}
    target, target_id, _ = choose_target(labels, names, ['airplane', 'person'])
    assert (target, target_id) == ('person', 2)
    # Archived v2 output remains reproducible, including the diagnosed bug.
    legacy, _, _ = choose_target(labels, names, ['airplane', 'person'], rule='legacy_median_lexical_v2')
    assert legacy == 'airplane'


def test_empty_semantic_evidence_is_explicitly_no_target_not_airplane():
    from scripts.counterfactual.refine_background_construction import refine_selected
    labels = np.zeros((4, 20, 20), dtype=np.uint8)
    target, target_id, _ = choose_target(labels, {0: 'sky', 1: 'airplane'}, ['airplane'])
    assert target is None and target_id is None
    masks, failures = refine_selected(np.zeros((4, 20, 20, 3), np.uint8), labels, target, target_id)
    assert not masks.any() and failures == []


def test_histogram_area_calculation_preserves_pixel_fractions():
    labels = np.random.default_rng(91).integers(0, 4, (7, 11, 13), dtype=np.uint8)
    names = {0: 'sky', 1: 'person', 2: 'car', 3: 'bed'}
    _, _, fractions = choose_target(labels, names, ['person', 'car', 'bed'])
    for index, name in names.items():
        if name in fractions:
            np.testing.assert_array_equal(fractions[name], (labels == index).mean((1, 2)))


def test_construction_partition_retains_every_candidate_exactly_once():
    from scripts.counterfactual.refine_background_construction import partition_entries
    population = list(range(680))
    parts = [partition_entries(population, i, 8) for i in range(8)]
    assert all(len(part) == 85 for part in parts)
    assert sorted(x for part in parts for x in part) == population
    with pytest.raises(ValueError, match='invalid construction shard'):
        partition_entries(population, 8, 8)


def test_merge_allows_canonical_json_but_rejects_changed_protocol(tmp_path):
    from scripts.counterfactual.common import sha256_file
    from scripts.counterfactual.merge_background_construction import validate_protocol
    source = tmp_path/'source.json'; source.write_text('{\n  "stage": "dev",\n  "count": 680\n}\n')
    shard = tmp_path/'shard'; shard.mkdir()
    copy = shard/'protocol.json'; copy.write_text('{"count":680,"stage":"dev"}\n')
    assert validate_protocol(source, sha256_file(source), [shard])['count'] == 680
    copy.write_text('{"count":679,"stage":"dev"}\n')
    with pytest.raises(ValueError, match='shard protocol changed'):
        validate_protocol(source, sha256_file(source), [shard])


def test_expanded_vocabulary_can_select_furniture_but_not_scene_surfaces():
    from scripts.counterfactual.common import ROOT
    protocol = json.loads((ROOT/'configs/background-repair/construction_refined_dev_v3.json').read_text())
    names = {int(k): v for k, v in json.loads((ROOT/'configs/subject-repair/ade20k_labels.json').read_text())['id2label'].items()}
    ids = {v: k for k, v in names.items()}
    labels = np.full((4, 20, 20), ids['wall'], np.uint8)
    labels[:, 12:16] = ids['bed']
    target, _, _ = choose_target(labels, names, protocol['foreground_labels'], rule=protocol['target_selection_rule'])
    assert target == 'bed'
    assert len(protocol['foreground_labels']) == protocol['foreground_label_count'] > 90
    assert set(protocol['foreground_labels']).isdisjoint(protocol['excluded_scene_labels'])
    assert set(protocol['foreground_labels']) | set(protocol['excluded_scene_labels']) == set(names.values())
    assert {'sky', 'road', 'water', 'wall', 'floor', 'dirt track', 'land', 'lake'} <= set(protocol['excluded_scene_labels'])


def test_empty_audit_separates_missing_class_selection_from_refinement_failure():
    from scripts.counterfactual.diagnose_background_empty_masks import empty_cause
    assert empty_cause({'airplane': [0, 0], 'person': [0, .2]}, 'airplane') == 'selected_absent_despite_other_foreground_evidence'
    assert empty_cause({'airplane': [0, 0], 'person': [0, .2]}, 'person') == 'selected_semantic_nonempty_refinement_empty'
    assert empty_cause({'airplane': [0, 0], 'person': [0, 0]}, 'airplane') == 'no_detection_in_ten_class_vocabulary'


def test_intervention_cannot_pass_on_large_mean_alone():
    protocol = {'semantic_presence_min_area': .01, 'semantic_presence_fraction_min': .9,
                'mean_foreground_min': .15, 'mean_foreground_max': .25,
                'frame_foreground_min': .01, 'frame_foreground_max': .5}
    semantic = np.array([0, 0, 0, .8])
    refined = np.array([0, 0, 0, .8])
    reasons = eligibility(semantic, refined, [], protocol)
    assert 'no_persistent_semantic_foreground' in reasons
    assert 'frame_foreground_outside_subject_protocol_range' in reasons
    assert eligibility(np.full(4, .2), np.full(4, .2), [], protocol) == []
    assert 'grabcut_frame_failure' in eligibility(np.full(4, .2), np.full(4, .2), [{'frame': 2}], protocol)


def test_verification_rejects_subject_source_even_when_mask_geometry_matches(tmp_path):
    from scripts.counterfactual.common import ROOT, sha256_file
    from scripts.counterfactual.subject_artifacts import write_json, write_npz
    from scripts.counterfactual.verify_background_interventions import verify_one

    protocol = json.loads((ROOT/'configs/background-repair/construction_refined_dev_v2.json').read_text())
    labels_map = {int(k): v for k, v in json.loads((ROOT/'configs/subject-repair/ade20k_labels.json').read_text())['id2label'].items()}
    person = next(k for k, v in labels_map.items() if v == 'person')
    sky = next(k for k, v in labels_map.items() if v == 'sky')
    labels = np.full((4, 20, 20), sky, np.uint8); labels[:, :4] = person
    masks = (labels == person).astype(np.uint8)
    target, _, fractions = choose_target(labels, labels_map, protocol['foreground_labels'])
    mask_path = tmp_path/'mask.npz'
    write_npz(mask_path, masks=masks, semantic_labels=labels, metadata_json=json.dumps({'role': 'construction'}))
    source = {'status': 'accepted', 'base': {'video_uid': 'x', 'dimension': 'subject_consistency'},
              'construction_mask': {'path': 'mask.npz', 'sha256': sha256_file(mask_path)},
              'construction_target': target, 'semantic_class_fractions': fractions,
              'construction_target_source': protocol['target_source']}
    manifest = tmp_path/'manifest.json'; write_json(manifest, source)
    entry = {'manifest': 'manifest.json', 'manifest_sha256': sha256_file(manifest), 'status': 'accepted', 'video_uid': 'x'}
    with pytest.raises(ValueError, match='official dimension'):
        verify_one((tmp_path, tmp_path, entry, protocol, {}))


@pytest.mark.parametrize('subject', ['person', 'car'])
def test_background_subject_blur_reuses_the_exact_subject_operator(subject):
    from scripts.counterfactual.build_background_interventions import foreground_and_background_edits
    from scripts.counterfactual.region_discrimination import region_discrimination

    frames = np.random.default_rng(72).integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    masks = np.zeros(frames.shape[:3], np.uint8); masks[:, 8:24, 10:22] = 1
    expected = region_discrimination(frames, masks, np.zeros_like(masks), subject=subject, position='full')
    foreground, _, _ = foreground_and_background_edits(frames, masks, frames[::-1], sigma=18 if subject == 'person' else 12)
    assert np.array_equal(foreground, expected.frames['subject_corrupt'])
