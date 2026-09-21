import json

import numpy as np
import pytest

from scripts.counterfactual.build_region_discrimination import build as build_parent, verify as verify_parent
from scripts.counterfactual.build_subject_frame_probe import POSITIONS, build, verify
from scripts.counterfactual.common import sha256_file
from scripts.counterfactual.generate_subject_masks import generate
from scripts.counterfactual.subject_artifacts import new_output, read_jsonl, read_png_sequence, write_json


def test_single_frame_composition_preserves_other_frames_and_rejects_changed_artifacts(tmp_path):
    frames = np.full((8, 48, 48, 3), 20, np.uint8)
    masks = np.zeros((8, 48, 48), np.uint8)
    masks[:, 12:36, 12:36] = 1
    frames[masks > 0] = [220, 40, 80]

    class Localizer:
        id2label = {0: 'sky', 1: 'person'}
        provenance = {'role': 'construction', 'family': 'segformer', 'test_fixture': True}
        def labels_for(self, frame):
            return masks[0]

    inputs = tmp_path / 'input'; inputs.mkdir()
    (inputs / 'video.bin').write_bytes(frames.tobytes())
    base = {'base_id': 'b0', 'video_uid': 'v0', 'subject_en': 'person', 'prompt_en': 'a person',
            'relative_video_path': 'video.bin'}
    source = new_output(tmp_path / 'construction')
    generate([base, {**base, 'base_id': 'b1', 'video_uid': 'v1', 'subject_en': 'cat'}], inputs, source,
             {'classes': {'person': ['person']}, 'unoccupied_stuff_labels': ['sky'], 'absence_scope': 'synthetic fixture'},
             Localizer(), decoder=lambda _: frames)
    parent = new_output(tmp_path / 'parent'); build_parent(source, parent)
    receipt = {**verify_parent(parent), 'index_sha256': sha256_file(parent / 'index.jsonl')}
    write_json(parent / 'integrity.json', receipt)
    protocol = tmp_path / 'protocol.json'
    config = {'parent_index_sha256': receipt['index_sha256'], 'parent_integrity_sha256': sha256_file(parent / 'integrity.json'),
              'parent_verified_frames': 64, 'cohort_candidates': 2, 'expected_constructed': 1, 'positions': list(POSITIONS)}
    write_json(protocol, config)
    output = new_output(tmp_path / 'output'); summary = build(parent, output, protocol)
    assert summary['accepted'] == 1 and summary['rejected'] == 1
    assert verify(parent, output)['verified_corrupted_frames'] == 64
    row = next(r for r in read_jsonl(output / 'index.jsonl') if r['status'] == 'accepted')
    manifest = json.loads((output / row['manifest']).read_text())
    original = json.loads((parent / row['manifest']).read_text())
    clean = read_png_sequence(parent, original['variants']['clean'])
    for position, index in [('first_frame', 0), ('middle_frame', 3), ('last_frame', 7)]:
        assert manifest['positions'][position]['parameters']['window_indices'] == [index]
        for level in ('background_corrupt', 'subject_corrupt'):
            full = read_png_sequence(parent, original['variants']['full/' + level])
            actual = read_png_sequence(output, manifest['variants'][position + '/' + level])
            expected = clean.copy(); expected[index] = full[index]
            assert np.array_equal(actual, expected)
    assert manifest['construction_masks']['sha256'] == original['construction_masks']['sha256']
    corrupted = output / manifest['variants']['first_frame/background_corrupt'][0]['path']
    corrupted.write_bytes(b'not the registered pixels')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        verify(parent, output)
    write_json(protocol, {**config, 'parent_integrity_sha256': 'wrong'})
    with pytest.raises(ValueError, match='parent replay receipt changed'):
        build(parent, new_output(tmp_path / 'bad'), protocol)
