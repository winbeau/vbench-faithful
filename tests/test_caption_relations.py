import json
from pathlib import Path

import pytest

from vbench_audit_models.caption_foreground import caption_head
from vbench_audit_models.caption_relations import normalize_caption_relations, select_caption_box_with_relations


POLICY = json.loads((Path(__file__).parents[1]/'configs/background-repair/scoring_coco80_caption_dev_v2.json').read_text())['caption_fallback']


@pytest.mark.parametrize('text,kind,label', [
    ('white curtain covering window', 'object', 'curtain'),
    ('a curtain covers the window', 'object', 'curtain'),
    ('window covered by a curtain', 'scene', 'windowpane'),
    ('a covered bed', 'object', 'bed'),
    ('a bed covered with a white blanket', 'object', 'bed'),
    ('glass shower door covered by a curtain', 'scene', 'door'),
    ('a chest of drawers containing clothes', 'object', 'chest of drawers'),
    ('the top of a tower covering the building', 'unknown', None),
])
def test_relation_objects_do_not_replace_the_described_head(text, kind, label):
    head = caption_head(normalize_caption_relations(text, POLICY), POLICY)
    assert (head['kind'], head['label']) == (kind, label)


def test_probe_retains_original_description_and_scene_exclusion():
    inputs = [{'text': 'window covered by a curtain', 'box': [0, 0, 12, 18], 'score': .9},
              {'text': 'white curtain covering window', 'box': [0, 2, 9, 20], 'score': .7}]
    box, trace = select_caption_box_with_relations(inputs, (20, 20), POLICY)
    assert box == [0, 2, 9, 20] and trace['selected_index'] == 1
    assert trace['captions'][1]['original_text'] == inputs[1]['text']
    assert trace['captions'][0]['head']['kind'] == 'scene'
