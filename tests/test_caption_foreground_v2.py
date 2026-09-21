import json
from pathlib import Path

from vbench_audit_models.caption_foreground import caption_head
from vbench_audit_models.caption_foreground_v2 import normalize_caption, select_caption_box_v2


def test_viewpoint_is_not_a_physical_part_or_context():
    policy = json.loads((Path(__file__).parents[1]/'configs/background-repair/scoring_coco80_caption_dev_v2.json').read_text())['caption_fallback']
    for text in ['the back of a white lighthouse', 'a blurry image of a boat', 'the side of a castle']:
        normalized = normalize_caption(text, policy)
        assert caption_head(normalized, policy)['kind'] == 'object'
    for text in ['window on a lighthouse', 'glass shower door', 'the wall behind a castle']:
        assert caption_head(normalize_caption(text, policy), policy)['kind'] == 'scene'
    assert caption_head(normalize_caption('the top of a lighthouse', policy), policy)['kind'] == 'unknown'


def test_existing_grit_proposals_do_not_get_a_second_confidence_cut():
    policy = json.loads((Path(__file__).parents[1]/'configs/background-repair/scoring_coco80_caption_dev_v2.json').read_text())['caption_fallback']
    box, trace = select_caption_box_v2([{'text':'a castle by the shore','score':.464,'box':[2,2,10,15]}], (20,20), policy)
    assert box == [2,2,10,15]
    assert trace['captions'][0]['original_text'] == 'a castle by the shore'
