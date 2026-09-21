import pytest

from vbench_audit_models.caption_foreground import caption_head, select_caption_box


POLICY = {'head_stop_regex': r'\b(?:on|in|with|behind|around|of|is|are)\b',
          'object_aliases': {'tower': 'tower', 'glass': 'glass', 'shower': 'shower', 'bathtub': 'bathtub', 'cabinet': 'cabinet', 'chest of drawers': 'chest of drawers'},
          'scene_aliases': {'wall': 'wall', 'window': 'window', 'door': 'door', 'railing': 'railing'},
          'threshold': .5, 'minimum_box_area': .01, 'maximum_box_area': .6}


@pytest.mark.parametrize('text,kind,label', [
    ('window on a tower', 'scene', 'window'), ('glass shower door', 'scene', 'door'),
    ('the tower is white', 'object', 'tower'), ('a white bathtub', 'object', 'bathtub'),
    ('medicine cabinet with mirror', 'object', 'cabinet'), ('top of a tower', 'unknown', None),
    ('the walls are beige', 'scene', 'wall'), ('a chest of drawers', 'object', 'chest of drawers')])
def test_caption_head_does_not_select_context_or_material(text, kind, label):
    result = caption_head(text, POLICY)
    assert (result['kind'], result['label']) == (kind, label)


def test_larger_wall_is_not_selected_and_corner_bathtub_is_allowed():
    records = [{'box': [8, 0, 20, 18], 'score': .9, 'text': 'the wall is tiled'},
               {'box': [0, 12, 15, 20], 'score': .7, 'text': 'a white bathtub'}]
    box, trace = select_caption_box(records, (20, 20), POLICY)
    assert box == [0, 12, 15, 20] and trace['selected_index'] == 1
    with pytest.raises(ValueError, match='native image'):
        select_caption_box([{**records[0], 'box': [0, 0, 40, 40]}], (20, 20), POLICY)


def test_grit_failure_restores_caller_determinism():
    import torch
    import numpy as np
    from types import SimpleNamespace
    from vbench_audit_models.caption_foreground import GritCaptionBoxDetector
    detector = GritCaptionBoxDetector.__new__(GritCaptionBoxDetector)
    detector.policy = POLICY

    def fail(frame):
        assert not torch.are_deterministic_algorithms_enabled()
        raise RuntimeError('probe failure')

    detector.model = SimpleNamespace(detect=fail)
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError, match='probe failure'):
            detector.box_for(np.zeros((20, 20, 3), np.uint8))
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn)
