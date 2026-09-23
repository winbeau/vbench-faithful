import numpy as np
import pytest

from scripts.counterfactual.evaluate_dynamic_lasiesta import auc, mask_state, state_windows


def test_official_static_object_is_not_moving_foreground():
    rgb = np.zeros((8, 8, 3), np.uint8)
    assert mask_state(rgb)['state'] == 'static'
    rgb[0, 0] = 255
    rgb[1, 0] = 128
    assert mask_state(rgb)['state'] == 'static'
    assert mask_state(rgb)['stationary_pixels'] == 1
    rgb[0, 0] = (255, 0, 0)
    assert mask_state(rgb)['state'] == 'motion'
    assert mask_state(rgb)['moving_pixels'] == 1


def test_unknown_only_cannot_be_called_static():
    rgb = np.zeros((8, 8, 3), np.uint8); rgb[0, 0] = 128
    assert mask_state(rgb)['state'] == 'ambiguous'
    rgb[1, 1] = (12, 13, 14)
    with pytest.raises(ValueError, match='unknown annotation colors'):
        mask_state(rgb)


def test_windows_use_all_intermediate_labels_and_do_not_overlap():
    states = ['static'] * 96 + ['motion'] * 70
    windows, runs = state_windows(states)
    assert [w['start'] for w in windows] == [0, 48, 96]
    assert windows[0]['indices'] == list(range(0, 46, 3))
    assert all(len(w['indices']) == 16 for w in windows)
    assert all(a['stop'] <= b['start'] for a, b in zip(windows, windows[1:]))
    states[17] = 'motion'  # this frame would be missed by 3-frame sampling
    windows, runs = state_windows(states)
    assert 0 not in [w['start'] for w in windows]
    assert any(r['reason'] == 'shorter_than_span' for r in runs)
    with pytest.raises(ValueError, match='overlap'):
        state_windows(states, step=16)


def test_auc_is_threshold_free_and_counts_binary_ties():
    assert auc([0, 0], [1, 1]) == 1
    assert auc([0, 1], [1, 1]) == .75
    assert auc([1, 1], [1, 1]) == .5
    assert auc([.3, .4], [.1, .2]) == 0
    assert auc([], [1]) is None
