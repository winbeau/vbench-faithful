import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dynamic_degree.anchored_probe import anchored_losses
from dynamic_degree.learned_probe import MotionProbe
from scripts.counterfactual.train_vjepa_anchored import anchor_views, statistics, acceptance
from scripts.counterfactual.train_vjepa_anchored_chunked import forward_chunks
from scripts.counterfactual.verify_vjepa_anchored import compare


ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / 'configs/dynamic-static-jitter/vjepa-anchored-v1.json').read_text())
PC = json.loads((ROOT / 'configs/dynamic-static-jitter/vjepa-probe-v1.json').read_text())


def test_zero_anchors_and_ordered_motion_supervise_model_gradients():
    q = torch.zeros((2, 8), requires_grad=True)
    pairs = torch.tensor([[0, 1], [0, 0]])
    labels = torch.tensor([1., .5])
    loss, parts = anchored_losses(q, pairs, labels, CONFIG['loss'])
    assert parts['still_zero'] == .25
    loss.backward()
    assert q.grad[:, 7].gt(0).all()  # optimizer lowers static-jitter logits
    assert q.grad[:, 5].lt(0).all()  # optimizer raises stronger-pan logits
    assert 'gauge' not in parts


def test_good_continuous_motion_predictions_have_smaller_loss():
    pairs = torch.tensor([[0, 1], [0, 0]])
    labels = torch.tensor([1., .5])
    target = torch.tensor([[.6, .6, .6, .01, .1, .25, .2, .01],
                           [.4, .4, .4, .01, .1, .25, .2, .01]])
    good, _ = anchored_losses(torch.logit(target), pairs, labels, CONFIG['loss'])
    constant, _ = anchored_losses(torch.zeros_like(target), pairs, labels, CONFIG['loss'])
    assert good.item() < constant.item() / 100
    with pytest.raises(ValueError, match='eight'):
        anchored_losses(torch.zeros(2, 3), pairs, labels, CONFIG['loss'])


def test_anchor_pixels_retain_exact_static_and_coherent_motion():
    frame = np.zeros((64, 64, 3), np.uint8); frame[20:24, 15:19] = 255
    frames = np.repeat(frame[None], 16, axis=0)
    views, _ = anchor_views(frames, 'dev_uid', CONFIG, PC)
    assert set(views) == {'still', 'pan8', 'pan32', 'reversal32', 'still_jitter8'}
    assert np.array_equal(views['still'], frames)
    assert np.array_equal(views['pan8'][-1, :, 8:], frame[:, :-8])
    assert np.array_equal(views['reversal32'][0], frame)
    assert np.array_equal(views['reversal32'][-1], frame)
    assert np.array_equal(views['reversal32'][7, :, 32:], frame[:, :-32])


def test_native_cf_statistics_exclude_synthetic_anchors():
    sources = [dict(video_uid='a'), dict(video_uid='b')]
    pairs = [dict(a='a', b='b', label=1), dict(a='a', b='a', label=.5)]
    s = np.array([[.6, .6, .6, .01, .1, .25, .2, .01], [.4, .4, .4, .01, .1, .25, .2, .01]])
    result = statistics(sources, pairs, np.log(s / (1 - s)))
    assert result['native_jitter_mae'] == 0
    assert result['weak_pan_above_still_fraction'] == 1
    gates = dict(CONFIG['acceptance'], validation_native_ordered_correct_min=1)
    assert acceptance(result, gates)['all_passed']


def test_chunked_execution_preserves_full_batch_outputs_and_gradients():
    torch.manual_seed(9)
    direct = MotionProbe().double()
    chunked = MotionProbe().double()
    chunked.load_state_dict(direct.state_dict())
    features = torch.randn(11, 13, 768, dtype=torch.float64)
    a = direct(features)
    b = forward_chunks(chunked, [features[:3], features[3:7], features[7:]])
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    torch.sigmoid(a).square().mean().backward()
    torch.sigmoid(b).square().mean().backward()
    for left, right in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)


def test_reload_verifier_rejects_nonfinite_mismatch_and_shape_changes():
    assert compare([.1, .2], [.1, .2]) == 0
    with pytest.raises(ValueError, match='disagrees'):
        compare([.1], [.3])
    with pytest.raises(ValueError, match='finite'):
        compare([float('nan')], [.1])
    with pytest.raises(ValueError, match='shape'):
        compare([.1], [[.1]])
