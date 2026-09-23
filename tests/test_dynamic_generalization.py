import json
from pathlib import Path

import numpy as np
import pytest

from scripts.counterfactual.evaluate_dynamic_generalization import (
    compute_statistics, normalize_prompt, rectangular_preprocess, window_indices,
)


def configuration():
    return json.loads((Path(__file__).parents[1] / 'configs/dynamic-generalization/forcing128-v1.json').read_text())


def test_sampling_is_time_preserving_and_independent_of_scores():
    windows = window_indices(489, 16)
    assert windows['start'] == list(range(0, 32, 2))
    assert windows['end'][-1] == 488
    assert all(len(v) == 16 and np.all(np.diff(v) == 2) for v in windows.values())
    assert windows['middle'][0] == 228
    with pytest.raises(ValueError, match='exact 8fps'):
        window_indices(300, 30)
    with pytest.raises(ValueError, match='three distinct'):
        window_indices(16, 8)


def test_normalized_prompt_overlap():
    assert normalize_prompt('  A DOG  runs. ') == normalize_prompt('a dog runs')


def test_preprocessing_square_exact_and_rectangular_whole_frame():
    torch = pytest.importorskip('torch')
    from vbench_audit_models.vjepa import FrozenVJEPA
    config = {'input': {'frames': 16, 'size': 24, 'mean': [.485, .456, .406], 'std': [.229, .224, .225]}}
    adapter = object.__new__(FrozenVJEPA); adapter.config = config; adapter.device = 'cpu'
    rng = np.random.default_rng(2)
    square = rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    assert torch.equal(rectangular_preprocess(square, config, 'cpu'), adapter.preprocess(square))
    rectangle = rng.integers(0, 256, (16, 32, 64, 3), dtype=np.uint8)
    assert rectangular_preprocess(rectangle, config, 'cpu').shape == (1, 3, 16, 24, 24)


def test_mae_does_not_cancel_and_origin_saturation_not_a_pass():
    config = configuration(); config['analysis']['bootstrap_replicates'] = 100
    pairs = [dict(prompt_id='a', origin=[[1, 1, 1]] * 3, repair=[[.5, .4, .6]] * 3),
             dict(prompt_id='b', origin=[[1, 1, 1]] * 3, repair=[[.2, .3, .1]] * 3)]
    result = compute_statistics(pairs, config)
    assert result['repair']['delta'] == pytest.approx(0)
    assert result['repair']['mae'] == pytest.approx(.1)
    assert result['origin']['base_window_one_fraction'] == 1
    assert not result['relative_increase_check']['applicable']
    assert result['relative_increase_check']['pass'] is None


def test_protocol_is_frozen_and_does_not_claim_full_long_video_scores():
    c = configuration()
    assert c['scope']['training_updates'] == 0
    assert c['analysis']['windowed_origin']
    assert not c['analysis']['whole_long_video_origin_score']
    assert c['intervention']['amplitude_pixels'] == 8
    assert c['head_sha256'] == '6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53'
