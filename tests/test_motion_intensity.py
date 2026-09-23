import numpy as np
import pytest

from dynamic_degree.motion_intensity import motion_intensity
from scripts.counterfactual.replay_mp4_dev32_intensity import summarize_pairs
from scripts.counterfactual.score_mp4_dev32 import evaluate_candidate


def batch_rows():
    return [{'base_id': str(i), 'prompt_id': str(i // 4),
             **{b: {'base': 0., 'cf': [0., 0.]} for b in ('origin', 'raw_ablation', 'repair')}} for i in range(32)]


def test_continuous_no_binary_jump_at_old_six_pixel_threshold():
    times = np.arange(16) / 8
    values = []
    for speed in (0., .1, 5.9, 6., 6.1, 8., 64.):
        d = np.zeros((15, 144, 2))
        d[..., 0] = speed
        result = motion_intensity(d, times, (256, 256))
        assert result['score'] == pytest.approx(speed * 8 / 256)
        assert result['thresholding'] == 'none' and result['clipping'] == 'none'
        values.append(result['score'])
    assert np.all(np.diff(values) > 0)
    assert values[-1] == 2  # intensity is not a probability and is not clipped


def test_spatial_scale_and_elapsed_time_have_physical_meaning():
    dt = np.array([.1, .2, .1, .3])
    times = np.r_[0, dt.cumsum()]
    d = np.zeros((4, 144, 2))
    d[..., 0] = dt[:, None] * 32  # 32 pixels/sec
    result = motion_intensity(d, times, (256, 256))
    assert result['score'] == pytest.approx(.125)
    assert motion_intensity(d * 2, times, (512, 512))['score'] == pytest.approx(.125)
    assert motion_intensity(d, times * 2, (256, 256))['score'] == pytest.approx(.0625)


def test_all_grid_points_and_exact_top5_population_are_used():
    d = np.zeros((15, 144, 2))
    d[:, :7, 0] = 8
    result = motion_intensity(d, np.arange(16)/8, (256, 256))
    assert result['top_points'] == 7 and result['grid_points'] == 144
    assert result['score'] == .25
    d[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        motion_intensity(d, np.arange(16)/8, (256, 256))


def test_current_video_only_candidate_defaults_to_continuous():
    from types import SimpleNamespace
    frames = np.zeros((16, 256, 256, 3), np.uint8)
    positions = np.c_[np.arange(16)*8, np.zeros(16)]
    def track(frames, query):
        xy = query[:, 1:][None] + (positions-positions[int(query[0, 0])])[:, None, :]
        return {'tracks': xy.astype(np.float32), 'visible': np.ones(xy.shape[:2], bool)}
    tracker = SimpleNamespace(track_queries=track)
    sam = SimpleNamespace(propose=lambda frame: {
        'masks_packed': np.full((1, 256, 32), 255, np.uint8), 'image_shape': np.array([256, 256])})
    result, _ = evaluate_candidate(frames, np.arange(16)/8, tracker, sam)
    assert result['score_kind'] == 'continuous_intensity'
    assert result['score'] == result['raw_ablation']['score']
    assert result['score'] == pytest.approx(.25, rel=1e-6)  # native float32 query-coordinate quantization
    assert result['guarded']['thresholding'] == 'none'


def test_batch_keeps_original_binary_but_repair_continuous():
    rows = batch_rows()
    for i, row in enumerate(rows):
        row['origin'] = {'base': i % 2, 'cf': [1, 1]}
        row['repair'] = {'base': 2.5, 'cf': [2.6, 2.4]}
        row['raw_ablation'] = {'base': 2.5, 'cf': [2.7, 2.5]}
    result = summarize_pairs(rows)
    assert result['origin']['base_mean'] == .5 and result['origin']['cf_mean'] == 1
    assert result['repair']['base_mean'] == result['repair']['cf_mean'] == 2.5
    assert result['repair']['delta'] == 0
    assert result['cross_metric_10_percent_criterion']['passes'] is None
    rows[0]['origin']['base'] = .4
    with pytest.raises(ValueError, match='binary'):
        summarize_pairs(rows)


def test_missing_source_cannot_be_discarded():
    with pytest.raises(ValueError):
        summarize_pairs(batch_rows()[:-1])
