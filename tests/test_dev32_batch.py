import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dynamic_degree.candidate_boolean import fixed_grid_decision
from scripts.counterfactual.select_mp4_dev32 import select_mp4
from scripts.counterfactual.score_mp4_dev32 import evaluate_candidate, grid_queries

ROOT = Path(__file__).resolve().parents[1]


def test_selection_keeps_24_and_excludes_holdout_without_scores():
    original = [json.loads(s) for s in (ROOT / 'configs/dynamic-static-jitter/sources.v1.jsonl').read_text().splitlines()]
    with (ROOT / 'data/processed/e0_scoring_manifest.csv').open() as f:
        pool = list(csv.DictReader(f))
    selected, replacements = select_mp4(original, pool)
    assert len(selected) == 32 and len(replacements) == 8
    assert sum(s['selection_role'] == 'retained_original_dev_mp4' for s in selected) == 24
    assert not {s['video_uid'] for s in selected} & {s['video_uid'] for s in original if s['split'] == 'test'}
    assert all(s['split'] == 'dev' and s['relative_video_path'].endswith('.mp4') for s in selected)
    assert select_mp4(original, pool[::-1])[0] == selected
    declared = [json.loads(s) for s in (ROOT / 'configs/dynamic-static-jitter/sources.local-texture-dev32-mp4-v1.jsonl').read_text().splitlines()]
    assert selected == declared


def test_boolean_uses_official_strict_threshold_and_time_count():
    d = np.zeros((15, 144, 2))
    d[:4, :, 0] = 6
    assert fixed_grid_decision(d, (256, 256))['score'] == 0
    d[:3, :, 0] = 7
    assert fixed_grid_decision(d, (256, 256))['score'] == 0
    d[3, :, 0] = 7
    result = fixed_grid_decision(d, (256, 256))
    assert result['score'] == 1 and result['top_points'] == 7 and result['count_num'] == 4
    assert fixed_grid_decision(d * 2, (512, 512))['score'] == 1


def test_fixed_denominator_keeps_all_predictions_and_rejects_missing():
    d = np.zeros((15, 144, 2))
    d[:, :7, 0] = 8
    assert fixed_grid_decision(d, (256, 256))['score'] == 1
    d[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fixed_grid_decision(d, (256, 256))


class FakeTracker:
    def __init__(self, positions):
        self.positions = np.asarray(positions)
        self.calls = []

    def track_queries(self, frames, query):
        t = int(query[0, 0])
        self.calls.append(t)
        tracks = query[:, 1:][None] + (self.positions - self.positions[t])[:, None, :]
        return {'tracks': tracks.astype(np.float32), 'visible': np.ones(tracks.shape[:2], bool)}


class FakeSAM:
    def propose(self, frame):
        from vbench_audit_models.sam_regions import pack_proposals
        h, w = frame.shape[:2]
        return pack_proposals([{'segmentation': np.ones((h, w), bool), 'area': h*w,
                               'predicted_iou': 1, 'stability_score': 1, 'bbox': [0, 0, w, h],
                               'point_coords': [[w/2, h/2]]}], (h, w))


@pytest.mark.parametrize('oscillates', [False, True])
def test_video_only_candidate_preserves_translation_and_periodic_affine_motion(oscillates):
    frames = np.zeros((16, 256, 256, 3), np.uint8)
    times = np.arange(16) / 8
    x = np.sin(np.arange(16) * 2.3) * 12 if oscillates else np.arange(16) * 8
    tracker = FakeTracker(np.c_[x, np.zeros(16)])
    result, evidence = evaluate_candidate(frames, times, tracker, FakeSAM(), score_kind='legacy_boolean')
    assert tracker.calls == list(range(16))
    assert result['score'] == result['raw_ablation']['score'] == 1
    assert result['diagnostic']['score'] is None  # old diagnostic stays a diagnostic
    np.testing.assert_allclose(evidence['removal'], 0, atol=1e-8)
    np.testing.assert_allclose(grid_queries(256, 256, 5)[:, 1:], evidence['xy'])


def test_unknown_cadence_is_not_silently_aliased():
    with pytest.raises(ValueError, match='8 FPS'):
        evaluate_candidate(np.zeros((16, 32, 32, 3), np.uint8), np.arange(16) / 16, None, None)


def batch_rows():
    return [{'base_id': str(i), 'prompt_id': str(i // 4),
             **{b: {'base': 0, 'cf': [0, 0]} for b in ('origin', 'raw_ablation', 'repair')}} for i in range(32)]


def test_batch_uses_signed_change_and_within_source_seed_mean():
    from scripts.counterfactual.summarize_mp4_dev32 import batch_statistics
    rows = batch_rows()
    for row in rows:
        row['origin']['cf'] = [1, 0]
    rows[0]['repair']['cf'] = [1, 1]
    rows[1]['repair'] = {'base': 1, 'cf': [0, 0]}
    result = batch_statistics(rows)
    assert result['origin']['delta'] == .5
    assert result['repair']['delta'] == 0  # MAE/positive-only average would not be zero
    assert result['repair']['base_mean'] == result['repair']['cf_mean'] == 1/32
    assert result['numerical_criterion']['passes_batch_point_estimate']


def test_batch_does_not_drop_missing_or_duplicate_sources():
    from scripts.counterfactual.summarize_mp4_dev32 import batch_statistics
    rows = batch_rows()
    with pytest.raises(ValueError):
        batch_statistics(rows[:-1])
    rows[-1]['base_id'] = rows[0]['base_id']
    with pytest.raises(ValueError):
        batch_statistics(rows)
    rows = batch_rows()
    rows[0]['repair']['cf'][0] = None
    with pytest.raises(ValueError):
        batch_statistics(rows)


def test_no_origin_increase_does_not_become_ratio_success():
    from scripts.counterfactual.summarize_mp4_dev32 import batch_statistics
    result = batch_statistics(batch_rows())
    assert result['numerical_criterion']['passes_batch_point_estimate'] is None
