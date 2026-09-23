"""Independent media, membership, frozen-model and arithmetic audit for BMC."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit(root, config_path, review_path):
    root = Path(root); selection = root / 'selection-v5'
    config = json.loads(Path(config_path).read_text()); review = json.loads(Path(review_path).read_text())
    freeze = json.loads((selection / 'selection.json').read_text())
    prepared = json.loads((selection / 'preparation.json').read_text())
    assert sha(config_path) == prepared['config_sha256'] == freeze['config_sha256']
    assert sha(review_path) == freeze['review_sha256'] and not review['scores_read'] and not review['official_ground_truth']
    assert sha(selection / 'sources.jsonl') == freeze['sources_sha256']
    assert sha(selection / 'preparation.json') == freeze['preparation_sha256']
    for name, key in [('candidates.jsonl', 'candidates_sha256'), ('originals.json', 'originals_sha256'), ('archive_files.json', 'archives_sha256')]:
        assert sha(selection / name) == prepared[key]
    sources = rows(selection / 'sources.jsonl'); candidates = rows(selection / 'candidates.jsonl')
    assert len(sources) == len(candidates) == len({r['source_id'] for r in sources})
    assert {r['source_id'] for r in sources} == set(review['clips'])
    files_checked = 0
    for item in json.loads((selection / 'archive_files.json').read_text()):
        assert sha(root / 'download' / (item['sequence'] + '.zip')) == item['archive_sha256']
        for record in item['files']:
            path = selection / 'extracted' / item['sequence'] / record['path']
            assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
            files_checked += 1
    native_frames = 0; checked_sampled = 0
    for original in json.loads((selection / 'originals.json').read_text()):
        assert sha(original['source_video']) == original['source_sha256']
        if original['status'] == 'NOT SCORED':
            if 'partial_timeline_sha256' in original:
                log = selection / 'timelines' / (original['sequence'] + '.log')
                assert sha(log) == original['partial_timeline_sha256']
                content = log.read_text(); tb = re.search(r'config in time_base: (\d+)/(\d+)', content)
                times = [int(t) * int(tb[1]) / int(tb[2]) for t in re.findall(r'\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:', content)]
                assert any(abs(t - times[0] - i / original['source_fps']) > config['sampling']['timestamp_tolerance'] for i, t in enumerate(times))
            else:
                assert original['source_fps'] < 8
            assert not any(s['sequence'] == original['sequence'] for s in sources)
            continue
        chosen = [s for s in sources if s['sequence'] == original['sequence']]; needed = {}; last = -1
        fps = original['source_fps']; count = original['source_frames']; n = min(12, int(count / fps // 2))
        timeline_file = selection / 'timelines' / (original['sequence'] + '.log')
        assert sha(timeline_file) == original['timeline_log_sha256']
        text = timeline_file.read_text(); tb = re.search(r'config in time_base: (\d+)/(\d+)', text)
        timebase = int(tb[1]) / int(tb[2])
        timestamps = [(int(i), int(t) * timebase) for i, t in re.findall(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:', text)]
        assert len(timestamps) == count and [i for i, t in timestamps] == list(range(count))
        assert max(abs(t - timestamps[0][1] - i / fps) for i, t in timestamps) <= config['sampling']['timestamp_tolerance']
        starts = np.linspace(0, count / fps - 2, n) if n > 1 else [0.]
        assert len(chosen) == n
        for source, start in zip(chosen, starts):
            assert source['condition'] == review['clips'][source['source_id']]['condition']
            indices = np.floor((start + np.arange(16) / 8) * fps + .5).astype(int).tolist()
            assert indices == source['windows']['clip'] and indices[0] > last; last = indices[-1]
            for i, path, expected in zip(indices, source['source_frames'], source['source_frame_sha256']):
                assert sha(path) == expected
                assert i not in needed; needed[i] = path
        cap = cv2.VideoCapture(original['source_video']); i = 0; max_pts_error = 0.; seen = 0
        assert cap.get(cv2.CAP_PROP_FPS) == fps
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            max_pts_error = max(max_pts_error, abs(pts - original['pts_origin_seconds'] - i / fps))
            if i in needed:
                assert np.array_equal(bgr, cv2.imread(needed[i], cv2.IMREAD_COLOR)); seen += 1
            i += 1
        cap.release()
        assert i == count and seen == len(needed)
        # OpenCV reports zero PTS for buffered AVI tail frames; presentation
        # cadence was independently checked from the saved FFmpeg timeline above.
        native_frames += i; checked_sampled += seen
    inputs = []; scores = []; gpu = []
    for path in sorted((root / 'construction').glob('shard-*/inputs.jsonl')):
        done = json.loads((path.parent / 'completion.json').read_text())
        assert done['status'] == 'finished' and not done['failed'] and sha(path) == done['manifest_sha256']
        assert done['config_sha256'] == sha(config_path) and done['selection_sha256'] == sha(selection / 'selection.json')
        inputs.extend(rows(path))
    for path in sorted((root / 'scores').glob('shard-*/scores.jsonl')):
        done = json.loads((path.parent / 'completion.json').read_text()); prov = json.loads((path.parent / 'provenance.json').read_text())
        assert done['status'] == 'finished' and not done['failed'] and sha(path) == done['scores_sha256']
        assert prov['head_sha256'] == config['head_sha256'] and prov['training_updates'] == 0
        assert not prov['posthoc_mapping_changed'] and prov['square_preprocessing_exact'] and prov['training_feature_exact']
        assert prov['head_reload_error'] <= 1e-6
        shard = rows(path); assert {r['evaluation_id'] for r in shard} == set(prov['expected_ids'])
        assert shard[0]['official_infer_parity']
        scores.extend(shard); gpu.append({'gpu': prov['gpu'], 'wall_seconds': done['wall_seconds']})
    expected_ids = {f"{s['source_id']}:clip:{v}" for s in sources for v in config['views']}
    assert len(inputs) == len(scores) == len(expected_ids)
    assert {r['evaluation_id'] for r in inputs} == {r['evaluation_id'] for r in scores} == expected_ids
    predictions = {r['evaluation_id']: r for r in scores}
    for item in inputs:
        pred = predictions[item['evaluation_id']]
        assert sha(item['video']) == item['sha256'] == pred['input_sha256']
        assert item['status'] == pred['status'] == 'ok'
        assert item['decoded_pixels_sha256'] == pred['decoded_pixels_sha256']
        cap = cv2.VideoCapture(item['video']); frames = []
        assert cap.get(cv2.CAP_PROP_FPS) == 8
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release(); assert len(frames) == 16
        assert hashlib.sha256(np.stack(frames).tobytes()).hexdigest() == item['decoded_pixels_sha256']
        sampling, official = pred['diagnostics']['sampling'], pred['diagnostics']['official']
        assert sampling['sampled_frame_count'] == 16 and sampling['source_fps'] == 8 and sampling['sampling_interval'] == 1
        threshold = 6 * min(sampling['frame_shape']) / 256
        assert official['official_threshold'] == threshold and len(official['raw_flow_top5_mean']) == 15
        assert pred['origin'] == float(sum(x > threshold for x in official['raw_flow_top5_mean']) >= 4)
        assert abs(pred['repair'] - 1 / (1 + np.exp(-pred['repair_latent']))) < 1e-12
    for source in sources:
        a = predictions[source['source_id'] + ':clip:original']; b = predictions[source['source_id'] + ':clip:encoding_control']
        assert all(a[k] == b[k] for k in ('decoded_pixels_sha256', 'feature_pixels_sha256', 'origin', 'repair'))
    summary = json.loads((root / 'analysis/summary.json').read_text()); pairs = rows(root / 'analysis/pairs.jsonl')
    assert sha(root / 'analysis/pairs.jsonl') == summary['pairs_sha256']
    assert len(pairs) == len(sources) and {p['source_id'] for p in pairs} == {s['source_id'] for s in sources}
    error_max = 0.
    def equal(a, b):
        nonlocal error_max
        error = float(np.max(abs(np.asarray(a) - np.asarray(b)))); error_max = max(error_max, error)
        assert error < 1e-12
    for pair in pairs:
        assert pair['condition'] == review['clips'][pair['source_id']]['condition']
        for backend in ('origin', 'repair'):
            equal(pair[backend], [predictions[pair['source_id'] + ':clip:' + v][backend] for v in ('original', 'jitter1701', 'jitter2904')])
    sequences = sorted({p['sequence'] for p in pairs}); a = config['analysis']
    draws = np.random.default_rng(a['bootstrap_seed']).integers(0, len(sequences), (a['bootstrap_replicates'], len(sequences)))
    for group in ('all', 'static', 'motion', 'ambiguous'):
        chosen = [p for p in pairs if group == 'all' or p['condition'] == group]; result = summary['statistics'][group]
        assert len(chosen) == result['n']
        if not chosen:
            continue
        deltas = {}
        def interval(vector):
            aggregates = np.array([[sum(p['sequence'] == s for p in chosen),
                sum(v for v, p in zip(vector, chosen) if p['sequence'] == s)] for s in sequences])
            sampled = aggregates[draws].sum(1); valid = sampled[:, 0] > 0
            return np.quantile(sampled[valid, 1] / sampled[valid, 0], [.025, .975])
        for backend in ('origin', 'repair'):
            v = np.array([p[backend] for p in chosen]); d = v[:, 1:] - v[:, 0, None]; delta = d.mean(1); mae = abs(d).mean(1)
            equal([v[:, 0].mean(), v[:, 1:].mean(), delta.mean(), mae.mean()], [result[backend][k] for k in ('base', 'cf', 'delta', 'mae')])
            equal(interval(delta), result[backend]['delta_ci95']); equal(interval(mae), result[backend]['mae_ci95'])
            deltas[backend] = delta
        margin = deltas['repair'] - .1 * deltas['origin']
        equal(margin.mean(), result['ten_percent_margin']['mean']); equal(interval(margin), result['ten_percent_margin']['ci95'])
    for backend in ('origin', 'repair'):
        for j, name in enumerate(('base', 'jitter1701', 'jitter2904')):
            s = [p[backend][j] for p in pairs if p['condition'] == 'static']; m = [p[backend][j] for p in pairs if p['condition'] == 'motion']
            if not s or not m:
                assert summary['discrimination'][backend][name + '_auroc'] is None
            else:
                auc = sum(float(y > x) + .5 * float(y == x) for x in s for y in m) / (len(s) * len(m))
                equal(auc, summary['discrimination'][backend][name + '_auroc'])
    return {'status': 'PASS', 'source_recordings': len(sequences), 'original_files_checked': files_checked,
        'original_frames_timing_rechecked': native_frames, 'sampled_frames_pixel_rechecked': checked_sampled,
        'scored_inputs_decoded': len(inputs), 'clips': len(sources), 'class_counts': dict(Counter(r['condition'] for r in sources)),
        'origin_booleans_recomputed': len(scores), 'repair_sigmoids_recomputed': len(scores),
        'encoding_controls_exact': len(sources), 'maximum_arithmetic_error': error_max,
        'summary_sha256': sha(root / 'analysis/summary.json'), 'auditor_sha256': sha(Path(__file__)), 'gpu_shards': gpu}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'config', 'review', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args(); cv2.setNumThreads(1)
    result = audit(args.root, args.config, args.review)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
