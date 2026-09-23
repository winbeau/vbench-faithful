"""Independently recompute labels, media identities and LASIESTA score statistics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root, config_path):
    root = Path(root); config = json.loads(Path(config_path).read_text())
    selection = json.loads((root / 'selection/selection.json').read_text())
    assert sha(config_path) == selection['config_sha256']
    assert sha(root / 'selection/sources.jsonl') == selection['sources_sha256']
    assert sha(root / 'selection/frame_labels.jsonl') == selection['labels_sha256']
    assert sha(root / 'selection/archive_files.json') == selection['archives_sha256']
    sources = rows(root / 'selection/sources.jsonl'); frame_labels = rows(root / 'selection/frame_labels.jsonl')
    assert len({r['source_id'] for r in sources}) == len(sources)
    files_checked = 0
    for archive in json.loads((root / 'selection/archive_files.json').read_text()):
        seq = archive['sequence']
        assert sha(root / 'download' / (seq + '.rar')) == archive['archive_sha256']
        for file in archive['files']:
            path = root / 'selection/extracted' / seq / file['path']
            assert path.stat().st_size == file['bytes'] and sha(path) == file['sha256']
            files_checked += 1
    label_lookup = {}
    for row in frame_labels:
        seq, frame = row['sequence'], row['frame']
        path = root / 'selection/extracted' / seq / (seq + '-GT') / f'{seq}-GT_{frame}.png'
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert bgr is not None
        moving = sum(np.count_nonzero(np.all(bgr == color, axis=-1)) for color in ((0,0,255),(0,255,0),(0,255,255)))
        stationary = np.count_nonzero(np.all(bgr == (255,255,255), axis=-1))
        unknown = np.count_nonzero(np.all(bgr == (128,128,128), axis=-1))
        expected = 'motion' if moving > 0 else 'ambiguous' if unknown > 0 and stationary == 0 else 'static'
        assert expected == row['state']
        assert moving == row['moving_pixels'] and stationary == row['stationary_pixels'] and unknown == row['unknown_pixels']
        label_lookup[(seq, frame)] = expected
    used = set()
    for source in sources:
        first, last = source['gt_span']; seq = source['sequence']
        indices = list(range(first, last + 1, 3))
        assert len(indices) == 16 and indices == source['windows']['clip']
        assert all(label_lookup[(seq, i)] == source['condition'] for i in range(first, last + 1))
        for frame in range(first, last + 1):
            assert (seq, frame) not in used
            used.add((seq, frame))
        for path, expected in zip(source['source_frames'], source['source_frame_sha256']):
            assert sha(path) == expected
    inputs = []; scores = []; gpu = []
    for ledger in sorted((root / 'construction').glob('shard-*/inputs.jsonl')):
        c = json.loads((ledger.parent / 'completion.json').read_text())
        assert c['status'] == 'finished' and c['failed'] == 0 and sha(ledger) == c['manifest_sha256']
        inputs.extend(rows(ledger))
    for ledger in sorted((root / 'scores').glob('shard-*/scores.jsonl')):
        c = json.loads((ledger.parent / 'completion.json').read_text())
        p = json.loads((ledger.parent / 'provenance.json').read_text())
        assert c['status'] == 'finished' and c['failed'] == 0 and sha(ledger) == c['scores_sha256']
        assert p['head_sha256'] == config['head_sha256'] and p['training_updates'] == 0
        assert not p['posthoc_mapping_changed'] and p['square_preprocessing_exact'] and p['training_feature_exact']
        assert p['head_reload_error'] <= 1e-6
        part = rows(ledger); assert {r['evaluation_id'] for r in part} == set(p['expected_ids'])
        scores.extend(part); gpu.append({'gpu': p['gpu'], 'wall_seconds': c['wall_seconds']})
    expected_ids = {f"{r['source_id']}:clip:{v}" for r in sources for v in config['views']}
    assert len(inputs) == len(scores) == len(expected_ids)
    assert {r['evaluation_id'] for r in inputs} == {r['evaluation_id'] for r in scores} == expected_ids
    predictions = {r['evaluation_id']: r for r in scores}
    for item in inputs:
        pred = predictions[item['evaluation_id']]
        assert sha(item['video']) == item['sha256'] == pred['input_sha256']
        assert item['status'] == pred['status'] == 'ok'
        assert item['decoded_pixels_sha256'] == pred['decoded_pixels_sha256']
        sampling, official = pred['diagnostics']['sampling'], pred['diagnostics']['official']
        assert sampling['sampled_frame_count'] == 16 and sampling['source_fps'] == 8 and sampling['sampling_interval'] == 1
        threshold = 6 * min(sampling['frame_shape']) / 256
        assert official['official_threshold'] == threshold
        assert len(official['raw_flow_top5_mean']) == 15
        assert pred['origin'] == float(sum(x > threshold for x in official['raw_flow_top5_mean']) >= 4)
        assert abs(pred['repair'] - 1 / (1 + np.exp(-pred['repair_latent']))) < 1e-12
    for source in sources:
        base = predictions[source['source_id'] + ':clip:original']
        control = predictions[source['source_id'] + ':clip:encoding_control']
        for key in ('decoded_pixels_sha256', 'feature_pixels_sha256', 'origin', 'repair'):
            assert base[key] == control[key]
    summary = json.loads((root / 'analysis/summary.json').read_text())
    pairs = rows(root / 'analysis/pairs.jsonl')
    assert sha(root / 'analysis/pairs.jsonl') == summary['pairs_sha256']
    assert {r['source_id'] for r in pairs} == {r['source_id'] for r in sources}
    seqs = sorted({r['sequence'] for r in sources})
    rng = np.random.default_rng(config['analysis']['bootstrap_seed'])
    draws = rng.integers(0, len(seqs), (config['analysis']['bootstrap_replicates'], len(seqs)))
    max_error = 0.
    def equal(x, y):
        nonlocal max_error
        error = float(np.max(abs(np.asarray(x)-np.asarray(y)))); max_error = max(max_error, error)
        assert error < 1e-12
    for pair in pairs:
        for backend in ('origin', 'repair'):
            equal(pair[backend], [predictions[pair['source_id'] + ':clip:' + v][backend]
                                 for v in ('original','jitter1701','jitter2904')])
    for condition in ('static','motion'):
        group = [p for p in pairs if p['condition'] == condition]
        for backend in ('origin','repair'):
            a = np.array([p[backend] for p in group]); result = summary['classes'][condition][backend]
            changes = a[:,1:] - a[:,0,None]; delta = changes.mean(1); mae = abs(changes).mean(1)
            equal([a[:,0].mean(),a[:,1:].mean(),delta.mean(),mae.mean()],
                  [result[k] for k in ('base','cf','delta','mae')])
            totals = np.array([[sum(p['sequence'] == seq for p in group),
                               sum(delta[i] for i,p in enumerate(group) if p['sequence'] == seq),
                               sum(mae[i] for i,p in enumerate(group) if p['sequence'] == seq)] for seq in seqs])
            sums = totals[draws].sum(axis=1); valid = sums[:,0] > 0
            for k, field in ((1,'delta_ci95'),(2,'mae_ci95')):
                equal(np.quantile(sums[valid,k]/sums[valid,0],[.025,.975]),result[field])
    for backend in ('origin','repair'):
        for j,name in enumerate(('base','jitter1701','jitter2904')):
            static = [p[backend][j] for p in pairs if p['condition'] == 'static']
            motion = [p[backend][j] for p in pairs if p['condition'] == 'motion']
            value = sum(float(m > s) + .5 * float(m == s) for m in motion for s in static)/(len(motion)*len(static))
            equal(value, summary['discrimination'][backend][name+'_auroc'])
    return {'status':'PASS', 'original_files_checked':files_checked, 'all_gt_frames_recomputed':len(frame_labels),
            'nonoverlapping_windows':len(sources), 'class_counts':dict(Counter(r['condition'] for r in sources)),
            'scored_inputs':len(inputs), 'origin_recomputed':len(scores), 'repair_sigmoids_recomputed':len(scores),
            'encoding_controls_exact':len(sources), 'maximum_arithmetic_error':max_error,
            'gpu_shards':gpu, 'summary_sha256':sha(root/'analysis/summary.json'),
            'auditor_sha256':sha(Path(__file__)), 'native_frame_timing_verified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','output'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args(); result=audit(args.root,args.config)
    with Path(args.output).open('x') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
