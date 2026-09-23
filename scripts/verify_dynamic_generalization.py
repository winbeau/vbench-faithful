#!/usr/bin/env python3
"""Verify the external Dynamic evidence release without media or GPU inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(root, entries):
    root = Path(root).resolve(); seen = set()
    for entry in entries:
        relative = PurePosixPath(entry['path']); path = root / relative
        require(not relative.is_absolute() and '..' not in relative.parts
                and root in path.resolve().parents and not path.is_symlink(), 'Unsafe member')
        require(relative.as_posix() not in seen, 'Duplicate member')
        seen.add(relative.as_posix())
        require(path.is_file() and path.stat().st_size == entry['bytes']
                and sha(path) == entry['sha256'], f'Content mismatch: {relative}')
    return len(seen)


def replay(root, cohort):
    root = Path(root)
    config = read(root / 'configs/dynamic-generalization' / (cohort + '.json'))
    evidence = root / 'output/dynamic-generalization' / cohort
    summary = read(evidence / 'analysis/summary.json')
    original_audit = read(evidence / 'analysis/audit.json')
    pairs = rows(evidence / 'analysis/pairs.jsonl')
    require(summary['status'] == 'finished' and summary['failures'] == 0, 'Incomplete evaluation')
    require(original_audit['status'] == 'PASS', 'Original media audit did not pass')
    require(sha(evidence / 'analysis/summary.json') == original_audit['summary_sha256'], 'Audit identity')
    require(sha(evidence / 'analysis/pairs.jsonl') == summary['pairs_sha256'], 'Pair identity')
    require(sha(root / 'configs/dynamic-generalization' / (cohort + '.json')) == summary['config_sha256'], 'Config identity')
    selection_dir = evidence / ('selection-v5' if cohort == 'bmc-v1' else 'selection')
    selection = read(selection_dir / 'selection.json')
    sources = rows(selection_dir / 'sources.jsonl')
    require(sha(selection_dir / 'sources.jsonl') == selection['sources_sha256'], 'Source identity')
    require(len({p['source_id'] for p in pairs}) == len(pairs) == len(sources), 'Pair coverage')
    require({p['source_id'] for p in pairs} == {s['source_id'] for s in sources}, 'Source coverage')
    records = []; inputs = []
    for path in sorted((evidence / 'scores').glob('shard-*/scores.jsonl')):
        done = read(path.parent / 'completion.json'); provenance = read(path.parent / 'provenance.json')
        shard = rows(path)
        require(done['status'] == 'finished' and done['failed'] == 0
                and done['completed'] == done['expected'] == len(shard), 'Incomplete shard')
        require(sha(path) == done['scores_sha256'], 'Score identity')
        require(provenance['head_sha256'] == config['head_sha256']
                and provenance['training_updates'] == 0 and not provenance['posthoc_mapping_changed'], 'Frozen model')
        require({r['evaluation_id'] for r in shard} == set(provenance['expected_ids']), 'Shard membership')
        records.extend(shard)
    for path in sorted((evidence / 'construction').glob('shard-*/inputs.jsonl')):
        done = read(path.parent / 'completion.json')
        require(done['status'] == 'finished' and not done['failed']
                and sha(path) == done['manifest_sha256'], 'Construction identity')
        inputs.extend(rows(path))
    expected = {p['source_id'] + ':clip:' + v for p in pairs for v in config['views']}
    predictions = {r['evaluation_id']: r for r in records}
    require(len(expected) == len(records) == len(inputs) == summary['inputs_per_backend'], 'Input count')
    require(set(predictions) == {r['evaluation_id'] for r in inputs} == expected, 'Input membership')
    error_max = 0.0

    def equal(actual, expected_value):
        nonlocal error_max
        error = float(np.max(abs(np.asarray(actual) - np.asarray(expected_value))))
        require(np.isfinite(error) and error < 1e-12, f'Numeric mismatch: {error}')
        error_max = max(error_max, error)

    for item in inputs:
        pred = predictions[item['evaluation_id']]
        require(item['status'] == pred['status'] == 'ok'
                and item['sha256'] == pred['input_sha256']
                and item['decoded_pixels_sha256'] == pred['decoded_pixels_sha256'], 'Input/score binding')
        sampling = pred['diagnostics']['sampling']; official = pred['diagnostics']['official']
        require(sampling['sampled_frame_count'] == 16 and sampling['source_fps'] == 8
                and sampling['sampling_interval'] == 1, 'Sampling protocol')
        threshold = 6 * min(sampling['frame_shape']) / 256
        require(threshold == official['official_threshold'] and len(official['raw_flow_top5_mean']) == 15, 'RAFT protocol')
        equal(pred['origin'], float(sum(x > threshold for x in official['raw_flow_top5_mean']) >= 4))
        equal(pred['repair'], 1 / (1 + np.exp(-pred['repair_latent'])))
    for pair in pairs:
        prefix = pair['source_id'] + ':clip:'
        base = predictions[prefix + 'original']; control = predictions[prefix + 'encoding_control']
        require(all(base[k] == control[k] for k in ('decoded_pixels_sha256', 'feature_pixels_sha256', 'origin', 'repair')), 'Encoding control')
        for backend in ('origin', 'repair'):
            equal(pair[backend], [predictions[prefix + v][backend] for v in ('original', 'jitter1701', 'jitter2904')])
    sequences = sorted({p['sequence'] for p in pairs})
    analysis = config['analysis']
    draws = np.random.default_rng(analysis['bootstrap_seed']).integers(0, len(sequences), (analysis['bootstrap_replicates'], len(sequences)))
    groups = summary.get('statistics', summary.get('classes'))
    for group, result in groups.items():
        chosen = [p for p in pairs if group == 'all' or p['condition'] == group]
        require(len(chosen) == result.get('n', result['origin'].get('n')), 'Group denominator')

        def interval(vector):
            aggregates = np.array([[sum(p['sequence'] == s for p in chosen),
                sum(v for v, p in zip(vector, chosen) if p['sequence'] == s)] for s in sequences])
            sampled = aggregates[draws].sum(1); valid = sampled[:, 0] > 0
            return np.quantile(sampled[valid, 1] / sampled[valid, 0], [.025, .975])

        deltas = {}
        for backend in ('origin', 'repair'):
            values = np.array([p[backend] for p in chosen]); diff = values[:, 1:] - values[:, 0, None]
            delta, mae = diff.mean(1), abs(diff).mean(1)
            equal([values[:, 0].mean(), values[:, 1:].mean(), delta.mean(), mae.mean()],
                  [result[backend][k] for k in ('base', 'cf', 'delta', 'mae')])
            equal(interval(delta), result[backend]['delta_ci95'])
            equal(interval(mae), result[backend]['mae_ci95'])
            deltas[backend] = delta
        if 'ten_percent_margin' in result:
            margin = deltas['repair'] - .1 * deltas['origin']
            equal(margin.mean(), result['ten_percent_margin']['mean'])
            equal(interval(margin), result['ten_percent_margin']['ci95'])
    for backend in ('origin', 'repair'):
        for j, name in enumerate(('base', 'jitter1701', 'jitter2904')):
            static = [p[backend][j] for p in pairs if p['condition'] == 'static']
            motion = [p[backend][j] for p in pairs if p['condition'] == 'motion']
            auc = sum(float(y > x) + .5 * float(y == x) for x in static for y in motion) / (len(static) * len(motion))
            equal(auc, summary['discrimination'][backend][name + '_auroc'])
    return {'cohort': cohort, 'clips': len(pairs), 'recordings': len(sequences),
            'inputs_per_backend': len(records), 'maximum_arithmetic_error': error_max,
            'status': 'PASS', 'media_redownloaded_or_decoded': False, 'gpu_inference_rerun': False}


def verify(root):
    manifest = read(Path(root) / 'manifest.json')
    require(manifest['schema'] == 'vbench-repair-dynamic-external/1', 'Unexpected schema')
    count = verify_files(root, manifest['files'])
    return {'status': 'PASS', 'files_verified': count, 'manifest_sha256': sha(Path(root) / 'manifest.json'),
            'cohorts': [replay(root, c) for c in ('lasiesta-v1', 'bmc-v1')]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = verify(args.bundle)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
