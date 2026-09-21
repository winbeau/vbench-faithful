"""Analyze all candidates, explicit rejections, and paired background responses."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_subject_natural import score
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json, write_jsonl


def cluster_summary(values, groups, *, resamples=10000, seed=20260920):
    """Keep every member of a sampled prompt, including repeated generators."""
    ids = sorted(values)
    if not ids:
        return {'n_bases': 0, 'n_prompt_clusters': 0, 'mean': None, 'median': None,
                'maximum': None, 'mean_ci95': None, 'median_ci95': None}
    array = np.asarray([values[uid] for uid in ids], dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('non-finite intervention response')
    clusters = sorted({groups[uid] for uid in ids})
    members = [np.asarray([i for i, uid in enumerate(ids) if groups[uid] == g]) for g in clusters]
    result = {'n_bases': len(ids), 'n_prompt_clusters': len(clusters),
              'mean': float(array.mean()), 'median': float(np.median(array)),
              'maximum': float(array.max()), 'mean_ci95': None, 'median_ci95': None}
    if len(clusters) > 1:
        rng = np.random.default_rng(seed)
        means, medians = [], []
        for draw in rng.integers(0, len(clusters), (resamples, len(clusters))):
            sample = array[np.concatenate([members[i] for i in draw])]
            means.append(sample.mean()); medians.append(np.median(sample))
        result['mean_ci95'] = np.quantile(means, [.025, .975]).tolist()
        result['median_ci95'] = np.quantile(medians, [.025, .975]).tolist()
    return result


def frame_presence(variant):
    diag = variant.get('localizer_diagnostics')
    return 1 - diag['num_missing_frames'] / diag['num_frames'] if diag else None


def summarize_cases(cases, *, resamples=10000, seed=20260920):
    groups = {r['base_id']: r['prompt_id'] for r in cases}
    def stats(key):
        return cluster_summary({r['base_id']: r[key] for r in cases if r[key] is not None},
                               groups, resamples=resamples, seed=seed)
    drops = [r['subject_drop'] for r in cases if r['subject_drop'] is not None]
    return {'n_construction_accepted': len(cases),
            'background_absolute_change': stats('background_abs_change'),
            'background_signed_drop': stats('background_drop'),
            'paired_background_abs_reduction_vs_official': stats('abs_reduction_vs_official'),
            'subject_signed_drop': stats('subject_drop'),
            'subject_response_counts': {'scored': len(drops), 'positive': sum(d > 0 for d in drops),
                                        'zero': sum(d == 0 for d in drops), 'negative': sum(d < 0 for d in drops)},
            'background_zero_scores_both_versions': sum(r['clean_score'] == r['background_score'] == 0 for r in cases),
            'clean_and_background_masks_all_empty': sum(r['clean_presence'] == r['background_presence'] == 0 for r in cases)}


def make_cases(records, position, method):
    result = []
    for row in records:
        if row['construction_status'] != 'accepted':
            continue
        variants = row['variants']
        clean, bg, fg = (variants.get(k, {}) for k in
                         ('clean', position+'/background_corrupt', position+'/subject_corrupt'))
        a, b, c = (score(v, method) for v in (clean, bg, fg))
        ao, bo = (score(v, 'official') for v in (clean, bg))
        bg_drop = a-b if a is not None and b is not None else None
        fg_drop = a-c if a is not None and c is not None else None
        reduction = abs(ao-bo)-abs(bg_drop) if None not in (ao, bo, bg_drop) else None
        result.append({'base_id': row['base']['base_id'], 'prompt_id': row['base']['prompt_id'],
            'subject_en': row['base']['subject_en'], 'position': position, 'method': method,
            'clean_score': a, 'background_score': b, 'subject_score': c,
            'background_drop': bg_drop, 'background_abs_change': abs(bg_drop) if bg_drop is not None else None,
            'subject_drop': fg_drop, 'abs_reduction_vs_official': reduction,
            'clean_presence': frame_presence(clean), 'background_presence': frame_presence(bg),
            'subject_presence': frame_presence(fg), 'runtime_status': row['status']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True, help='Dataset metadata or full dataset')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = ROOT/'configs/subject-repair'
    protocol_path = config/'background288_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    manifest_path = config/'background288_manifest.jsonl'
    if sha256_file(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('candidate manifest changed')
    expected = {r['video_uid']: r for r in read_jsonl(manifest_path)}
    method = json.loads((config/'natural1440_protocol_v2.json').read_text())
    if sha256_file(config/'natural1440_protocol_v2.json') != protocol['method_config_sha256']:
        raise ValueError('method protocol changed')
    methods = ['official', 'aggregation'] + [m+'_'+p for m in method['representations'] for p in ('zero', 'exclude')]
    dataset_index = {r['video_uid']: r for r in read_jsonl(args.dataset/'index.jsonl')}
    records, runs = [], []
    for directory in sorted(args.run.glob('shard*')):
        if not directory.is_dir():
            continue
        run = json.loads((directory/'run.json').read_text())
        if (not run.get('completed') or run['protocol_sha256'] != sha256_file(protocol_path)
                or run['dataset_index_sha256'] != sha256_file(args.dataset/'index.jsonl')
                or run['method_config_sha256'] != protocol['method_config_sha256']):
            raise ValueError('incomplete or mismatched score shard')
        if sha256_file(directory/'scores.jsonl') != run['scores_sha256']:
            raise ValueError('score hash mismatch')
        rows = read_jsonl(directory/'scores.jsonl')
        if len(rows) != run['base_count']:
            raise ValueError('shard row count mismatch')
        records.extend(rows); runs.append(run)
    videos = {r['base']['video_uid']: r for r in records}
    if len(videos) != len(records) or set(videos) != set(expected) or set(dataset_index) != set(expected):
        raise ValueError('duplicate or incomplete candidate population')
    for uid, row in videos.items():
        if any(row['base'][k] != expected[uid][k] for k in ('base_id', 'prompt_id', 'subject_en', 'relative_video_path')):
            raise ValueError('candidate identity mismatch')
        if row['construction_status'] != dataset_index[uid]['status']:
            raise ValueError('construction status mismatch')
        if row['status'] == 'completed':
            expected_variants = {'clean'} | {p+'/'+l for p in protocol['positions'] for l in protocol['levels'][1:]}
            if set(row['variants']) != expected_variants or any(set(v['scores']) != set(methods) for v in row['variants'].values()):
                raise ValueError('incomplete score grid marked completed')
    report = {'stage': protocol['stage'], 'total_candidates': len(records),
              'status_counts': dict(Counter(r['status'] for r in records)),
              'construction_status_counts': dict(Counter(r['construction_status'] for r in records)),
              'rejection_reason_counts': dict(Counter(reason for r in records for reason in r['rejection_reasons'])),
              'primary_method': protocol['primary_representation'], 'positions': {}, 'localization': {},
              'bootstrap': {'unit': 'source_prompt_id', 'resamples': 10000, 'seed': 20260920},
              'claim_limit': 'Automatic construction and scoring masks are not human ground truth; rejected cases are not measured invariance successes.'}
    flat = []
    for position in protocol['positions']:
        report['positions'][position] = {}
        for name in methods:
            cases = make_cases(records, position, name); flat.extend(cases)
            stats = summarize_cases(cases)
            stats['coverage_strata'] = {}
            for threshold in (.5, .9):
                selected = [r for r in cases if None not in (r['clean_presence'], r['background_presence'])
                            and min(r['clean_presence'], r['background_presence']) >= threshold]
                stats['coverage_strata'][str(threshold)] = summarize_cases(selected)
            report['positions'][position][name] = stats
    keys = ['clean'] + [p+'/'+l for p in protocol['positions'] for l in protocol['levels'][1:]]
    for key in keys:
        variants = [r['variants'][key] for r in records if key in r['variants']]
        diags = [v['localizer_diagnostics'] for v in variants if v.get('localizer_diagnostics')]
        presence = [frame_presence(v) for v in variants if frame_presence(v) is not None]
        report['localization'][key] = {'n_variants': len(variants), 'n_mask_diagnostics': len(diags),
            'n_frames': sum(d['num_frames'] for d in diags),
            'n_missing_frames': sum(d['num_missing_frames'] for d in diags),
            'n_variants_all_empty': sum(v == 0 for v in presence),
            'n_variants_all_frames_present': sum(v == 1 for v in presence),
            'mean_frame_presence_per_variant': float(np.mean(presence)) if presence else None}
    report['provenance'] = {'protocol_sha256': sha256_file(protocol_path),
        'dataset_index_sha256': sha256_file(args.dataset/'index.jsonl'),
        'score_shards_sha256': [r['scores_sha256'] for r in runs],
        'source_sha256': sha256_file(Path(__file__))}
    report['timing'] = {'gpu_devices': [r['physical_gpu'] for r in runs],
        'scoring_wall_seconds': max(r['finished_unix'] for r in runs)-min(r['started_unix'] for r in runs),
        'note': 'Per-shard timing starts after full dataset integrity verification.'}
    output = new_output(args.output)
    write_json(output/'statistics.json', report)
    write_jsonl(output/'scores_merged.jsonl', sorted(records, key=lambda r: r['base']['video_uid']))
    if flat:
        with (output/'per_case.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({'status_counts': report['status_counts'], 'full': report['positions']['full'],
                      'localization': report['localization'], 'timing': report['timing']}, indent=2))


if __name__ == '__main__':
    main()
