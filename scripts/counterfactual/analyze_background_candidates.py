"""Joint dev assessment of nuisance, sensitivity and natural preference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from background_consistency.algorithms import METHODS
from .analyze_background_development import summarize_region
from .analyze_background_native import load_native, summarize_native
from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .run_background_candidates import previous_rows
from .subject_artifacts import new_output, read_jsonl, write_json


CANDIDATES = tuple(k+'_'+s+'_'+a for k in ('sam03', 'box03')
                   for s in ('frame', 'union') for a in ('official', 'all_pairs_fp16'))
ALL_METHODS = (*METHODS, 'aggregation_fp16', *CANDIDATES)
PATCH_METHODS = (*METHODS, *(f'patch_{s}_{a}' for s in ('global', 'frame', 'union') for a in ('official', 'all_pairs')))


def checked_candidates(path, cohort, expected_count, *, protocol_path=None, methods=ALL_METHODS):
    rows, runs = previous_rows(path)
    protocol_hash = sha256_file(protocol_path or ROOT/'configs/background-repair/development_localizer_v3.json')
    if not runs or any(r['protocol_sha256'] != protocol_hash or r['cohort'] != cohort for r in runs):
        raise ValueError('unexpected or mixed candidate protocol/cohort')
    if len(rows) != expected_count or len({r['video_uid'] for r in rows}) != expected_count:
        raise ValueError('incomplete or duplicated candidate population')
    for row in rows:
        if row['status'] == 'completed':
            variants = row.get('variants', {'natural': row})
            if any(set(v['scores']) != set(methods) for v in variants.values()):
                raise ValueError('incomplete method grid marked completed')
            if any(v['origin_reencode_absolute_error'] > 1e-6 for v in variants.values()):
                raise ValueError('baseline input/precision parity failed')
    return rows, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-run', type=Path, required=True)
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--native-dataset', type=Path, required=True)
    parser.add_argument('--diagnostic-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/background-repair/development_localizer_v3.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['stage'] == 'background_localizer_development_v3':
        methods = ALL_METHODS
    elif protocol['stage'] == 'background_patch_pooling_development_v4':
        methods = PATCH_METHODS
    else:
        raise ValueError('unrecognized development analysis protocol')
    options = {'protocol_path': args.protocol, 'methods': methods}
    natural, natural_runs = checked_candidates(args.natural_run, 'natural', 680, **options)
    native, native_runs = checked_candidates(args.native_run, 'native', 680, **options)
    diagnostic, diagnostic_runs = checked_candidates(args.diagnostic_run, 'diagnostic', 288, **options)
    # Independently check each native identity, manifest hash and variant grid.
    _, manifests, _ = load_native(args.native_run, args.native_dataset, methods=methods)
    parent = json.loads((ROOT/'configs/background-repair/development_protocol_v2.json').read_text())
    expected = {r['video_uid']: r for r in read_jsonl(ROOT/'configs/background-repair'/parent['manifest_file'])
                if r['split'] == 'dev'}
    videos = {r['video_uid']: r for r in natural}
    if set(videos) != set(expected):
        raise ValueError('natural development identity mismatch')
    for uid, row in videos.items():
        if any(row[k] != value for k, value in expected[uid].items()):
            raise ValueError('natural input identity mismatch')
    pair_path = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(pair_path) != parent['pair_labels_sha256']:
        raise ValueError('background preference labels changed')
    pairs = [p for p in csv.DictReader(pair_path.open()) if p['dimension'] == 'background_consistency' and p['split'] == 'dev']
    if len(pairs) != 1020:
        raise ValueError('incomplete natural development pairs')
    for pair in pairs:
        if any(videos[pair[k]]['prompt_id'] != pair['prompt_id'] for k in ('video_a_uid', 'video_b_uid')):
            raise ValueError('preference pair does not match source prompt')
    natural_stats = evaluate_population(pairs, videos, methods, {m: 0 for m in methods})
    natural_stats.update(video_count=len(natural), status_counts=dict(Counter(r['status'] for r in natural)))
    native_stats, native_cases = summarize_native(native, manifests, methods=methods)
    diagnostic_stats, diagnostic_cases = summarize_region(diagnostic, methods=methods)
    report = {'stage': protocol['stage']+'; held-out scoring NOT RUN', 'methods': methods,
              'natural': natural_stats, 'native': native_stats, 'diagnostic': diagnostic_stats,
              'source_runs': {'natural': natural_runs, 'native': native_runs, 'diagnostic': diagnostic_runs}}
    output = new_output(args.output)
    write_json(output/'statistics.json', report)
    for name, cases in (('native', native_cases), ('diagnostic', diagnostic_cases)):
        with (output/(name+'_per_case.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader(); writer.writerows(cases)
    print(json.dumps({'output': str(output), 'natural_status': natural_stats['status_counts'],
                      'native_status': native_stats['status_counts'], 'diagnostic_status': diagnostic_stats['status_counts']}))


if __name__ == '__main__':
    main()
