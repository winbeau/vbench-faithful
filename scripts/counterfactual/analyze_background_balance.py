"""Evaluate the single declared balanced temporal candidate, on dev only."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from background_consistency.calibration import PATCH_GAIN, PATCH_OFFICIAL_WEIGHT, REPAIR_GAIN, balanced_patch_score, calibrate_patch_score
from .analyze_background_candidates import PATCH_METHODS, checked_candidates
from .analyze_background_development import summarize_region
from .analyze_background_native import load_native, summarize_native
from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, write_json, write_jsonl

METHOD = 'patch_frame_balanced_gain2'


def add_balance(rows, *, final_scale=False):
    rows = copy.deepcopy(rows)
    for row in rows:
        for v in row.get('variants', {'natural': row}).values():
            scores = v.get('scores', {})
            a, b = (scores.get('patch_frame_'+k, {}) for k in ('official', 'all_pairs'))
            if a.get('status') == b.get('status') == 'succeeded':
                method = 'patch_frame_all_pairs_calibrated' if final_scale else METHOD
                value = calibrate_patch_score(b['score'], gain=REPAIR_GAIN) if final_scale else balanced_patch_score(a['score'], b['score'])
                scores[method] = {'status': 'succeeded', 'score': value}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Local background experiment root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final-scale', action='store_true')
    args = parser.parse_args()
    config = ROOT/'configs/background-repair'
    protocol_path = config/('development_scale_v7.json' if args.final_scale else 'development_balance_v6.json')
    protocol = json.loads(protocol_path.read_text())
    expected_gain = REPAIR_GAIN if args.final_scale else PATCH_GAIN
    expected_weight = 0 if args.final_scale else PATCH_OFFICIAL_WEIGHT
    if protocol['gain'] != expected_gain or protocol['official_weight'] != expected_weight or protocol['all_pairs_weight'] != 1-expected_weight:
        raise ValueError('balance protocol and implementation differ')
    options = {'protocol_path': config/'development_patches_v4.json', 'methods': PATCH_METHODS}
    runs, cohorts = {}, {}
    for cohort, count in (('natural', 680), ('native', 680), ('diagnostic', 288)):
        rows, runs[cohort] = checked_candidates(args.root/f'background-{cohort}-patches-v4-20260920', cohort, count, **options)
        cohorts[cohort] = add_balance(rows, final_scale=args.final_scale)
    _, manifests, _ = load_native(args.root/'background-native-patches-v4-20260920',
                                 args.root/'background-native-construction-v1-20260920', methods=PATCH_METHODS)
    parent = json.loads((config/'development_protocol_v2.json').read_text())
    labels = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(labels) != parent['pair_labels_sha256']:
        raise ValueError('background preference labels changed')
    pairs = [r for r in csv.DictReader(labels.open()) if r['dimension'] == 'background_consistency' and r['split'] == 'dev']
    if len(pairs) != 1020:
        raise ValueError('incomplete development preference cohort')
    method = 'patch_frame_all_pairs_calibrated' if args.final_scale else METHOD
    methods = ('official', method)
    report = {'stage': protocol['stage'], 'protocol_sha256': sha256_file(protocol_path), 'source_runs': runs,
        'natural': evaluate_population(pairs, {r['video_uid']: r for r in cohorts['natural']}, methods, {m: 0 for m in methods}),
        'native': summarize_native(cohorts['native'], manifests, methods=methods)[0],
        'diagnostic': summarize_region(cohorts['diagnostic'], methods=methods)[0]}
    output = new_output(args.output)
    for path in (protocol_path, Path(__file__), ROOT/'metrics/background-consistency/src/background_consistency/calibration.py'):
        target = output/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    write_json(output/'statistics.json', report)
    for name, rows in cohorts.items():
        write_jsonl(output/(name+'_scores.jsonl'), rows)
    print(json.dumps({'output': str(output), 'method': method}))


if __name__ == '__main__':
    main()
