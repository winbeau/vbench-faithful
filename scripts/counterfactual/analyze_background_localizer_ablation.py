"""Descriptive dev ablations from frozen scores; no fitting or model inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from background_consistency.calibration import calibrate_patch_score, REPAIR_GAIN
from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def verified_run(path):
    run = json.loads((path/'run.json').read_text())
    if not run.get('completed') or sha256_file(path/'scores.jsonl') != run['scores_sha256']:
        raise ValueError('complete verified scores required')
    rows = read_jsonl(path/'scores.jsonl')
    if len(rows) != run['video_count'] or len({row['video_uid'] for row in rows}) != len(rows):
        raise ValueError('incomplete or duplicated run')
    return run, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural', type=Path, required=True)
    parser.add_argument('--counterfactual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, rows = verified_run(args.natural)
    protocol = json.loads((args.natural/'protocol.json').read_text())
    labels = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(labels) != protocol['pair_labels_sha256']:
        raise ValueError('preference labels changed')
    pairs = [row for row in csv.DictReader(labels.open())
             if row['dimension'] == 'background_consistency' and row['split'] == 'dev']
    if len(pairs) != 1020 or len(rows) != 680 or any(row['split'] != 'dev' for row in rows):
        raise ValueError('full native development population required')
    videos = {}
    for row in rows:
        values = dict(row['scores'])
        if row['status'] == 'completed':
            grid = row['score_grids']['caption']
            if grid['patch_global_all_pairs'] != row['score_grids']['coco80']['patch_global_all_pairs']:
                raise ValueError('global ablation unexpectedly depends on localization')
            for key, value in {
                'aggregation_fp16': grid['aggregation_fp16'],
                'global_patch_official_gain1p75': calibrate_patch_score(grid['patch_global_official'], gain=REPAIR_GAIN),
                'global_patch_gain1p75': calibrate_patch_score(grid['patch_global_all_pairs'], gain=REPAIR_GAIN),
            }.items():
                values[key] = {'score': value, 'status': 'succeeded'}
        videos[row['video_uid']] = {**row, 'scores': values}
    for pair in pairs:
        if any(videos[pair[key]]['prompt_id'] != pair['prompt_id']
               for key in ('video_a_uid', 'video_b_uid')):
            raise ValueError('preference prompt mismatch')
    methods = ['official', 'aggregation_fp16', 'global_patch_official_gain1p75',
               'global_patch_gain1p75', 'frozen19', 'coco80', 'caption']
    natural = evaluate_population(pairs, videos, methods, {method: 0 for method in methods})
    paired = evaluate_population(pairs, videos, ['frozen19', 'global_patch_gain1p75'],
                                 {'frozen19': 0, 'global_patch_gain1p75': 0})['methods']['global_patch_gain1p75']
    natural['paired_global_vs_frozen19'] = {
        'delta': paired['paired_delta_vs_official_full_denominator'],
        'ci95': paired['paired_delta_ci95'],
    }

    cf_run, cf_rows = verified_run(args.counterfactual)
    if (cf_run.get('vocabulary_protocol_sha256') != run['candidate_protocol_sha256']
            or cf_run['method_protocol_sha256'] != protocol['method_protocol_sha256']):
        raise ValueError('natural and counterfactual methods must match')
    # No localizer is removed or substituted in the primary result. These
    # quantities were already available in the frozen all-method score grid.
    cases = []
    for row in cf_rows:
        if row['status'] != 'completed':
            continue
        if row['base']['split'] != 'dev':
            raise ValueError('counterfactual ablation must remain development-only')
        if row['variants']['clean']['frame_array_sha256'] != videos[row['video_uid']]['frame_array_sha256']:
            raise ValueError('natural and counterfactual clean pixels differ')
        clean = row['variants']['clean']['scores']
        blurred = row['variants']['full/subject_blur']['scores']
        scores = {
            'official': [clean['official'], blurred['official']],
            'aggregation_fp16': [clean['aggregation_fp16'], blurred['aggregation_fp16']],
            'global_patch_official_gain1p75': [calibrate_patch_score(grid['patch_global_official'], gain=REPAIR_GAIN)
                                             for grid in (clean, blurred)],
            'global_patch_gain1p75': [calibrate_patch_score(grid['patch_global_all_pairs'], gain=REPAIR_GAIN)
                                    for grid in (clean, blurred)],
            'caption': [clean['patch_frame_all_pairs_calibrated'], blurred['patch_frame_all_pairs_calibrated']],
        }
        cases.append({'video_uid': row['video_uid'], 'prompt_id': row['base']['prompt_id'],
                      'scores': scores, 'absolute_changes': {key: abs(b-a) for key, (a, b) in scores.items()}})
    summary = {method: {'count': len(cases),
                       'mean_absolute_change': float(np.mean([row['absolute_changes'][method] for row in cases])),
                       'maximum_absolute_change': max(row['absolute_changes'][method] for row in cases)}
               for method in ('official', 'aggregation_fp16', 'global_patch_official_gain1p75',
                              'global_patch_gain1p75', 'caption')} if cases else {}
    result = {
        'natural': natural, 'counterfactual_descriptive': summary, 'counterfactual_cases': cases,
        'natural_scores_sha256': run['scores_sha256'], 'counterfactual_scores_sha256': cf_run['scores_sha256'],
        'source_sha256': sha256_file(Path(__file__)), 'fixed_gain': REPAIR_GAIN,
        'scope': 'Exploratory attribution on existing dev runs, not a new independent validation or a selected new default. '
                 'Same frozen gain 1.75 for the global patch ablation and masked repair; no labels used to fit it. '
                 'Counterfactual statistics cover the reviewed subset only.',
    }
    output = new_output(args.output)
    write_json(output/'statistics.json', result)
    print(json.dumps({'natural_accuracy': {key: value['accuracy_full_denominator']
                     for key, value in natural['methods'].items()}, 'counterfactual': summary}))


if __name__ == '__main__':
    main()
