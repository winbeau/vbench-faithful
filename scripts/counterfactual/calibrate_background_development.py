"""Calibrate completed v4 development scores without re-running image models."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path

from background_consistency.calibration import PATCH_GAIN, calibrate_patch_score
from .analyze_background_candidates import PATCH_METHODS, checked_candidates
from .analyze_background_development import summarize_region
from .analyze_background_native import load_native, summarize_native
from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json, write_jsonl


RAW = tuple(m for m in PATCH_METHODS if m.startswith('patch_'))
METHODS = (*PATCH_METHODS, *(m+'_gain2' for m in RAW))


def calibrated_rows(rows):
    result = copy.deepcopy(rows)
    clipping = Counter()
    for row in result:
        for variant in row.get('variants', {'natural': row}).values():
            for method in RAW:
                value = variant.get('scores', {}).get(method, {})
                if value.get('status') != 'succeeded':
                    continue
                raw = value['score']
                clipping[method] += raw < .5 or raw > 1
                variant['scores'][method+'_gain2'] = {'status': 'succeeded', 'score': calibrate_patch_score(raw)}
    return result, dict(clipping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-run', type=Path, required=True)
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--diagnostic-run', type=Path, required=True)
    parser.add_argument('--native-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = ROOT/'configs/background-repair'
    protocol = config/'development_calibration_v5.json'
    if json.loads(protocol.read_text())['gain'] != PATCH_GAIN:
        raise ValueError('calibration implementation and protocol disagree')
    options = {'protocol_path': config/'development_patches_v4.json', 'methods': PATCH_METHODS}
    natural, nruns = checked_candidates(args.natural_run, 'natural', 680, **options)
    native, iruns = checked_candidates(args.native_run, 'native', 680, **options)
    diagnostic, druns = checked_candidates(args.diagnostic_run, 'diagnostic', 288, **options)
    _, manifests, _ = load_native(args.native_run, args.native_dataset, methods=PATCH_METHODS)
    parent = json.loads((config/'development_protocol_v2.json').read_text())
    expected = {r['video_uid']: r for r in read_jsonl(config/parent['manifest_file']) if r['split'] == 'dev'}
    if {r['video_uid'] for r in natural} != set(expected):
        raise ValueError('calibration may use development inputs only')
    labels = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(labels) != parent['pair_labels_sha256']:
        raise ValueError('background preference labels changed')
    pairs = [p for p in csv.DictReader(labels.open()) if p['dimension'] == 'background_consistency' and p['split'] == 'dev']
    if len(pairs) != 1020:
        raise ValueError('incomplete development preference population')
    natural, nclip = calibrated_rows(natural)
    native, iclip = calibrated_rows(native)
    diagnostic, dclip = calibrated_rows(diagnostic)
    nstats = evaluate_population(pairs, {r['video_uid']: r for r in natural}, METHODS, {m: 0 for m in METHODS})
    istats, _ = summarize_native(native, manifests, methods=METHODS)
    dstats, _ = summarize_region(diagnostic, methods=METHODS)
    report = {'stage': 'v5 development calibration; independent test NOT RUN',
        'protocol_sha256': sha256_file(protocol), 'natural': nstats, 'native': istats, 'diagnostic': dstats,
        'calibration_source_sha256': sha256_file(ROOT/'metrics/background-consistency/src/background_consistency/calibration.py'),
        'analysis_source_sha256': sha256_file(Path(__file__)),
        'clipping_counts': {'natural': nclip, 'native': iclip, 'diagnostic': dclip},
        'source_runs': {'natural': nruns, 'native': iruns, 'diagnostic': druns}}
    output = new_output(args.output)
    write_json(output/'statistics.json', report)
    for name, rows in (('natural', natural), ('native', native), ('diagnostic', diagnostic)):
        write_jsonl(output/(name+'_scores.jsonl'), rows)
    print(json.dumps({'output': str(output), 'clipping_counts': report['clipping_counts']}))


if __name__ == '__main__':
    main()
