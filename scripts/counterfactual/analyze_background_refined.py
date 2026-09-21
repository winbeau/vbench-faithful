"""Summarize reviewed development scores while retaining pending/failed inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_background_native import summarize_native
from .common import sha256_file
from .compare_background_construction import load_completed
from .subject_artifacts import new_output, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    construction, manifests = load_completed(args.dataset)
    run = json.loads((args.run/'run.json').read_text())
    if (not run.get('completed') or run['scores_sha256'] != sha256_file(args.run/'scores.jsonl')
            or run['dataset_index_sha256'] != construction['index_sha256']):
        raise ValueError('incomplete or mismatched scores')
    rows = read_jsonl(args.run/'scores.jsonl')
    if len(rows) != len(manifests) or {r['video_uid'] for r in rows} != set(manifests):
        raise ValueError('score coverage mismatch')
    primary = run['primary_method']; methods = ['official', primary]
    normalized = []
    parity, localization = [], []
    for row in rows:
        source = manifests[row['video_uid']]
        if row['base'] != source['base'] or row['construction_status'] != source['status']:
            raise ValueError('score/construction identity mismatch')
        variants = {}
        if row['status'] == 'completed':
            for name, value in row['variants'].items():
                variants[name] = {**value, 'scores': {method: {'score': score, 'status': 'succeeded'}
                                                     for method, score in value['scores'].items()}}
                localization.append({'video_uid': row['video_uid'], 'variant': name,
                                     'diagnostics': value['localizer_diagnostics']})
            parity.append(row['variants']['clean']['parity_absolute_error'])
        normalized.append({**row, 'variants': variants})
    statistics, cases = summarize_native(normalized, manifests, methods=methods)
    counts = dict(Counter(row['status'] for row in rows))
    large = {}
    for method in methods:
        values = [r['foreground_absolute_change'] for r in cases
                  if r['method'] == method and r['position'] == 'full' and r['foreground_absolute_change'] is not None]
        large[method] = {'scored_bases': len(values), **{
            f'fraction_at_least_{threshold:g}': float(np.mean(np.asarray(values) >= threshold)) if values else None
            for threshold in (.1, .2)}}
    statistics.update(primary_method=primary, dataset_index_sha256=construction['index_sha256'],
                      scoring_candidate=run.get('scoring_candidate', 'frozen_coco19'),
                      vocabulary_protocol_sha256=run.get('vocabulary_protocol_sha256'),
                      score_run_sha256=sha256_file(args.run/'run.json'), scores_sha256=run['scores_sha256'],
                      semantic_review_counts=run['semantic_review_counts'],
                      staged_review_subset=bool(counts.get('semantic_review_pending', 0)),
                      full_blur_large_changes=large, max_official_parity_error=max(parity) if parity else None,
                      localizer_diagnostics=localization,
                      claim_scope='Development diagnostic only. Pending reviews are not scored, not silently rejected; the reviewed subset is not representative of the entire cohort.',
                      source_sha256=sha256_file(Path(__file__)))
    output = new_output(args.output); write_json(output/'statistics.json', statistics)
    with (output/'per_case.csv').open('w', newline='') as handle:
        if cases:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0])); writer.writeheader(); writer.writerows(cases)
    write_json(output/'per_case.json', cases)
    print(json.dumps({'status_counts': counts, 'full': statistics['by_position']['full'], 'large_changes': large}))


if __name__ == '__main__':
    main()
