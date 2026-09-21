"""Exploratory dev-only alternative to scoring unavailable subject evidence zero."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analyze_subject_natural import evaluate_population, score
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cls-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.cls_run/'run.json').read_text())
    path = args.cls_run/'scores.jsonl'
    if not run.get('completed') or sha256_file(path) != run['scores_sha256']:
        raise ValueError('changed or incomplete CLS run')
    rows = read_jsonl(path)
    videos = {r['video_uid']: r for r in rows}
    if len(rows) != 580 or len(videos) != 580 or any(r['split'] != 'dev' for r in rows):
        raise ValueError('only the 580 unique development videos are permitted')
    output = new_output(args.output)
    protocol = {'stage': 'post_hoc_development_missing_evidence_diagnostic', 'split': 'dev only',
        'rule': 'Use isolated CLS exclude when available; otherwise use recorded Official only for fewer than two subject frames. Runtime errors do not silently fall back.',
        'views': ['full', 'crop'], 'tie_margin': 0., 'parameters_fitted': False,
        'claim_limit': 'Exploratory same-development-set evidence, not a new held-out validation or adopted production method.',
        'input_sha256': sha256_file(path), 'source_sha256': sha256_file(Path(__file__))}
    write_json(output/'protocol.json', protocol)
    methods = ['official', 'aggregation']; counts = {}
    for view in protocol['views']:
        source = 'cls_'+view+'_exclude'; name = source+'_fallback_official'
        methods.extend((source, name)); counts[name] = 0
        for row in rows:
            value = score(row, source)
            fallback = value is None and row['scores'].get(source, {}).get('failure_reason') == 'fewer than two frames carry subject evidence'
            if fallback:
                value = score(row, 'official'); counts[name] += 1
            row['scores'][name] = {'status': 'succeeded' if value is not None else 'failed', 'score': value}
    pairs = [p for p in csv.DictReader((ROOT/'data/processed/pairwise_master_split.csv').open())
             if p['dimension'] == 'subject_consistency' and p['split'] == 'dev']
    if len(pairs) != 870:
        raise ValueError('development preference population changed')
    result = evaluate_population(pairs, videos, methods, {m: 0. for m in methods})
    write_json(output/'statistics.json', {'protocol': protocol, 'fallback_video_counts': counts, 'development': result})
    for name, r in result['methods'].items():
        print(json.dumps({'method': name, 'accuracy': r['accuracy_full_denominator'],
                          'n_pairs': r['scored_pairs'], 'delta_ci95': r['paired_delta_ci95']}))
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
