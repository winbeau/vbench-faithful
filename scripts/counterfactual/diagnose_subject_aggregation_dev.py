"""Post-hoc development-only check of adjacency and missing-frame treatment.

Uses already recorded pair means, without changing localization or representation.
These are exploratory development results, never a new held-out test claim.
"""
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
    parser.add_argument('--natural-analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.natural_analysis/'scores_merged.jsonl'
    # Discard test records before deriving any new score or diagnostic.
    videos = {r['video_uid']: r for r in read_jsonl(path) if r['split'] == 'dev'}
    pairs = [p for p in csv.DictReader((ROOT/'data/processed/pairwise_master_split.csv').open())
             if p['dimension'] == 'subject_consistency' and p['split'] == 'dev']
    if len(videos) != 580 or len(pairs) != 870:
        raise ValueError('expected the unchanged official development partition')
    methods = ['official', 'aggregation']
    for mode in ('post_pool', 'preencode_full', 'preencode_crop'):
        for policy in ('zero', 'exclude'):
            original = mode+'_'+policy
            derived = original+'_local_global'
            methods.extend((original, derived))
            for row in videos.values():
                value = score(row, original)
                adjacent = (row['scores'][original].get('diagnostics') or {}).get('adjacent_score')
                valid = value is not None and adjacent is not None
                row['scores'][derived] = {'status': 'succeeded' if valid else 'failed',
                    'score': .5*(value+adjacent) if valid else None}
    output = new_output(args.output)
    protocol = {'stage': 'post_hoc_development_mechanism_diagnostic', 'split': 'dev only; test discarded before derivation',
        'methods': methods, 'derived_formula': '0.5 * recorded adjacent_score + 0.5 * recorded all_pairs_score',
        'tie_margin': 0., 'parameters_fitted': False, 'representation_and_masks': 'Unchanged cached diagnostics',
        'missing_policy': 'Preserve original zero/exclude; exclude adjacency with no valid adjacent pairs stays zero, as recorded.',
        'primary_use': 'Separate aggregation and missing-policy effects; do not choose a winner on exposed test labels.',
        'input_sha256': sha256_file(path), 'source_sha256': sha256_file(Path(__file__))}
    write_json(output/'protocol.json', protocol)
    result = evaluate_population(pairs, videos, methods, {m: 0. for m in methods})
    write_json(output/'statistics.json', {'protocol': protocol, 'development': result})
    for name, r in result['methods'].items():
        print(json.dumps({'method': name, 'dev_accuracy_full_denominator': r['accuracy_full_denominator'],
                          'scored_pairs': r['scored_pairs'], 'delta_on_common_pairs': r['delta_on_common_pairs']}))


if __name__ == '__main__':
    main()
