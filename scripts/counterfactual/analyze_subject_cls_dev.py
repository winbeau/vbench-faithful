"""Analyze the frozen development-only CLS readout experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    run = json.loads((args.run/'run.json').read_text())
    if (not run.get('completed') or run['protocol_sha256'] != sha256_file(args.run/'protocol.json')
            or run['scores_sha256'] != sha256_file(args.run/'scores.jsonl')):
        raise ValueError('incomplete or changed CLS run')
    natural_path = ROOT/'configs/subject-repair/natural1440_protocol_v2.json'
    if protocol['natural_protocol_sha256'] != sha256_file(natural_path):
        raise ValueError('natural protocol changed')
    natural = json.loads(natural_path.read_text())
    rows = read_jsonl(args.run/'scores.jsonl')
    videos = {r['video_uid']: r for r in rows}
    if len(videos) != 580 or len(rows) != 580 or any(r['split'] != 'dev' for r in rows):
        raise ValueError('expected 580 unique development videos only')
    pair_path = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(pair_path) != natural['pair_labels_sha256']:
        raise ValueError('human preference labels changed')
    pairs = [p for p in csv.DictReader(pair_path.open()) if p['dimension'] == 'subject_consistency' and p['split'] == 'dev']
    if len(pairs) != 870:
        raise ValueError('development pair population changed')
    for pair in pairs:
        if any(videos[pair[k]]['prompt_id'] != pair['prompt_id'] for k in ('video_a_uid', 'video_b_uid')):
            raise ValueError('prompt identity mismatch')
    methods = protocol['methods']
    result = evaluate_population(pairs, videos, methods, {m: protocol['tie_margin'] for m in methods})
    output = new_output(args.output)
    write_json(output/'statistics.json', {'stage': protocol['stage'], 'development': result,
        'runtime_failures': run['runtime_failures'], 'wall_seconds': run['finished_unix']-run['started_unix'],
        'protocol_sha256': run['protocol_sha256'], 'score_sha256': run['scores_sha256'],
        'analysis_source_sha256': sha256_file(Path(__file__))})
    for name, value in result['methods'].items():
        print(json.dumps({'method': name, 'accuracy_full_denominator': value['accuracy_full_denominator'],
            'accuracy_scored_pairs': value['accuracy_scored_pairs'], 'scored_pairs': value['scored_pairs'],
            'delta_on_common_pairs': value['delta_on_common_pairs'], 'ci95_delta': value['paired_delta_ci95']}))


if __name__ == '__main__':
    main()
