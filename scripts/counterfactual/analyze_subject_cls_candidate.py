"""Complete-denominator evaluation of the fixed CLS/fallback candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .analyze_subject_background_auto import make_cases, summarize_cases
from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-run', type=Path, required=True)
    parser.add_argument('--background-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path = ROOT/'configs/subject-repair/cls_candidate_protocol.json'
    protocol = json.loads(protocol_path.read_text()); primary = protocol['primary_method']
    populations, runs = {}, {}
    for kind, directory in [('natural-test', args.test_run), ('background', args.background_run)]:
        run = json.loads((directory/'run.json').read_text())
        if (not run.get('completed') or run['kind'] != kind
                or run['protocol_sha256'] != sha256_file(protocol_path)
                or run['scores_sha256'] != sha256_file(directory/'scores.jsonl')):
            raise ValueError('incomplete or changed candidate run')
        populations[kind] = read_jsonl(directory/'scores.jsonl'); runs[kind] = run
    test = populations['natural-test']; bg = populations['background']
    videos = {r['video_uid']: r for r in test}
    if len(test) != 860 or len(videos) != 860 or any(r['split'] != 'test' for r in test):
        raise ValueError('test cohort mismatch')
    manifest = {r['video_uid']: r for r in read_jsonl(ROOT/'configs/subject-repair/natural1440_manifest.jsonl') if r['split'] == 'test'}
    if set(videos) != set(manifest):
        raise ValueError('test video identity mismatch')
    if len(bg) != 288 or len({r['base']['video_uid'] for r in bg}) != 288:
        raise ValueError('background candidate cohort mismatch')
    expected_bg = {r['video_uid'] for r in read_jsonl(ROOT/'configs/subject-repair/background288_manifest.jsonl')}
    if {r['base']['video_uid'] for r in bg} != expected_bg:
        raise ValueError('background identity mismatch')
    pair_path = ROOT/'data/processed/pairwise_master_split.csv'
    natural = json.loads((ROOT/'configs/subject-repair/natural1440_protocol_v2.json').read_text())
    if sha256_file(pair_path) != natural['pair_labels_sha256']:
        raise ValueError('human pair labels changed')
    pairs = [p for p in csv.DictReader(pair_path.open()) if p['dimension'] == 'subject_consistency' and p['split'] == 'test']
    if len(pairs) != 1290:
        raise ValueError('test pair cohort changed')
    for pair in pairs:
        if any(videos[pair[k]]['prompt_id'] != pair['prompt_id'] for k in ('video_a_uid', 'video_b_uid')):
            raise ValueError('prompt mismatch')
    methods = protocol['controls']+[primary]
    def fallback(value):
        return value.get('scores', {}).get(primary, {}).get('diagnostics', {}).get('fallback_official', False)
    report = {'stage': protocol['stage'], 'protocol_sha256': sha256_file(protocol_path),
        'natural_test': evaluate_population(pairs, videos, methods, {m: protocol['tie_margin'] for m in methods}),
        'test_status_counts': dict(Counter(r['candidate_status'] for r in test)),
        'test_fallback_videos': sum(fallback(r) for r in test),
        'test_pairs_touching_fallback': sum(any(fallback(videos[p[k]]) for k in ('video_a_uid', 'video_b_uid')) for p in pairs),
        'background_status_counts': dict(Counter(r['candidate_status'] for r in bg)),
        'background': {}, 'fallback_by_variant': {}}
    flat = []
    for position in ('full', 'start', 'middle', 'end'):
        report['background'][position] = {}
        for method in methods:
            cases = make_cases(bg, position, method); flat.extend(cases)
            stats = summarize_cases(cases)
            common = [r for r in cases if r['subject_drop'] is not None and r['background_abs_change'] is not None]
            stats['subject_drop_greater_than_matched_background_change'] = {
                'passed': sum(r['subject_drop'] > r['background_abs_change'] for r in common), 'total': len(common)}
            report['background'][position][method] = stats
    keys = ['clean']+[p+'/'+l for p in ('full', 'start', 'middle', 'end') for l in ('background_corrupt','subject_corrupt')]
    for key in keys:
        values = [r['variants'][key] for r in bg if key in r['variants']]
        report['fallback_by_variant'][key] = {'n_variants': len(values), 'n_fallback': sum(fallback(v) for v in values)}
    report['provenance'] = {'input_scores_sha256': {k: r['scores_sha256'] for k, r in runs.items()},
        'analysis_sha256': sha256_file(Path(__file__)),
        'timing_seconds': {k: r['finished_unix']-r['started_unix'] for k, r in runs.items()}}
    output = new_output(args.output)
    write_json(output/'statistics.json', report)
    with (output/'background_per_case.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({'natural_test': report['natural_test'], 'background_full': report['background']['full'],
                      'test_fallback_videos': report['test_fallback_videos'],
                      'test_pairs_touching_fallback': report['test_pairs_touching_fallback'],
                      'fallback_by_variant': report['fallback_by_variant']}, indent=2))


if __name__ == '__main__':
    main()
