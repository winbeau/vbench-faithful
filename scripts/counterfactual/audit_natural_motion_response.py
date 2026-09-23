"""Audit existing Dynamic DEV human-pair overlap; never open reserved media.

Uses only already-scored originals. No tie-margin search, score fitting,
relabeling, CF selection, or human-label inference from another dimension.
Sparse overlap is reported as sparse overlap, not natural-motion validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .static_jitter import digest


def cluster_interval(rows, values):
    if not rows:
        return None
    prompts = sorted({r['prompt_id'] for r in rows})
    sums = np.array([sum(v for r, v in zip(rows, values) if r['prompt_id'] == p) for p in prompts])
    sizes = np.array([sum(r['prompt_id'] == p for r in rows) for p in prompts])
    indices = np.random.default_rng(20260923).integers(len(prompts), size=(20000, len(prompts)))
    bootstrap = sums[indices].sum(1) / sizes[indices].sum(1)
    return np.quantile(bootstrap, [.025, .975]).tolist()


def evaluate_pairs(rows, scores):
    # Filter before accessing labels. Test and other-dimension labels are not
    # consulted, even if their media UID happens to occur in this cohort.
    dev = [r for r in rows if r['dimension'] == 'dynamics_degree' and r['split'] == 'dev']
    for values in scores.values():
        if set(values) != {'origin', 'repair'} or not all(np.isfinite(v) for v in values.values()):
            raise ValueError('finite Origin and Repair scores required for every original')
    covered, identities = [], set()
    for row in dev:
        a, b = row['video_a_uid'], row['video_b_uid']
        if a not in scores or b not in scores:
            continue
        key = (row['prompt_id'], *sorted((a, b)))
        if a == b or key in identities:
            raise ValueError('duplicate or self pair')
        identities.add(key)
        label = float(row['human_label'])
        if label not in (0., .5, 1.) or row['label_source'] != 'official_human_anno':
            raise ValueError('official Dynamic human preference required')
        result = {k: row[k] for k in ('prompt_id', 'video_a_uid', 'video_b_uid', 'label_source')}
        result['human_label'] = label
        for method in ('origin', 'repair'):
            gap = scores[a][method]-scores[b][method]
            prediction = .5 if gap == 0 else float(gap > 0)
            result[method] = {'a': scores[a][method], 'b': scores[b][method],
                              'gap': gap, 'prediction': prediction,
                              'strict_correct': prediction == label}
        covered.append(result)
    ordered = [r for r in covered if r['human_label'] != .5]
    ties = [r for r in covered if r['human_label'] == .5]
    methods = {}
    for method in ('origin', 'repair'):
        strict = [float(r[method]['strict_correct']) for r in ordered]
        gaps = [abs(r[method]['gap']) for r in ties]
        methods[method] = {
            'ordered_strict_correct': int(sum(strict)),
            'ordered_strict_agreement': float(np.mean(strict)) if strict else None,
            'ordered_strict_ci95_prompt_cluster': cluster_interval(ordered, strict),
            'ordered_metric_ties': sum(r[method]['prediction'] == .5 for r in ordered),
            'human_tie_exact_score_ties': sum(r[method]['prediction'] == .5 for r in ties),
            'human_tie_absolute_gap_mean': float(np.mean(gaps)) if gaps else None,
            'human_tie_absolute_gap_max': max(gaps) if gaps else None,
            'all_pairs_exact_agreement': float(np.mean([r[method]['strict_correct'] for r in covered])) if covered else None,
        }
    deltas = [float(r['repair']['strict_correct'])-float(r['origin']['strict_correct']) for r in ordered]
    mp4 = [r for r in dev if r['video_a_path'].endswith('.mp4') and r['video_b_path'].endswith('.mp4')]
    return {'status': 'development_overlap_audit_not_motion_validation',
            'scored_originals': len(scores), 'total_dynamic_dev_pairs': len(dev),
            'total_dynamic_dev_mp4_pairs': len(mp4), 'covered_pairs': len(covered),
            'pair_coverage_all_dev': len(covered)/len(dev) if dev else None,
            'covered_prompts': len({r['prompt_id'] for r in covered}),
            'ordered_pairs': len(ordered), 'ordered_prompts': len({r['prompt_id'] for r in ordered}),
            'human_tie_pairs': len(ties), 'methods': methods,
            'ordered_repair_minus_origin': float(np.mean(deltas)) if deltas else None,
            'ordered_delta_ci95_prompt_cluster': cluster_interval(ordered, deltas),
            'pairs': covered, 'human_label_counts': dict(Counter(str(r['human_label']) for r in covered)),
            'tie_margin': 0, 'tie_margin_fitted': False,
            'uncertainty': '20000 prompt-cluster bootstrap draws; sparse posthoc DEV overlap, not an independent population estimate',
            'interpretation': 'strict exact-tie agreement is unsuitable as a lone headline for continuous scores against binary/tied labels; ordered and tied cases are separate',
            'true_motion_strength_validation': 'INCOMPLETE; preference labels are not measured speeds and few ordered pairs cannot establish all motion types',
            'holdout': 'NOT RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--pairs', default='data/processed/pairwise_master_split.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    score_path, pair_path, output = Path(args.scores), Path(args.pairs), Path(args.output)
    source_paths = [Path('configs/dynamic-static-jitter')/name for name in
                    ('sources.local-texture-dev32-mp4-v1.jsonl', 'sources.natural63-calibration-v1.jsonl')]
    expected = {}
    for path in source_paths:
        for row in map(json.loads, path.read_text().splitlines()):
            uid = row['video_uid']
            if uid in expected or row['split'] != 'dev' or row['dimension'] != 'dynamics_degree':
                raise ValueError('unique predeclared Dynamic DEV originals required')
            expected[uid] = row
    if len(expected) != 95:
        raise ValueError('all 95 declared originals required')
    scores = {}
    for row in map(json.loads, score_path.read_text().splitlines()):
        if row['family'] != 'original':
            continue
        uid = row['base_id']
        if uid not in expected or uid in scores or row['status'] != 'finished' or row['prompt_id'] != expected[uid]['prompt_id']:
            raise ValueError('unknown, duplicate, mismatched or failed original')
        scores[uid] = {m: row[m]['score'] for m in ('origin', 'repair')}
    if set(scores) != set(expected):
        raise ValueError('missing declared original')
    with pair_path.open(newline='') as handle:
        result = evaluate_pairs(csv.DictReader(handle), scores)
    result['provenance'] = {'score_file': str(score_path), 'score_sha256': digest(score_path),
        'pair_file': str(pair_path), 'pair_sha256': digest(pair_path),
        'source_sha256': {str(p): digest(p) for p in source_paths}, 'script_sha256': digest(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ('pairs', 'provenance')}, indent=2))


if __name__ == '__main__':
    main()
