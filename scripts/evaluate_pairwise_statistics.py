#!/usr/bin/env python3
"""Frozen-E0 pairwise statistics for official or repair per-video scores.

This module never runs evaluator inference.  The formulas are the reconstructed
baseline definitions; ``--predictions-root`` changes only the score source.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from collections import defaultdict
from pathlib import Path
from scipy.stats import kendalltau, pearsonr

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / 'results/e0/raw_official_scores'
DIMS = {
    'dynamic_degree': 'dynamics_degree',
    'subject_consistency': 'subject_consistency',
    'human_action': 'human_action',
    'spatial_relationship': 'spatial_relationship',
    'motion_smoothness': 'motion_smoothness',
}


def rows(path: Path):
    with path.open(newline='', encoding='utf8') as handle:
        return list(csv.DictReader(handle))


def outcome(difference, delta=0.0):
    return .5 if abs(difference) <= delta else (1.0 if difference > 0 else 0.0)


def _finite_score(row, repair):
    status_key, score_key = ('repair_status', 'repair_score') if repair else ('status', 'score')
    if row.get(status_key) not in ({'succeeded_scalar'} if repair else {'success'}):
        return None
    raw = row.get(score_key)
    if raw in (None, ''):
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def build(alias, predictions_root=None, scores_root=None):
    official_dimension = DIMS[alias]
    repair = predictions_root is not None
    official_root = Path(scores_root) if scores_root is not None else SCORES
    result_path = (Path(predictions_root) / alias / 'predictions.csv' if repair
                   else official_root / official_dimension / 'results.csv')
    results = {row['video_uid']: row for row in rows(result_path)}
    all_pairs = [row for row in rows(ROOT / 'data/processed/pairwise_master_split.csv')
                 if row['dimension'] == official_dimension]
    valid = []
    for pair in all_pairs:
        a, b = results.get(pair['video_a_uid']), results.get(pair['video_b_uid'])
        if a is None or b is None:
            continue
        score_a, score_b = _finite_score(a, repair), _finite_score(b, repair)
        if score_a is not None and score_b is not None:
            valid.append((pair, score_a - score_b))
    return all_pairs, valid


def accuracy(items, delta):
    return sum(outcome(difference, delta) == float(pair['human_label'])
               for pair, difference in items) / len(items)


def calibrate(dev):
    candidates = sorted({0.0} | {abs(difference) for _, difference in dev})
    best = max(accuracy(dev, candidate) for candidate in candidates)
    return min(candidate for candidate in candidates
               if abs(accuracy(dev, candidate) - best) < 1e-15)


def model_pearson(valid):
    metric, human = defaultdict(list), defaultdict(list)
    for pair, difference in valid:
        prediction = outcome(difference)
        metric[pair['model_a']].append(prediction)
        metric[pair['model_b']].append(1 - prediction)
        label = float(pair['human_label'])
        human[pair['model_a']].append(label)
        human[pair['model_b']].append(1 - label)
    models = sorted(metric)
    metric_means = [sum(metric[model]) / len(metric[model]) for model in models]
    human_means = [sum(human[model]) / len(human[model]) for model in models]
    return float(pearsonr(metric_means, human_means).statistic), models, metric_means, human_means


def ci(items, delta, seed, iterations):
    rng, count, values = random.Random(seed), len(items), []
    for _ in range(iterations):
        values.append(accuracy([items[rng.randrange(count)] for _ in range(count)], delta))
    values.sort()
    return [values[int(.025 * (iterations - 1))], values[int(.975 * (iterations - 1))]]


def calculate(iterations=2000, seed=2026, predictions_root=None, dimensions=None, scores_root=None):
    requested = dimensions or list(DIMS)
    output = []
    for alias in requested:
        total, valid = build(alias, predictions_root, scores_root)
        dev = [item for item in valid if item[0]['split'] == 'dev']
        test = [item for item in valid if item[0]['split'] == 'test']
        if not dev or not test:
            raise RuntimeError(f'{alias}: no valid dev/test pairs in score source')
        delta = calibrate(dev)
        pearson, models, metric_means, human_means = model_pearson(valid)
        output.append({
            'dimension': alias, 'official_dimension': DIMS[alias],
            'dev_acc0': accuracy(dev, 0), 'test_acc0': accuracy(test, 0),
            'tie_margin_delta': delta, 'test_tie_aware_acc': accuracy(test, delta),
            'kendall_tau_b': float(kendalltau([float(pair['human_label']) for pair, _ in test],
                                               [difference for _, difference in test]).statistic),
            'coverage': len(valid) / len(total), 'valid_pairs': len(valid), 'total_pairs': len(total),
            'model_level_pearson_n4': pearson, 'models': models,
            'metric_outcome_means': metric_means, 'human_outcome_means': human_means,
            'test_acc0_ci95': ci(test, 0, seed, iterations),
            'tie_aware_ci95': ci(test, delta, seed + 1, iterations),
            'bootstrap_seed': seed, 'bootstrap_iterations': iterations,
        })
    return output


FIELDS = ['dimension', 'official_dimension', 'dev_acc0', 'test_acc0', 'tie_margin_delta',
          'test_tie_aware_acc', 'kendall_tau_b', 'coverage', 'valid_pairs', 'total_pairs',
          'model_level_pearson_n4', 'test_acc0_ci95', 'tie_aware_ci95',
          'bootstrap_seed', 'bootstrap_iterations']


def write_metrics(data, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{key: (json.dumps(value) if isinstance(value, list) else value)
                          for key, value in record.items() if key in FIELDS} for record in data])


def write_comparison(official_path, repair_path, csv_path, markdown_path):
    official = {row['dimension']: row for row in rows(official_path)}
    repair = {row['dimension']: row for row in rows(repair_path)}
    fields = ['Dimension', 'Official Pearson', 'Repair Pearson', 'Delta Pearson',
              'Official Test Acc0', 'Repair Test Acc0', 'Delta Acc0',
              'Official Tie-aware Acc', 'Repair Tie-aware Acc', 'Delta Tie-aware',
              'Official Kendall tau-b', 'Repair Kendall tau-b', 'Delta Kendall',
              'Official Coverage', 'Repair Coverage', 'Delta Coverage',
              'Official Test Acc0 CI', 'Repair Test Acc0 CI',
              'Official Tie-aware CI', 'Repair Tie-aware CI']
    table = []
    mappings = [('Pearson', 'model_level_pearson_n4'), ('Test Acc0', 'test_acc0'),
                ('Tie-aware Acc', 'test_tie_aware_acc'), ('Kendall tau-b', 'kendall_tau_b'),
                ('Coverage', 'coverage')]
    for dimension in DIMS:
        base, repaired = official[dimension], repair[dimension]
        row = {'Dimension': dimension}
        for label, key in mappings:
            row[f'Official {label}'] = base[key]
            row[f'Repair {label}'] = repaired[key]
            row[f'Delta {"Acc0" if label == "Test Acc0" else "Tie-aware" if label == "Tie-aware Acc" else "Kendall" if label == "Kendall tau-b" else label}'] = float(repaired[key]) - float(base[key])
        row['Official Test Acc0 CI'] = base['test_acc0_ci95']
        row['Repair Test Acc0 CI'] = repaired['test_acc0_ci95']
        row['Official Tie-aware CI'] = base['tie_aware_ci95']
        row['Repair Tie-aware CI'] = repaired['tie_aware_ci95']
        table.append(row)
    with csv_path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    lines = ['# Official vs Repair', '', '| ' + ' | '.join(fields) + ' |',
             '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    for row in table:
        lines.append('| ' + ' | '.join(str(row[field]) for field in fields) + ' |')
    markdown_path.write_text('\n'.join(lines) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-root', type=Path,
                        help='Root containing <dimension>/predictions.csv; omit for frozen official results.')
    parser.add_argument('--official-scores-root', type=Path,
                        help='Alternative read-only root containing <dimension>/results.csv.')
    parser.add_argument('--dimension', choices=DIMS, action='append',
                        help='Compute only this dimension; repeatable.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/official_repair')
    parser.add_argument('--bootstrap-iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--official-metrics', type=Path)
    parser.add_argument('--comparison-csv', type=Path)
    parser.add_argument('--comparison-md', type=Path)
    args = parser.parse_args()
    data = calculate(args.bootstrap_iterations, args.seed, args.predictions_root, args.dimension, args.official_scores_root)
    output = args.output or args.output_dir / ('repair_metrics.csv' if args.predictions_root else 'reconstructed_baseline_metrics.csv')
    write_metrics(data, output)
    if args.official_metrics or args.comparison_csv or args.comparison_md:
        if not (args.official_metrics and args.comparison_csv and args.comparison_md):
            parser.error('comparison requires --official-metrics, --comparison-csv, and --comparison-md')
        write_comparison(args.official_metrics, output, args.comparison_csv, args.comparison_md)
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
