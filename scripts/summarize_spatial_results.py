#!/usr/bin/env python3
"""Base-level summaries for frozen Spatial Relationship contracts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

from scipy.stats import spearmanr


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def mean(values):
    return float(statistics.fmean(values)) if values else None


def safe_spearman(xs, ys):
    if len(xs) < 2:
        return None, 'insufficient_n'
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, 'constant_input'
    value = float(spearmanr(xs, ys).statistic)
    return (value, None) if math.isfinite(value) else (None, 'undefined')


def level_name(value):
    return str(value).strip().lower().replace('-', '_')


def coverage_level(value):
    text = level_name(value).replace('coverage_', '').replace('%', '')
    return float(text)


def paired_levels(rows, left_names, right_names):
    left = [row for row in rows if level_name(row['intervention_level']) in left_names]
    right = [row for row in rows if level_name(row['intervention_level']) in right_names]
    return (left[0], right[0]) if len(left) == len(right) == 1 else (None, None)


def summarize_group(base_id, family, variant, rows, detection_conditioned):
    valid = [row for row in rows if row.get('failure_or_abstention') is None and isinstance(row.get('score'), (int, float))]
    if detection_conditioned:
        valid = [row for row in valid if row.get('detected_a') is True and row.get('detected_b') is True]
    result = {'base_id': base_id, 'family': family, 'metric_variant': variant, 'mode': 'detection_conditioned' if detection_conditioned else 'end_to_end', 'input_count': len(rows), 'valid_count': len(valid)}
    if family == 'directional_inversion':
        original, inverted = paired_levels(valid, {'original', 'clean'}, {'hflip', 'horizontal_flip', 'inverted'})
        result.update({'paired_score_gap': None if not original or not inverted else float(original['score']-inverted['score']), 'inversion_pass': None if not original or not inverted else bool(original['score'] > inverted['score'])})
    elif family == 'role_swap':
        correct, swapped = paired_levels(valid, {'original', 'correct'}, {'role_swap', 'swapped'})
        result.update({'correct_vs_swapped_gap': None if not correct or not swapped else float(correct['score']-swapped['score']), 'role_swap_correct': None if not correct or not swapped else bool(correct['score'] > swapped['score'])})
    elif family == 'temporal_persistence':
        pairs = sorted((coverage_level(row['intervention_level']), float(row['score'])) for row in valid)
        rho, reason = safe_spearman([x for x, _ in pairs], [y for _, y in pairs])
        strict = mean([float(right > left) for (_, left), (_, right) in zip(pairs, pairs[1:])]) if len(pairs) >= 2 else None
        result.update({'spearman_rho': rho, 'spearman_reason': reason, 'strict_order_rate': strict, 'coverage_levels': [x for x, _ in pairs]})
    elif family == 'multi_instance':
        clean, distractor = paired_levels(valid, {'original', 'clean', 'target_pair'}, {'multi_instance', 'distractor', 'with_distractor'})
        result.update({'target_pair_success': None if not distractor else bool(distractor['score'] > 0), 'distractor_robustness_gap': None if not clean or not distractor else float(clean['score']-distractor['score'])})
    elif family == 'detection_control':
        result.update({'mean_score': mean([float(row['score']) for row in valid]), 'detected_pair_rate': mean([float(row.get('detected_a') is True and row.get('detected_b') is True) for row in rows])})
    return result


def main():
    parser = argparse.ArgumentParser(description='Summarize Spatial Relationship results using base_id as the independent unit')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    required = {'base_id', 'derived_id', 'intervention_family', 'intervention_level', 'metric_variant', 'score', 'detected_a', 'detected_b', 'failure_or_abstention'}
    for number, row in enumerate(rows, 1):
        missing = required-set(row)
        if missing:
            raise ValueError(f'result row {number} missing fields: {sorted(missing)}')
    if args.validate_only:
        print(json.dumps({'status': 'VALID', 'records': len(rows), 'independent_unit': 'base_id'}))
        return
    groups = defaultdict(list)
    for row in rows:
        groups[(row['base_id'], row['intervention_family'], row['metric_variant'])].append(row)
    per_base = []
    for (base_id, family, variant), items in sorted(groups.items()):
        per_base.append(summarize_group(base_id, family, variant, items, False))
        per_base.append(summarize_group(base_id, family, variant, items, True))
    summaries = defaultdict(list)
    for row in per_base:
        summaries[(row['family'], row['metric_variant'], row['mode'])].append(row)
    summary = []
    for (family, variant, mode), items in sorted(summaries.items()):
        entry = {'family': family, 'metric_variant': variant, 'mode': mode, 'independent_unit': 'base_id', 'base_count': len(items), 'valid_base_count': sum(item['valid_count'] > 0 for item in items)}
        numeric = sorted({key for item in items for key, value in item.items() if key not in {'input_count', 'valid_count'} and isinstance(value, (int, float)) and not isinstance(value, bool)})
        entry['means'] = {key: mean([float(item[key]) for item in items if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool)]) for key in numeric}
        booleans = sorted({key for item in items for key, value in item.items() if isinstance(value, bool)})
        entry['rates'] = {key: mean([float(item[key]) for item in items if isinstance(item.get(key), bool)]) for key in booleans}
        summary.append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'per_base': per_base, 'summary': summary}, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'COMPLETE', 'records': len(rows), 'per_base_rows': len(per_base), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
