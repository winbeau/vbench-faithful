#!/usr/bin/env python3
"""Score frozen small-model, LoRA, interface, and evidence ablations without dropping failures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile.generalization import behavioral_score, correct, semantic_key, TASKS
from vbench_prompts_compile.action_repair import compile_action
from vbench_prompts_compile.spatial_repair import entity_name
from vbench_prompts_compile.sources import load_k400
from predict_ablation import jobs_for


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interface(row, value, vocab):
    if semantic_key(row['task'], value) is None:
        return None
    if row['task'] == 'action':
        # This is the frozen, pre-audit v2 comparison. Later bug fixes must
        # have their own labeled regression report, never rewrite this table.
        return compile_action(row['input']['prompt'], value, vocab, scope_guard=False)['target']
    if row['task'] == 'spatial':
        return {'relationships': [{**t, 'subject': entity_name(t['subject']), 'object': entity_name(t['object'])}
                                  for t in value['relationships']]}
    # Objects backend temporal filtering is evaluated over real video caches,
    # not applied to a text target; Scene keeps its three-label contract.
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/ablation-v1/eval.jsonl')
    parser.add_argument('--prediction-dir', default='output/ablation-v1')
    parser.add_argument('--out', default='docs/deterministic/ablation-v1')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    manifest = json.loads(Path(args.data).with_suffix('.manifest.json').read_text())
    if sha(args.data) != manifest['data_sha256']:
        raise SystemExit('evaluation data hash mismatch')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = load_k400()
    reports, accuracy_rows, group_rows, caption_rows, error_rows = {}, [], [], [], []
    integrity = {}
    expected = {j['key'] for j in jobs_for(rows)}
    for size in ('0.6', '8'):
        pred_path = Path(args.prediction_dir) / f'predictions-{size}b.jsonl'
        pred = {}
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            if r['key'] in pred:
                raise SystemExit('duplicate prediction key')
            pred[r['key']] = r
        if set(pred) != expected:
            raise SystemExit(f'{size}B run is incomplete: missing={len(expected - pred.keys())}, extra={len(pred.keys() - expected)}')
        plan_path = pred_path.with_suffix('.plan.json')
        if json.loads(plan_path.read_text())['data_sha256'] != sha(args.data):
            raise SystemExit('prediction data mismatch')
        run_report = json.loads(pred_path.with_suffix('.report.json').read_text())
        if (not run_report['complete'] or run_report['predictions_sha256'] != sha(pred_path)
                or run_report['plan_sha256'] != sha(plan_path)):
            raise SystemExit('prediction completion/hash report mismatch')
        integrity[size] = {'predictions_sha256': sha(pred_path), 'plan_sha256': sha(plan_path), 'rows': len(pred)}
        for mode in ('base', 'sft'):
            for codec in ('raw', 'interface'):
                name = f'{size}B/{mode}/{codec}'
                predictions = {}
                for row in rows:
                    result = pred[f'{row["id"]}/{mode}/intact']
                    value = result['value']
                    if codec == 'interface':
                        value = interface(row, value, vocab)
                    predictions[row['id']] = value
                    if not correct(row['task'], value, row['target']):
                        error_rows.append({'system': name, 'id': row['id'], 'task': row['task'],
                                           'source': row['source'], 'variant': row['variant'], 'errors': result['errors'],
                                           'expected': row['target'], 'prediction': value})
                reports[name] = behavioral_score(rows, predictions)
                for task in TASKS:
                    cohorts = sorted({(r['source'], r['variant']) for r in rows if r['task'] == task})
                    cohorts += [('engineered', 'all')]
                    for source, variant in cohorts:
                        cohort = [r for r in rows if r['task'] == task and r['source'] == source
                                  and (variant == 'all' or r['variant'] == variant)]
                        valid = [semantic_key(task, predictions.get(r['id'])) is not None for r in cohort]
                        legal = [ok and (task != 'action' or all(v in (*vocab.labels, 'other') for v in predictions[r['id']]['actions']))
                                 for r, ok in zip(cohort, valid)]
                        hits = [correct(task, predictions.get(r['id']), r['target']) for r in cohort]
                        accuracy_rows.append({'system': name, 'task': task, 'source': source, 'variant': variant,
                                              'n': len(cohort), 'families': len({r['family'] for r in cohort}),
                                              'correct': sum(hits), 'accuracy': sum(hits) / len(cohort),
                                              'valid_schema': sum(valid), 'schema_coverage': sum(valid) / len(cohort),
                                              'label_space_coverage': sum(legal) / len(cohort)})
                for group, result in reports[name]['groups'].items():
                    group_rows.append({'system': name, 'group': group, 'n': result['families'],
                                       'pair_successes': result['pair_successes'], 'cc': result['cc'],
                                       'simultaneous_lower95': result['simultaneous_lower95'],
                                       'accuracy_delta': result['paired_accuracy_delta']['mean'],
                                       'delta_ci95': json.dumps(result['paired_accuracy_delta']['ci95']),
                                       'noninferior_at_minus_003': result['paired_accuracy_delta']['noninferior_at_minus_003']})
            for source in ('engineered', 'snli_test_locations'):
                cohort = [r for r in rows if r['task'] == 'scene' and r['source'] == source
                          and (r['variant'] == 'canonical' or source == 'snli_test_locations')]
                for cap_mode in ('intact', 'missing', 'shuffled'):
                    hits = [correct('scene', pred[f'{r["id"]}/{mode}/{cap_mode}']['value'], r['target']) for r in cohort]
                    per_label = {}
                    for label in ('supported', 'contradicted', 'insufficient'):
                        indices = [i for i, r in enumerate(cohort) if r['target'] == label]
                        per_label[label] = sum(hits[i] for i in indices) / len(indices) if indices else None
                    caption_rows.append({'size': size, 'mode': mode, 'source': source, 'caption': cap_mode,
                                         'n': len(cohort), 'accuracy_against_intact_labels': sum(hits)/len(hits),
                                         'macro_recall': sum(per_label.values()) / 3,
                                         'insufficient_prediction_rate': sum(pred[f'{r["id"]}/{mode}/{cap_mode}']['value'] == 'insufficient' for r in cohort) / len(cohort),
                                         **per_label})
    engineering = [r for r in rows if r['source'] == 'engineered']
    canonical = {r['family']: r['target'] for r in engineering if r['variant'] == 'canonical'}
    control = {}
    for label in ('oracle', 'copy_oracle_canonical', 'constant_empty'):
        pred = {}
        for r in engineering:
            value = r['target'] if label == 'oracle' else canonical[r['family']]
            if label == 'constant_empty':
                value = 'insufficient' if r['task'] == 'scene' else {{'spatial': 'relationships', 'action': 'actions', 'objects': 'entities'}[r['task']]: []}
            pred[r['id']] = value
        control[label] = behavioral_score(rows, pred)
    assert control['oracle']['wg_cc'] == 1
    assert control['copy_oracle_canonical']['wg_cc'] == control['constant_empty']['wg_cc'] == 0
    summary = {'protocol_sha256': manifest['protocol_sha256'], 'evaluation_sha256': sha(args.data),
               'manifest': manifest, 'integrity': integrity, 'systems': reports, 'anti_shortcut_controls': control,
               'interpretation': ['WG-CC is a behavioral diagnostic, not proof against arbitrary overfitting.',
                                  'Confidence bounds assume independent families despite shared authored templates.',
                                  'Scene caption removal scored against intact labels is an information-use ablation.',
                                  'Raw/interface target correctness is not end-to-end detector/video accuracy.',
                                  'Single seed; do not select checkpoints, aliases, or thresholds using this test.']}
    (out / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(out / 'accuracy.csv', accuracy_rows)
    write_csv(out / 'groups.csv', group_rows)
    write_csv(out / 'scene-evidence.csv', caption_rows)
    failures = Path(args.prediction_dir) / 'errors.jsonl'
    failures.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in error_rows))
    print(json.dumps({name: {k: v for k, v in r.items() if k != 'groups'} for name, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
