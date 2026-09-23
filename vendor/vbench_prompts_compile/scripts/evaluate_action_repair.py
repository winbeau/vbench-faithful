#!/usr/bin/env python3
"""Audit Action synonym repair, raw-output ablations and full-matrix regression."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J, records as R
from vbench_prompts_compile.action_lexicon import ACTION_SYNONYMS
from vbench_prompts_compile.action_repair import ActionNormalizer, LEXICON_SHA256
from vbench_prompts_compile.experiments import cluster_summary, text_key
from vbench_prompts_compile.inference import parse_output
from vbench_prompts_compile.sources import load_k400
from score_matrix import score_one
from render_matrix_tables import table, ci_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError('duplicate record: ' + str(k))
        result[k] = row
    return result


def regress(previous_path, current_path):
    unchanged, previous_action, current_action = Counter(), [], []
    total = 0
    with Path(previous_path).open() as old, Path(current_path).open() as new:
        for a, b in itertools.zip_longest(old, new):
            if a is None or b is None:
                raise ValueError('changed matrix denominator')
            a, b = json.loads(a), json.loads(b)
            if (a['sample_id'], a['scheme']) != (b['sample_id'], b['scheme']):
                raise ValueError('matrix observation identity/order changed')
            total += 1
            if a['task'] != 'action' or a['scheme'] != 'Repair-model':
                if a != b:
                    raise ValueError('untargeted record changed: ' + str((a['sample_id'], a['scheme'])))
                unchanged[a['task'] + '/' + a['scheme']] += 1
            if a['task'] == 'action':
                previous_action.append(a)
                current_action.append(b)
    return {'total_rows': total, 'untouched_rows_exact': dict(unchanged),
            'updated_action_model_rows': sum(r['scheme'] == 'Repair-model' for r in current_action)}, previous_action, current_action


def unique_transforms(matrix):
    result = {}
    for row in matrix:
        key = (row['transform'], row['prompt'])
        if key in result and result[key].get('expected_actions') != row.get('expected_actions'):
            raise ValueError('conflicting protocol target for same prompt')
        result[key] = row
    return result


def expected_actions(row, vocab):
    if row['transform'] == 'identity':
        return [vocab.resolve(row['official_target'])]
    return row['expected_actions']


def target_equal(value, expected):
    return isinstance(value, dict) and set(value.get('actions', [])) == set(expected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'previous-paired', 'paired', 'score-report', 'predictions', 'repaired-predictions', 'cache', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--vocabulary', default=str(ROOT / 'data/raw/k400/labels.json'))
    p.add_argument('--baseline-commit', default='ea9d500f21d800507ef220b6e95fee672d0e70f1')
    args = p.parse_args(argv)
    regression, before, after = regress(args.previous_paired, args.paired)
    matrix = [r for r in J.read_jsonl(Path(args.matrix)) if r['task'] == 'action']
    unique = unique_transforms(matrix)
    old_predictions = indexed(J.read_jsonl(Path(args.predictions)), lambda r: r['request_id'])
    repaired = indexed(J.read_jsonl(Path(args.repaired_predictions)), lambda r: r['request_id'])
    if set(old_predictions) != set(repaired):
        raise ValueError('repaired predictions changed the request denominator')
    vocab = load_k400(Path(args.vocabulary))
    normalizer = ActionNormalizer(vocab)
    canonical = {v['label']: v['id'] for v in vocab.entries}
    if len(canonical) != 400 or not all(normalizer.resolve(v) == v for v in canonical):
        raise ValueError('K400 names/IDs no longer roundtrip')
    # Verify this is the pre-existing lexicon, not aliases added after observing
    # each failed prediction. Its closed-domain status is still explicit.
    prior_source = subprocess.check_output(['git', '-C', str(ROOT), 'show', args.baseline_commit + ':scripts/build_matrix_plan.py'], text=True)
    node = next(n for n in ast.parse(prior_source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ACTION_SYNONYMS' for t in n.targets))
    if ast.literal_eval(node.value) != ACTION_SYNONYMS:
        raise ValueError('declared synonym lexicon changed')
    variants = {'legacy_model': {}, 'output_normalization_only': {}, 'repair_v2': {}}
    for key, row in old_predictions.items():
        if row['raw'] != repaired[key]['raw']:
            raise ValueError('raw model generation changed')
        raw, _ = parse_output('action', row['raw'])
        variants['legacy_model'][key] = row['target']
        variants['output_normalization_only'][key] = normalizer.normalize_output(raw)
        variants['repair_v2'][key] = repaired[key]['target']
        if normalizer.compile(row['prompt'], raw)['target'] != repaired[key]['target']:
            raise ValueError('replay no longer matches live postprocessor')
    cache = indexed(J.read_jsonl(Path(args.cache)), lambda r: r['relative_path'])
    # Scoring from stored legacy targets plus current prompt must agree with a
    # fresh postprocess of preserved raw generations, including coverage/OOV.
    current_model = {(r['sample_id'], r['scheme']): r for r in after if r['scheme'] == 'Repair-model'}
    checked = 0
    for row in matrix:
        live = score_one(row, 'Repair-model', cache[row['relative_path']], variants['repair_v2'], vocab,
                         action_interface='legacy-model')
        recorded = current_model[row['sample_id'], 'Repair-model']
        for key, destination in [('score', 'transformed_score'), ('coverage', 'transformed_coverage'),
                                 ('abstention', 'transformed_abstention'), ('target', 'transformed_target')]:
            if live[key] != recorded[destination]:
                raise ValueError('live/replayed/scored Action mismatch: ' + row['sample_id'])
        checked += 1
    regression['live_replay_scoring_rows_exact'] = checked
    parsing_rows, parsing = [], []
    for variant, predictions in variants.items():
        for (transform, prompt), row in unique.items():
            expected = expected_actions(row, vocab)
            value = predictions[text_key('action', prompt)]
            parsing_rows.append({'variant': variant, 'transform': transform, 'prompt': prompt,
                                 'family': 'action-text:' + text_key('action', prompt),
                                 'correct': target_equal(value, expected), 'target': value, 'expected': expected})
        for transform in sorted({r['transform'] for r in matrix}):
            rows = [r for r in parsing_rows if r['variant'] == variant and r['transform'] == transform]
            parsing.append({'variant': variant, 'transform': transform, 'correct': cluster_summary(rows, 'correct')})
    consistency = []
    for variant, predictions in variants.items():
        rows = []
        for (transform, prompt), row in unique.items():
            if transform != 'synonym':
                continue
            original = predictions[text_key('action', row['original_prompt'])]
            changed = predictions[text_key('action', prompt)]
            expected = expected_actions(row, vocab)
            rows.append({'family': row['family'], 'correct_and_consistent': target_equal(original, expected) and target_equal(changed, expected),
                         'target_unchanged': original == changed})
        consistency.append({'variant': variant, 'correct_and_consistent': cluster_summary(rows, 'correct_and_consistent'),
                            'target_unchanged': cluster_summary(rows, 'target_unchanged')})
    ablations = []
    original_by_path = {r['relative_path']: r for r in matrix if r['transform'] == 'identity'}
    for variant, predictions in variants.items():
        values = []
        for row in matrix:
            if row['transform'] != 'synonym':
                continue
            evidence = cache[row['relative_path']]
            a = score_one(original_by_path[row['relative_path']], 'Repair-model', evidence, predictions, vocab, action_interface='legacy-model')
            b = score_one(row, 'Repair-model', evidence, predictions, vocab, action_interface='legacy-model')
            values.append({'family': row['family'], 'original': a['score'], 'synonym': b['score'],
                           'delta': b['score'] - a['score'], 'exact_invariance': a['coverage'] == b['coverage'] == 1 and
                               a['abstention'] < 1 and b['abstention'] < 1 and a['score'] == b['score']})
        ablations.append({'variant': variant, **{k: cluster_summary(values, k) for k in ('original', 'synonym', 'delta', 'exact_invariance')}})
    failures = Counter()
    for (transform, prompt), row in unique.items():
        if transform != 'synonym':
            continue
        old = old_predictions[text_key('action', prompt)]
        if old['target'] == {'actions': ['other']}:
            failures['legacy_other'] += 1
            raw, _ = parse_output('action', old['raw'])
            failures['raw_other' if raw == {'actions': ['other']} else 'non_other_raw_rejected_by_legacy_canonicalizer'] += 1
    score_report = json.loads(Path(args.score_report).read_text())
    if not score_report['execution_complete'] or score_report.get('action_interface') != 'repair-v2':
        raise ValueError('full matrix did not use the repaired interface')
    paths = [getattr(args, v) for v in ('matrix', 'previous_paired', 'paired', 'score_report', 'predictions', 'repaired_predictions', 'cache', 'vocabulary')]
    sources = ['scripts/evaluate_action_repair.py', 'scripts/repair_action_predictions.py', 'scripts/score_matrix.py',
               'scripts/render_matrix_tables.py', 'src/vbench_prompts_compile/action_repair.py',
               'src/vbench_prompts_compile/action_lexicon.py', 'src/vbench_prompts_compile/inference.py',
               'src/vbench_prompts_compile/official_replay.py', 'src/vbench_prompts_compile/experiments.py',
               'src/vbench_prompts_compile/sources.py', 'src/vbench_prompts_compile/records.py']
    report = {'action_interface': 'repair-v2', 'model': 'unchanged Action v9 final plus declared text contract',
              'execution_complete': True, 'legacy_failure_attribution': dict(failures), 'regression': regression,
              'declared_lexicon_unchanged_from_commit': args.baseline_commit, 'lexicon_sha256': LEXICON_SHA256,
              'k400_roundtrip': len(canonical), 'unique_requests': len(old_predictions),
              'resolution_methods': dict(Counter(r['postprocess']['method'] for r in repaired.values())),
              'parsing': parsing, 'synonym_consistency': consistency, 'score_ablations': ablations,
              'summary': [r for r in score_report['summary'] if r['task'] == 'action'],
              'input_sha256': {path: sha(path) for path in paths},
              'source_sha256': {path: sha(ROOT / path) for path in sources},
              'new_training_steps': 0, 'new_model_generations': 0, 'new_api_calls': 0,
              'notes': ['Closed 60-pair declared synonym contract, not an unseen-synonym generalization test.',
                        'Repair-model means v9 plus explicit deterministic interface; native/legacy model remains separately reported.',
                        'No visual predictions or transformed/original-pair identity enter the normalizer.',
                        'Other scores zero. Repeated OOV text is one unique protocol input, not 1200 independent samples.',
                        '2000 family bootstrap draws, seed 20260919. Model weights, UMT evidence and .85 threshold are unchanged.']}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(out / 'parsing-audit.jsonl', parsing_rows)
    report['output_sha256'] = {'parsing-audit.jsonl': sha(out / 'parsing-audit.jsonl')}
    J.atomic_json(out / 'report.json', report)
    summaries = [{'Scheme': r['scheme'], 'Transform': r['transform'], 'N': r['eligible'], 'Families': r['families'],
                  'Original': r['original_score']['estimate'], 'Transformed': r['transformed_score']['estimate'],
                  'Delta (CI)': ci_text(r['paired_delta']), 'Invariance within 0.03 (CI)': ci_text(r['invariance_success'])}
                 for r in report['summary']]
    table(out, 'scores', summaries, list(summaries[0]))
    summaries = [{'Variant': r['variant'], 'Transform': r['transform'], 'Unique prompts': r['correct']['n'],
                  'Target correct (CI)': ci_text(r['correct'])} for r in parsing]
    table(out, 'parsing', summaries, list(summaries[0]))
    summaries = [{'Variant': r['variant'], 'N': r['original']['n'], 'Families': r['original']['families'],
                  'Original': r['original']['estimate'], 'Synonym': r['synonym']['estimate'],
                  'Delta (CI)': ci_text(r['delta']), 'Exact invariance (CI)': ci_text(r['exact_invariance'])} for r in ablations]
    table(out, 'ablations', summaries, list(summaries[0]))
    summaries = [{'Variant': r['variant'], 'Unique pairs': r['correct_and_consistent']['n'],
                  'Correct and consistent (CI)': ci_text(r['correct_and_consistent']),
                  'Same target (CI)': ci_text(r['target_unchanged'])} for r in consistency]
    table(out, 'consistency', summaries, list(summaries[0]))
    print(json.dumps({key: report[key] for key in ('legacy_failure_attribution', 'regression', 'synonym_consistency', 'resolution_methods')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
