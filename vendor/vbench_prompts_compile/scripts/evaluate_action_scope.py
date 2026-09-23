#!/usr/bin/env python3
"""Post-audit Action scope-guard regression; explicitly NOT a new blind evaluation."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile.action_repair import compile_action, LEXICON_SHA256
from vbench_prompts_compile.generalization import correct
from vbench_prompts_compile.inference import parse_output
from vbench_prompts_compile.sources import load_k400
from score_matrix import score_one


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='docs/deterministic/ablation-v1')
    args = parser.parse_args()
    out = Path(args.out)
    vocab = load_k400()
    frozen_path = Path('output/ablation-v1/frozen-action-repair-v2.py')
    expected = json.loads((out / 'frozen-analysis.json').read_text())['action_interface']
    if sha(frozen_path) != expected:
        raise ValueError('frozen original compiler SHA mismatch')
    spec = importlib.util.spec_from_file_location('vbench_prompts_compile._frozen_action_v2', frozen_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    data_path = 'data/ablation-v1/eval-extended.jsonl'
    data = [r for r in read(data_path) if r['task'] == 'action']
    rows, frozen_checks = [], 0
    for size in ('0.6', '8'):
        predictions = {r['key']: r for r in read(f'output/ablation-v1/predictions-{size}b.jsonl')}
        for mode in ('base', 'sft'):
            values = {}
            for row in data:
                raw = predictions[f'{row["id"]}/{mode}/intact']['value']
                old = compile_action(row['input']['prompt'], raw, vocab, scope_guard=False)
                assert old == frozen.compile_action(row['input']['prompt'], raw, vocab)
                frozen_checks += 1
                new = compile_action(row['input']['prompt'], raw, vocab)
                values[row['id']] = (raw, old['target'], new['target'], bool(new.get('scope_guard')))
            for source, variant in sorted({(r['source'], r['variant']) for r in data}):
                cohort = [r for r in data if (r['source'], r['variant']) == (source, variant)]
                counts = [sum(correct('action', values[r['id']][i], r['target']) for r in cohort) for i in range(3)]
                rows.append({'size': size, 'mode': mode, 'source': source, 'variant': variant, 'n': len(cohort),
                             'raw_correct': counts[0], 'v2_correct': counts[1], 'v2_1_correct': counts[2],
                             'guard_used': sum(values[r['id']][3] for r in cohort)})
    old_cache_path = 'data/repair/matrix-v1/action-v9.jsonl'
    old_cache = read(old_cache_path)
    for row in old_cache:
        raw, _ = parse_output('action', row['raw'])
        old = frozen.compile_action(row['prompt'], raw, vocab)
        assert old == compile_action(row['prompt'], raw, vocab, scope_guard=False)
        assert old['target'] == compile_action(row['prompt'], raw, vocab)['target']
        frozen_checks += 1
    matrix_path = 'data/deterministic/matrix-v1/matrix.jsonl'
    visual_path = 'data/backend-cache/matrix-v1/action-test.jsonl'
    matrix = [r for r in read(matrix_path) if r['task'] == 'action']
    cache = {r['relative_path']: r for r in read(visual_path)}
    pred = {r['request_id']: r['target'] for r in old_cache}
    checked = Counter()
    for row in matrix:
        for scheme in ('Origin', 'Repair-rule', 'Repair-model'):
            before = score_one(row, scheme, cache[row['relative_path']], pred, vocab, action_interface='repair-v2')
            after = score_one(row, scheme, cache[row['relative_path']], pred, vocab, action_interface='repair-v2.1')
            for field in ('score', 'coverage', 'abstention', 'missing', 'target', 'status', 'frame_scores'):
                assert before[field] == after[field], (row['sample_id'], scheme, field)
            checked[scheme] += 1
    with (out / 'action-scope-regression.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = [data_path, old_cache_path, matrix_path, visual_path, str(frozen_path)]
    sources = ['src/vbench_prompts_compile/action_repair.py', 'src/vbench_prompts_compile/action_lexicon.py',
               'scripts/evaluate_action_scope.py', 'scripts/score_matrix.py']
    report = {'status': 'post_audit_development_regression_not_blind_generalization',
              'frozen_v2_full_result_matches': frozen_checks, 'old_unique_predictions_unchanged': len(old_cache),
              'matrix_scores_targets_coverage_unchanged': dict(checked), 'lexicon_sha256': LEXICON_SHA256,
              'new_training_steps': 0, 'new_generations': 0, 'test_labels_changed': 0,
              'input_sha256': {p: sha(p) for p in paths}, 'source_sha256': {p: sha(p) for p in sources},
              'note': 'The interface bug was discovered on v1. Keep the original v1 table frozen. This replay verifies the fix and old-score preservation, not independent generalization.'}
    (out / 'action-scope-regression.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
