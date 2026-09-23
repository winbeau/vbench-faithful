#!/usr/bin/env python3
"""Replay the production Action interface over preserved raw generations on CPU.

The adapter is unchanged. Output keeps raw/model/legacy/final targets and the
resolution trace; no GPU inference, API calls or evaluation targets are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J, records as R
from vbench_prompts_compile.action_repair import compile_action, LEXICON_SHA256, INTERFACE_VERSION
from vbench_prompts_compile.inference import parse_output
from vbench_prompts_compile.sources import load_k400
from vbench_prompts_compile.experiments import text_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repair_rows(rows, vocab):
    output, seen = [], set()
    for row in rows:
        if row['task'] != 'action' or row['request_id'] != text_key('action', row['prompt']):
            raise ValueError('invalid Action request identity')
        if row['request_id'] in seen:
            raise ValueError('duplicate Action request identity')
        seen.add(row['request_id'])
        model_target, errors = parse_output('action', row['raw'])
        resolution = compile_action(row['prompt'], model_target, vocab)
        output.append({**row, 'legacy_target': row['target'], 'model_target': model_target,
                       'legacy_errors': row.get('errors', []), 'errors': list(errors),
                       'postprocess': resolution, 'target': resolution['target']})
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--vocabulary', default=str(ROOT / 'data/raw/k400/labels.json'))
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == Path(args.predictions).resolve():
        raise ValueError('preserved raw predictions must not be overwritten')
    rows = J.read_jsonl(Path(args.predictions))
    if not rows:
        raise ValueError('empty Action prediction cache')
    paths = [args.predictions, args.vocabulary]
    supporting = [Path(args.predictions).with_suffix(suffix) for suffix in ('.plan.json', '.report.json')]
    sources = ['scripts/repair_action_predictions.py', 'src/vbench_prompts_compile/action_repair.py',
               'src/vbench_prompts_compile/action_lexicon.py', 'src/vbench_prompts_compile/inference.py',
               'src/vbench_prompts_compile/sources.py', 'src/vbench_prompts_compile/records.py']
    plan = {'task': 'action', 'operation': 'raw_generation_postprocessing', 'action_interface': INTERFACE_VERSION,
            'input_sha256': {path: sha(path) for path in paths},
            'supporting_sha256': {str(path): sha(path) for path in supporting if path.exists()},
            'source_sha256': {path: sha(ROOT / path) for path in sources},
            'lexicon_sha256': LEXICON_SHA256, 'new_generation_requests': 0, 'new_training_steps': 0}
    J.bind_job(out.with_suffix('.plan.json'), plan)
    repaired = repair_rows(rows, load_k400(Path(args.vocabulary)))
    if out.exists() and J.read_jsonl(out) != repaired:
        raise ValueError('existing repaired cache differs; use a new output path')
    R.write_jsonl(out, repaired)
    report = {**plan, 'requests': len(rows), 'completed': len(repaired), 'complete': True,
              'invalid': sum(r['target'] is None for r in repaired),
              'resolution_methods': dict(Counter(r['postprocess']['method'] for r in repaired)),
              'raw_generation_unchanged': all(a['raw'] == b['raw'] for a, b in zip(rows, repaired)),
              'output_sha256': sha(out)}
    J.atomic_json(out.with_suffix('.report.json'), report)
    print(json.dumps({key: report[key] for key in ('requests', 'complete', 'invalid', 'resolution_methods', 'raw_generation_unchanged')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
