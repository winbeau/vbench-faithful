#!/usr/bin/env python3
"""Resume adapter inference over a frozen text plan, including EVERY Scene frame.

Inputs: prompt rows, annotation input pairs, or backend caches (all 16 captions).
Exact duplicate texts share one greedy prediction; the plan retains every source
reference and missing evidence. No filtering by model success or gold label.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.experiments import text_key
from vbench_prompts_compile.inference import AdapterRouter, SceneVerifier


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_plan(task, rows):
    requests, references = {}, []
    for i, row in enumerate(rows):
        inputs = row.get('input', row)
        prompt = inputs.get('prompt', '')
        if task == 'scene':
            if 'caption' in inputs:
                captions = [inputs['caption']]
            elif row.get('frame_captions') is not None:
                captions = list(row['frame_captions'])
                if len(captions) > 16:
                    raise ValueError('backend cache exceeds the frozen 16-frame plan')
                captions += [None] * (16 - len(captions))
            else:
                captions = [None] * 16
        else:
            captions = [None]
        for frame_index, caption in enumerate(captions):
            available = bool(prompt) and (task != 'scene' or isinstance(caption, str))
            key = text_key(task, prompt, caption) if available else None
            references.append({'row': i, 'source_id': row.get('sample_id', row.get('relative_path', row.get('video_path'))),
                               'frame_index': frame_index if task == 'scene' else None, 'request_id': key,
                               'eligible': row.get('eligible', True)})
            if key:
                requests[key] = {'request_id': key, 'task': task, 'prompt': prompt,
                                 **({'caption': caption} if task == 'scene' else {})}
    return list(requests.values()), references


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=['scene', 'spatial', 'action', 'objects'])
    p.add_argument('--prompts', required=True, action='append', help='repeat to share duplicate texts across input files')
    p.add_argument('--out', required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--adapter', required=True, action='append', help='task=final adapter directory')
    p.add_argument('--batch-size', type=int, default=8, help='Scene only; frozen before evaluation')
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--canonicalize-action-output', action='store_true')
    p.add_argument('--action-interface', choices=['repair-v2.1', 'repair-v2', 'legacy-model'], default='repair-v2.1',
                   help='Action text interface; canonicalize-action-output applies to legacy-model only')
    p.add_argument('--canonicalize-entity-output', action='store_true')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--max-items', type=int, help='pilot new requests; retains full denominator')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def run(args, out):
    rows = [row for path in args.prompts for row in J.read_jsonl(Path(path))]
    requests, references = build_plan(args.task, rows)
    adapters = dict(v.split('=', 1) for v in args.adapter)
    if args.task not in adapters or args.batch_size < 1:
        raise ValueError('missing requested adapter or invalid batch size')
    artifacts = {}
    for name, path in adapters.items():
        directory = Path(path)
        for p in sorted(directory.iterdir()):
            if p.is_file() and (p.name.startswith('adapter_') or p.name in {'training_config.json', 'run_summary.json'}):
                artifacts[f'{name}/{p.name}'] = sha(p)
    base = Path(args.base_model)
    for name in ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json']:
        p = base / name
        if p.exists():
            artifacts['base/' + name] = sha(p)
    plan = {'schema': 2, 'task': args.task, 'inputs_sha256': {p: sha(p) for p in args.prompts},
        'inference_code_sha256': {name: sha(ROOT / name) for name in ['scripts/batch_predict.py', 'src/vbench_prompts_compile/inference.py',
            'src/vbench_prompts_compile/action_repair.py', 'src/vbench_prompts_compile/action_lexicon.py',
            'src/vbench_prompts_compile/sources.py', 'src/vbench_prompts_compile/records.py']},
        'model_artifacts_sha256': artifacts, 'adapters': adapters, 'base_model': args.base_model,
        'batch_size': args.batch_size if args.task == 'scene' else 1,
        'max_new_tokens': 8 if args.task == 'scene' else args.max_new_tokens,
        'canonicalize_action': args.canonicalize_action_output, 'canonicalize_entity': args.canonicalize_entity_output,
        'action_interface': args.action_interface if args.task == 'action' else None,
        'action_vocabulary_sha256': sha(ROOT / 'data/raw/k400/labels.json') if args.task == 'action' else None,
        'requests': requests, 'references': references}
    J.bind_job(out.with_suffix('.plan.json'), plan)
    if args.dry_run:
        print(json.dumps({'unique_text_requests': len(requests), 'source_references': len(references)}))
        return 0
    if not args.allow_download:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    done = {r['request_id']: r for r in J.read_jsonl(out)}
    if not set(done) <= {r['request_id'] for r in requests}:
        raise ValueError('unplanned prediction in output')
    pending = [r for r in requests if r['request_id'] not in done]
    if args.max_items is not None:
        pending = pending[:args.max_items]
    if pending:
        if args.task == 'scene':
            model = SceneVerifier(base_model_path=args.base_model, adapter_path=adapters['scene'], local_files_only=not args.allow_download)
            batch_size = args.batch_size
        else:
            model = AdapterRouter(base_model_path=args.base_model, adapters=adapters, local_files_only=not args.allow_download)
            batch_size = 1
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            if args.task == 'scene':
                predictions = model.predict_batch([(r['prompt'], r['caption']) for r in batch])
            else:
                predictions = [model.predict(args.task, model.user_text({'task': args.task, 'input': {'prompt': r['prompt']}}),
                    max_new_tokens=args.max_new_tokens, canonicalize_action_output=args.canonicalize_action_output,
                    canonicalize_entity_output=args.canonicalize_entity_output,
                    action_interface=args.action_interface, action_prompt=r['prompt']) for r in batch]
            if len(predictions) != len(batch):
                raise ValueError('prediction count mismatch')
            for row, prediction in zip(batch, predictions):
                result = {**row, 'target': prediction.value, 'raw': prediction.raw, 'errors': list(prediction.errors)}
                if prediction.postprocess is not None:
                    result['postprocess'] = prediction.postprocess
                J.append_jsonl(out, result)
                done[row['request_id']] = result
            print(json.dumps({'done': len(done), 'planned': len(requests), 'task': args.task}), flush=True)
    report = {'requests': len(requests), 'completed': len(done), 'missing': len(requests) - len(done),
        'invalid': sum(r['target'] is None for r in done.values()), 'source_references': len(references),
        'missing_evidence': sum(r['request_id'] is None and r['eligible'] for r in references),
        'ineligible_references': sum(not r['eligible'] for r in references),
        'complete': len(done) == len(requests), 'errors': dict(Counter(e for r in done.values() for e in r['errors']))}
    J.atomic_json(out.with_suffix('.report.json'), report)
    print(json.dumps(report))
    return 0 if report['complete'] else 2


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    with J.job_lock(out.with_suffix('.lock')):
        return run(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
