#!/usr/bin/env python3
"""Batched, greedy, frozen-base/SFT ablations with isolated Scene loading and full denominators."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile.inference import AdapterRouter, SceneVerifier, parse_output, validate_direction_output
from vbench_prompts_compile.training import user_text


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jobs_for(rows):
    jobs = []
    scene_caps = sorted({r['input']['caption'] for r in rows if r['task'] == 'scene'})
    for mode in ('base', 'sft'):
        for row in rows:
            caption_modes = ['intact']
            if row['task'] == 'scene' and (row['variant'] == 'canonical' or row['source'] == 'snli_test_locations'):
                caption_modes += ['missing', 'shuffled']
            for caption_mode in caption_modes:
                inp = dict(row['input'])
                if caption_mode == 'missing':
                    inp['caption'] = ''
                elif caption_mode == 'shuffled':
                    i = scene_caps.index(inp['caption'])
                    inp['caption'] = scene_caps[(i + 1) % len(scene_caps)]
                jobs.append({'key': f'{row["id"]}/{mode}/{caption_mode}', 'id': row['id'],
                             'task': row['task'], 'mode': mode, 'caption_mode': caption_mode, 'input': inp})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/ablation-v1/eval.jsonl')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--input-budget', type=int, default=1792)
    args = p.parse_args()
    import gc
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20260919)
    cfg = json.loads(Path(args.config).read_text())
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights = {task: {name: sha(Path(path) / name) for name in ('adapter_model.safetensors', 'adapter_config.json', 'training_config.json')}
               for task, path in cfg['adapters'].items()}
    base_weights = {p.name: sha(p) for p in sorted(Path(cfg['model']).glob('model*.safetensors'))}
    if not base_weights:
        raise SystemExit('no local safetensors base weights found')
    plan = {'data_sha256': sha(args.data), 'config': cfg, 'runner_sha256': sha(__file__),
            'inference_sha256': sha(ROOT / 'src/vbench_prompts_compile/inference.py'),
            'training_sha256': sha(ROOT / 'src/vbench_prompts_compile/training.py'),
            'adapter_sha256': weights, 'base_config_sha256': sha(Path(cfg['model']) / 'config.json'),
            'base_weight_sha256': base_weights, 'lock_sha256': sha(ROOT / 'uv.lock'),
            'tokenizer_sha256': sha(Path(cfg['model']) / 'tokenizer.json'),
            'batch_size': args.batch_size, 'input_budget': args.input_budget,
            'parse_max_new_tokens': 128, 'scene_max_new_tokens': 8, 'seed': 20260919,
            'decoding': 'greedy, non-thinking, bf16, left padding; Scene newline boundary',
            'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}
    plan_path = out.with_suffix('.plan.json')
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise SystemExit('frozen inference plan mismatch')
    else:
        if out.exists():
            raise SystemExit('predictions without plan')
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    complete = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r['key'] in complete:
                raise SystemExit('duplicate result key')
            complete[r['key']] = r
    all_jobs = jobs_for(rows)
    resource_path = out.with_suffix('.resources.jsonl')
    with out.open('a') as output, resource_path.open('a') as resources:
        for system in ('parse', 'scene'):
            tasks = ['spatial', 'action', 'objects'] if system == 'parse' else ['scene']
            pending = [j for j in all_jobs if j['task'] in tasks and j['key'] not in complete]
            if not pending:
                continue
            torch.cuda.reset_peak_memory_stats()
            loaded_at = time.monotonic()
            if system == 'parse':
                engine = AdapterRouter(base_model_path=cfg['model'], adapters={t: cfg['adapters'][t] for t in tasks})
            else:
                engine = SceneVerifier(base_model_path=cfg['model'], adapter_path=cfg['adapters']['scene'])
            tokenizer, model = engine.tokenizer, engine.model
            tokenizer.padding_side = 'left'
            model.eval()
            config = model.config.to_dict()
            resources.write(json.dumps({'event': 'load', 'system': system, 'seconds': time.monotonic() - loaded_at,
                                        'parameters_with_loaded_adapters': sum(p.numel() for p in model.parameters()),
                                        'architecture': {k: config.get(k) for k in ('architectures', 'hidden_size', 'intermediate_size',
                                                        'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                                                        'head_dim', 'vocab_size', 'tie_word_embeddings')}}) + '\n')
            resources.flush()
            for task in tasks:
                for mode in ('base', 'sft'):
                    if system == 'parse':
                        engine.route(task)
                    jobs = [j for j in pending if j['task'] == task and j['mode'] == mode]
                    for j in jobs:
                        user = user_text({'task': task, 'input': j['input']}, spatial_relation_space='four_directions')
                        j['text'] = tokenizer.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                                                  add_generation_prompt=True, enable_thinking=False)
                        j['input_tokens'] = len(tokenizer.encode(j['text']))
                    jobs.sort(key=lambda j: (j['input_tokens'], j['key']))
                    for start in range(0, len(jobs), args.batch_size):
                        batch = jobs[start:start + args.batch_size]
                        overflow = [j for j in batch if j['input_tokens'] > args.input_budget]
                        batch = [j for j in batch if j['input_tokens'] <= args.input_budget]
                        results = [{**{k: v for k, v in j.items() if k not in ('text', 'input')}, 'raw': '', 'value': None,
                                    'errors': ['input_over_budget'], 'seconds_per_item_amortized': None} for j in overflow]
                        if batch:
                            inputs = tokenizer([j['text'] for j in batch], return_tensors='pt', padding=True)
                            inputs.pop('token_type_ids', None)
                            inputs = inputs.to(model.device)
                            torch.cuda.synchronize()
                            t = time.monotonic()
                            new_tokens = 8 if task == 'scene' else 128
                            kwargs = {'stop_strings': ['\n'], 'tokenizer': tokenizer} if task == 'scene' else {}
                            with torch.inference_mode(), (model.disable_adapter() if mode == 'base' else nullcontext()):
                                generated = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, num_beams=1,
                                                           temperature=None, top_p=None, top_k=None,
                                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **kwargs)
                            torch.cuda.synchronize()
                            elapsed = time.monotonic() - t
                            tokens = generated[:, inputs['input_ids'].shape[1]:]
                            raws = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                            for j, raw, token_ids in zip(batch, raws, tokens.tolist()):
                                value, errors = parse_output(task, raw.strip())
                                ended = tokenizer.eos_token_id in token_ids or (task == 'scene' and '\n' in raw)
                                if not ended and len(token_ids) >= new_tokens:
                                    value, errors = None, tuple(errors) + ('output_budget_exhausted',)
                                if task == 'spatial' and value is not None:
                                    value, more = validate_direction_output(value)
                                    errors = tuple(errors) + more
                                results.append({**{k: v for k, v in j.items() if k not in ('text', 'input')},
                                                'raw': raw.strip(), 'value': value, 'errors': list(errors),
                                                'generated_tokens_including_padding': len(token_ids),
                                                'seconds_per_item_amortized': elapsed / len(batch)})
                            resources.write(json.dumps({'event': 'batch', 'task': task, 'mode': mode, 'items': len(batch),
                                                        'seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}) + '\n')
                            resources.flush()
                        for result in results:
                            output.write(json.dumps(result, ensure_ascii=False) + '\n')
                            complete[result['key']] = result
                        output.flush()
                    print(json.dumps({'system': system, 'task': task, 'mode': mode, 'complete': len(complete)}), flush=True)
            del engine, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    expected = {j['key'] for j in all_jobs}
    missing, extra = expected - complete.keys(), complete.keys() - expected
    report = {'planned': len(expected), 'completed': len(complete), 'missing': sorted(missing), 'extra': sorted(extra),
              'predictions_sha256': sha(out), 'resources_sha256': sha(resource_path),
              'plan_sha256': sha(plan_path), 'complete': not missing and not extra}
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
