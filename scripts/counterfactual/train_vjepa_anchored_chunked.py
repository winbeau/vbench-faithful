"""Numerical-runtime recovery: identical full-batch objective, bounded tensors.

Keep all data, initial weights and 300 optimizer steps from anchored-v1. Build
the full loss on concatenated head outputs; do not use stochastic mini-batches.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from . import train_vjepa_anchored as original
from .static_jitter import ROOT, digest
from .vjepa_motion_probe import fresh_output, gpu_precheck, read_rows, write_json, write_rows


def forward_chunks(model, chunks):
    import torch
    outputs = [model(x) for x in chunks]
    return torch.cat(outputs)


def train(args, config):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from dynamic_degree.learned_probe import MotionProbe
    from dynamic_degree.anchored_probe import anchored_losses

    start = time.monotonic(); gpu = gpu_precheck()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['optimization']['seed'])
    _, sources, pairs, features = original.previous(args.probe_root, config)
    for p in sorted(Path(args.features).glob('shard-*/features.jsonl')):
        c = json.loads((p.parent / 'completion.json').read_text())
        provenance = json.loads((p.parent / 'provenance.json').read_text())
        if c['status'] != 'finished' or c['failed_sources'] or not c['parity']['prior_base_feature_exact']:
            raise ValueError('anchor extraction incomplete')
        original.check(p, c['ledger_sha256']); original.check(args.config, provenance['config_sha256'])
        if provenance['code'] != original.identity():
            raise ValueError('original extraction code changed')
        for r in read_rows(p):
            key = (r['video_uid'], r['view'])
            if key in features or r['status'] != 'ok':
                raise ValueError('duplicate or failed features')
            features[key] = r
    if len(features) != 2160 or set(features) != {(r['video_uid'], v) for r in sources for v in original.VIEWS}:
        raise ValueError('complete source/view coverage required')
    out = fresh_output(args.output)
    write_json(out / 'provenance.json', {'config_sha256': digest(Path(args.config)), 'code': original.identity(),
        'trainer_sha256': digest(Path(__file__)), 'gpu': gpu, 'torch': torch.__version__, 'numpy': np.__version__,
        'encoder_updates': 0, 'chunk_views': 24, 'effective_batch': 'all210_sources_all8_views_one_optimizer_step',
        'initial_head_sha256': config['initial_joint_head_sha256'], 'prior_failed_attempt': '../training',
        'objective_or_hyperparameters_changed': False, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    def load(role):
        cohort = [r for r in sources if r['role'] == role]
        selected = [features[r['video_uid'], v] for r in cohort for v in original.VIEWS]
        chunks = []
        for start_index in range(0, len(selected), 24):
            arrays = []
            for r in selected[start_index:start_index + 24]:
                original.check(r['feature_path'], r['feature_sha256'])
                array = np.load(r['feature_path'], allow_pickle=False)
                if not np.isfinite(array).all() or array.shape != (4608, 768):
                    raise ValueError('feature shape/finite check failed')
                arrays.append(array)
            tensor = torch.from_numpy(np.stack(arrays)).to('cuda:0', torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError('feature GPU transfer finite check failed')
            chunks.append(tensor)
        print(json.dumps({'loaded_role': role, 'sources': len(cohort), 'views': len(selected), 'gpu_chunks': len(chunks)}), flush=True)
        return cohort, chunks
    cohort, chunks = load('train')
    pp = [r for r in pairs if r['role'] == 'train']
    index = {r['video_uid']: i for i, r in enumerate(cohort)}
    pair_index = torch.tensor([[index[r['a']], index[r['b']]] for r in pp], device='cuda:0')
    labels = torch.tensor([r['label'] for r in pp], device='cuda:0')
    model = MotionProbe().to('cuda:0')
    initial = torch.load(Path(args.probe_root) / 'training/joint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(initial, strict=True)
    with torch.inference_mode():
        q = forward_chunks(model, chunks).reshape(len(cohort), 8)
        if not torch.isfinite(q).all():
            raise ValueError('bounded initial head forward failed')
        previous_scores = {r['video_uid']: r for r in read_rows(Path(args.probe_root) / 'training/scores.jsonl')
                           if r['role'] == 'train' and r['arm'] == 'joint'}
        reference = np.array([previous_scores[r['video_uid']]['latent'] for r in cohort])
        parity_error = float(abs(q[:, :3].cpu().numpy() - reference).max())
        if parity_error > 1e-6:
            raise ValueError('bounded initial model disagrees with prior trained head')
        write_json(out / 'initial_loading_check.json', {'native630_outputs_max_abs_error': parity_error,
            'all1680_outputs_finite': True, 'chunk_views': 24})
    opt = config['optimization']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    curve = []
    with (out / 'progress.jsonl').open('x') as progress:
        for step in range(opt['steps']):
            model.train(); optimizer.zero_grad(set_to_none=True)
            q = forward_chunks(model, chunks).reshape(len(cohort), 8)
            loss, parts = anchored_losses(q, pair_index, labels, config['loss'])
            if not torch.isfinite(q).all() or not torch.isfinite(loss):
                raise ValueError(f'nonfinite forward/loss before optimizer step {step + 1}')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f'nonfinite gradient before optimizer step {step + 1}')
            optimizer.step()
            record = {'step': step + 1, 'total': float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}}
            curve.append(record); progress.write(json.dumps(record) + '\n'); progress.flush()
            if (step + 1) % 10 == 0:
                print(json.dumps(record), flush=True)
    model.eval().requires_grad_(False)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'anchored.pt')
    write_json(out / 'training_curves.json', curve)
    old = MotionProbe().to('cuda:0'); old.load_state_dict(initial, strict=True); old.eval().requires_grad_(False)
    del optimizer, q, loss, parts
    records = []; evaluations = {}
    for role in ('train', 'validation'):
        if role == 'validation':
            del chunks; torch.cuda.empty_cache(); cohort, chunks = load(role)
        role_pairs = [r for r in pairs if r['role'] == role]
        evaluations[role] = {}
        for arm, head in [('old_joint', old), ('anchored', model)]:
            with torch.inference_mode():
                q = forward_chunks(head, chunks).reshape(len(cohort), 8).cpu().numpy().astype(float)
            evaluations[role][arm] = original.statistics(cohort, role_pairs, q)
            for source, values in zip(cohort, q):
                records.append(dict(video_uid=source['video_uid'], prompt_id=source['prompt_id'], role=role,
                                    arm=arm, views=original.VIEWS, latent=values.tolist(), scores=(1 / (1 + np.exp(-values))).tolist()))
    write_rows(out / 'scores.jsonl', records)
    controls = []
    if args.diagnostic_root:
        with torch.inference_mode():
            for r in read_rows(Path(args.diagnostic_root) / 'scores/vjepa/scores.jsonl'):
                original.check(r['feature_path'], r['feature_sha256'])
                tensor = torch.from_numpy(np.load(r['feature_path'], allow_pickle=False)).to('cuda:0', torch.float32)[None]
                q = float(model(tensor)[0])
                controls.append(dict(case=r['family'], old_joint=r['joint']['score'], anchored=float(1 / (1 + np.exp(-q))), latent=q))
    write_json(out / 'summary.json', {'status': 'finished_fixed_model_training', 'evaluations': evaluations,
        'acceptance': original.acceptance(evaluations['validation']['anchored'], config['acceptance']),
        'trained_parameters': sum(p.numel() for p in model.parameters()), 'encoder_updates': 0, 'optimizer_steps': opt['steps'],
        'head_sha256': digest(out / 'anchored.pt'), 'scores_sha256': digest(out / 'scores.jsonl'),
        'user_requested_control_after_freeze': controls, 'posthoc_mapping_changed': False,
        'numerical_runtime': '24-view blocks; concatenated full-batch loss, one update per step',
        'initial_native_parity_max_abs_error': parity_error, 'previous_failed_attempt_preserved': True,
        'limitations': config['limitations'], 'wall_seconds': time.monotonic() - start,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'summary.json').read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'probe-root', 'features', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--diagnostic-root')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-vjepa-model-retraining-anchored-v1':
        raise ValueError('unexpected training protocol')
    train(args, config)


if __name__ == '__main__':
    main()
