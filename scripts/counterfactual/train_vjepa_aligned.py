"""One fixed head-training trial adding TRAIN-only Origin population scale.

Retain anchored-v1 supervision and features. Do not fit posthoc output transforms,
per-video Origin binary targets, validation targets or exposed TEST450 means.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from . import train_vjepa_anchored as parent
from .static_jitter import ROOT, digest
from .train_vjepa_anchored_chunked import forward_chunks
from .validate_vjepa_motion import check_sha
from .vjepa_motion_probe import fresh_output, gpu_precheck, read_rows, write_json, write_rows


def load_teacher(root, sources, config_path):
    expected = {r['video_uid']: r for r in sources}; output = {}; receipts = []
    paths = sorted(Path(root).glob('shard-*'))
    if len(paths) != 4:
        raise ValueError('four complete Origin shards required')
    for path in paths:
        c = json.loads((path / 'completion.json').read_text())
        p = json.loads((path / 'provenance.json').read_text())
        if c['status'] != 'finished' or c['failed'] or c['expected'] != c['completed']:
            raise ValueError('Origin measurement incomplete; no training')
        check_sha(config_path, p['config_sha256']); check_sha(path / 'scores.jsonl', c['scores_sha256'])
        part = read_rows(path / 'scores.jsonl')
        if len(part) != c['completed'] or {r['video_uid'] for r in part} != set(p['expected_uids']):
            raise ValueError('Origin count mismatch')
        for r in part:
            uid = r['video_uid']
            if uid not in expected or uid in output or r['status'] != 'ok' or r['score'] not in (0., 1.):
                raise ValueError('duplicate, extra or invalid teacher')
            for field in ('role', 'prompt_id', 'generator', 'relative_video_path'):
                if r[field] != expected[uid][field]:
                    raise ValueError('teacher metadata mismatch')
            output[uid] = r
        receipts.append({'path': str(path), 'completion': c, 'provenance_sha256': digest(path / 'provenance.json')})
    if set(output) != set(expected):
        raise ValueError('missing Origin source')
    return output, receipts


def train_mean(sources, teacher):
    train = [r for r in sources if r['role'] == 'train']
    if not train:
        raise ValueError('training sources required')
    return float(np.mean([teacher[r['video_uid']]['score'] for r in train]))


def acceptance(stats, parent_gates, origin_validation_mean, config):
    result = parent.acceptance(stats, parent_gates)
    gap = abs(stats['mean_by_view']['base'] - origin_validation_mean)
    result['checks']['natural_scale_alignment'] = gap <= config['acceptance']['validation_native_mean_abs_error_vs_origin_max']
    result.update(all_passed=all(result['checks'].values()), native_mean_abs_error_vs_origin=gap,
                  origin_validation_mean=origin_validation_mean)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'probe-root', 'anchored-root', 'origin-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--diagnostic-root')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-vjepa-origin-scale-model-training-v1':
        raise ValueError('unexpected protocol')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from dynamic_degree.aligned_probe import aligned_losses
    from dynamic_degree.learned_probe import MotionProbe
    start = time.monotonic(); gpu = gpu_precheck()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['optimization']['seed'])
    parent_path = ROOT / config['parent_config']; check_sha(parent_path, config['parent_config_sha256'])
    pc = json.loads(parent_path.read_text())
    _, sources, pairs, features = parent.previous(args.probe_root, pc)
    if Counter(r['role'] for r in sources) != {'train': 210, 'validation': 60}:
        raise ValueError('fixed source split changed')
    anchored = Path(args.anchored_root)
    check_sha(anchored / 'training-chunked/summary.json', config['parent_summary_sha256'])
    old_summary = json.loads((anchored / 'training-chunked/summary.json').read_text())
    check_sha(anchored / 'training-chunked/scores.jsonl', old_summary['scores_sha256'])
    initial_path = anchored / 'training-chunked/anchored.pt'; check_sha(initial_path, config['initial_head_sha256'])
    for path in sorted((anchored / 'features').glob('shard-*/features.jsonl')):
        c = json.loads((path.parent / 'completion.json').read_text()); pv = json.loads((path.parent / 'provenance.json').read_text())
        if c['status'] != 'finished' or c['failed_sources'] or not c['parity']['prior_base_feature_exact']:
            raise ValueError('anchor features incomplete')
        check_sha(path, c['ledger_sha256']); check_sha(parent_path, pv['config_sha256'])
        if pv['code'] != parent.identity():
            raise ValueError('parent feature implementation changed')
        for r in read_rows(path):
            key = r['video_uid'], r['view']
            if key in features or r['status'] != 'ok':
                raise ValueError('duplicate or failed feature')
            features[key] = r
    if len(features) != 2160 or set(features) != {(r['video_uid'], v) for r in sources for v in parent.VIEWS}:
        raise ValueError('full eight-view coverage required')
    teacher, receipts = load_teacher(args.origin_root, sources, args.config)
    target = train_mean(sources, teacher)  # NO validation values enter this scalar.
    out = fresh_output(args.output)
    code = parent.identity()
    for path in (Path(__file__), ROOT / 'metrics/dynamic-degree/src/dynamic_degree/aligned_probe.py',
                 ROOT / 'scripts/counterfactual/train_vjepa_anchored_chunked.py'):
        code[str(path.relative_to(ROOT))] = digest(path)
    write_json(out / 'provenance.json', {'config_sha256': digest(Path(args.config)), 'code': code, 'gpu': gpu,
        'initial_head_sha256': config['initial_head_sha256'], 'origin_train_mean_target': target,
        'teacher_training_sources': 210, 'teacher_validation_sources_used_in_loss': 0,
        'teacher_receipts': receipts, 'encoder_updates': 0, 'torch': torch.__version__, 'numpy': np.__version__,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    def load(role):
        cohort = [r for r in sources if r['role'] == role]
        selected = [features[r['video_uid'], v] for r in cohort for v in parent.VIEWS]
        chunks = []
        for offset in range(0, len(selected), 24):
            arrays = []
            for r in selected[offset:offset + 24]:
                check_sha(r['feature_path'], r['feature_sha256'])
                array = np.load(r['feature_path'], allow_pickle=False)
                if array.shape != (4608, 768) or not np.isfinite(array).all():
                    raise ValueError('invalid cached tokens')
                arrays.append(array)
            chunks.append(torch.from_numpy(np.stack(arrays)).to('cuda:0', torch.float32))
        print(json.dumps({'loaded_role': role, 'sources': len(cohort), 'views': len(selected)}), flush=True)
        return cohort, chunks
    cohort, chunks = load('train')
    pp = [r for r in pairs if r['role'] == 'train']; index = {r['video_uid']: i for i, r in enumerate(cohort)}
    pair_index = torch.tensor([[index[r['a']], index[r['b']]] for r in pp], device='cuda:0')
    labels = torch.tensor([r['label'] for r in pp], device='cuda:0')
    model = MotionProbe().to('cuda:0')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True); model.load_state_dict(initial, strict=True)
    with torch.inference_mode():
        before = forward_chunks(model, chunks).reshape(210, 8).cpu().numpy().astype(float)
        stored = {r['video_uid']: r['latent'] for r in read_rows(anchored / 'training-chunked/scores.jsonl')
                  if r['arm'] == 'anchored' and r['role'] == 'train'}
        parity = float(abs(before - np.array([stored[r['video_uid']] for r in cohort])).max())
        if parity > 1e-6:
            raise ValueError('initial anchored head parity failed')
    opt = config['optimization']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    curve = []
    with (out / 'progress.jsonl').open('x') as progress:
        for step in range(opt['steps']):
            model.train(); optimizer.zero_grad(set_to_none=True)
            q = forward_chunks(model, chunks).reshape(210, 8)
            loss, parts = aligned_losses(q, pair_index, labels, pc['loss'], target, config['loss']['origin_mean_weight'])
            if not torch.isfinite(q).all() or not torch.isfinite(loss):
                raise ValueError(f'nonfinite forward/loss step {step + 1}')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f'nonfinite gradient step {step + 1}')
            optimizer.step()
            record = {'step': step + 1, 'total': float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}}
            curve.append(record); progress.write(json.dumps(record) + '\n'); progress.flush()
            if (step + 1) % 25 == 0:
                print(json.dumps(record), flush=True)
    model.eval().requires_grad_(False)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'aligned.pt')
    write_json(out / 'training_curves.json', curve)
    reloaded = MotionProbe().to('cuda:0')
    reloaded.load_state_dict(torch.load(out / 'aligned.pt', map_location='cpu', weights_only=True), strict=True)
    reloaded.eval().requires_grad_(False)
    with torch.inference_mode():
        error = float((forward_chunks(model, chunks) - forward_chunks(reloaded, chunks)).abs().max())
        if error > 1e-6:
            raise ValueError('saved model cannot reproduce trained outputs')
    del optimizer, q, loss, parts, model
    old = MotionProbe().to('cuda:0'); old.load_state_dict(initial, strict=True); old.eval().requires_grad_(False)
    evaluations = {}; records = []; origin_means = {}
    for role in ('train', 'validation'):
        if role == 'validation':
            del chunks; torch.cuda.empty_cache(); cohort, chunks = load(role)
        origin_means[role] = float(np.mean([teacher[r['video_uid']]['score'] for r in cohort]))
        evaluations[role] = {}
        for arm, head in [('previous_anchored', old), ('aligned', reloaded)]:
            with torch.inference_mode():
                values = forward_chunks(head, chunks).reshape(len(cohort), 8).cpu().numpy().astype(float)
            stats = parent.statistics(cohort, [r for r in pairs if r['role'] == role], values)
            stats['origin_native_mean'] = origin_means[role]
            stats['native_mean_abs_error_vs_origin'] = abs(stats['mean_by_view']['base'] - origin_means[role])
            evaluations[role][arm] = stats
            for r, latent in zip(cohort, values):
                records.append(dict(video_uid=r['video_uid'], prompt_id=r['prompt_id'], role=role, arm=arm,
                    views=parent.VIEWS, latent=latent.tolist(), scores=(1 / (1 + np.exp(-latent))).tolist()))
    write_rows(out / 'scores.jsonl', records)
    controls = []
    if args.diagnostic_root:
        prior_controls = {r['case']: r for r in old_summary['user_requested_control_after_freeze']}
        with torch.inference_mode():
            for r in read_rows(Path(args.diagnostic_root) / 'scores/vjepa/scores.jsonl'):
                check_sha(r['feature_path'], r['feature_sha256'])
                x = torch.from_numpy(np.load(r['feature_path'], allow_pickle=False)).to('cuda:0', torch.float32)[None]
                q = float(reloaded(x)[0])
                controls.append(dict(case=r['family'], previous_anchored=prior_controls[r['family']]['anchored'],
                                     aligned=float(1 / (1 + np.exp(-q))), latent=q))
    write_json(out / 'summary.json', {'status': 'finished_fixed_scale_supervised_model_training',
        'evaluations': evaluations, 'origin_means': origin_means, 'origin_train_mean_target': target,
        'acceptance': acceptance(evaluations['validation']['aligned'], pc['acceptance'], origin_means['validation'], config),
        'trained_parameters': sum(p.numel() for p in reloaded.parameters()), 'encoder_updates': 0,
        'optimizer_steps': opt['steps'], 'head_sha256': digest(out / 'aligned.pt'),
        'scores_sha256': digest(out / 'scores.jsonl'), 'initial_loading_max_abs_error': parity,
        'saved_head_reload_max_abs_error': error, 'user_requested_control_after_freeze': controls,
        'posthoc_mapping_changed': False, 'individual_origin_labels_used': False, 'validation_teacher_used_in_loss': False,
        'test450_used_for_training': False, 'test450_this_model': 'NOT RUN', 'calibration45_opened': False,
        'limitations': config['limitations'], 'wall_seconds': time.monotonic() - start,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
