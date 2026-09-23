"""One fixed model-retraining trial with supervised still and motion anchors.

Frozen V-JEPA encoder, trained video-head parameters, no posthoc score mapping.
Uses only the original prompt-separated DEV210/60; never trains on TEST450.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .static_jitter import ROOT, digest
from .official_video_jitter import native_video
from .probe_vjepa_static_frame import global_pan
from .vjepa_motion_probe import augment, fresh_output, gpu_precheck, read_rows, write_json, write_rows


ANCHORS = ('still', 'pan8', 'pan32', 'reversal32', 'still_jitter8')
VIEWS = ('base', 'alternating', 'aperiodic') + ANCHORS


def check(path, expected):
    if digest(Path(path)) != expected:
        raise ValueError(f'identity changed: {path}')


def identity():
    paths = [Path(__file__), ROOT / 'metrics/dynamic-degree/src/dynamic_degree/anchored_probe.py',
             ROOT / 'metrics/dynamic-degree/src/dynamic_degree/learned_probe.py',
             ROOT / 'packages/audit-models/src/vbench_audit_models/vjepa.py']
    paths += [Path(__file__).with_name(n) for n in ('vjepa_motion_probe.py', 'probe_vjepa_static_frame.py',
                                                  'official_video_jitter.py', 'local_texture_jitter.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def previous(probe_root, config):
    root = Path(probe_root)
    pc_path = ROOT / 'configs/dynamic-static-jitter/vjepa-probe-v1.json'
    check(pc_path, config['probe_config_sha256'])
    check(root / 'selection/sources.jsonl', config['probe_sources_sha256'])
    check(root / 'selection/pairs.jsonl', config['probe_pairs_sha256'])
    check(root / 'training/joint.pt', config['initial_joint_head_sha256'])
    check(ROOT / 'metrics/dynamic-degree/src/dynamic_degree/learned_probe.py', config['head_implementation_sha256'])
    sources = [r for r in read_rows(root / 'selection/sources.jsonl') if r['role'] in ('train', 'validation')]
    pairs = read_rows(root / 'selection/pairs.jsonl')
    features = {}
    for p in sorted((root / 'features').glob('shard-*/features.jsonl')):
        c = json.loads((p.parent / 'completion.json').read_text())
        if c['status'] != 'finished' or c['failed_sources']:
            raise ValueError('previous features incomplete')
        check(p, c['ledger_sha256'])
        for r in read_rows(p):
            key = (r['video_uid'], r['view'])
            if r['status'] != 'ok' or key in features:
                raise ValueError('previous feature failure or duplicate')
            features[key] = r
    if len(sources) != 270 or len(features) != 810:
        raise ValueError('previous source/feature coverage changed')
    if set(features) != {(r['video_uid'], v) for r in sources for v in VIEWS[:3]}:
        raise ValueError('previous source-view mismatch')
    return json.loads(pc_path.read_text()), sources, pairs, features


def anchor_views(frames, uid, config, pc):
    frame = frames[config['anchors']['reference_frame_index']]
    still = np.repeat(frame[None], len(frames), axis=0)
    height, width = frame.shape[:2]
    positions = np.concatenate([np.linspace(0, 32, 8), np.linspace(32, 0, 8)])
    reversal = np.stack([cv2.warpAffine(frame, np.array([[1, 0, dx], [0, 1, 0]], np.float32),
                          (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for dx in positions])
    jitter, geometry = augment(still, uid, 'alternating', pc)
    return {'still': still, 'pan8': global_pan(frame, 16, 8), 'pan32': global_pan(frame, 16, 32),
            'reversal32': reversal, 'still_jitter8': jitter}, geometry


def extract(args, config):
    import torch
    from vbench_audit_models.vjepa import FrozenVJEPA

    start = time.monotonic(); gpu = gpu_precheck()
    torch.set_num_threads(3); cv2.setNumThreads(1)
    pc, sources, _, old = previous(args.probe_root, config)
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard')
    selected = sources[args.shard::args.shards]
    out = fresh_output(args.output); (out / 'features').mkdir()
    probe = Path(args.probe_root)
    encoder = FrozenVJEPA(probe / 'assets/vjepa2', probe / 'assets/vjepa2_1_vitb_dist_vitG_384.pt', pc)
    write_json(out / 'provenance.json', {'config_sha256': digest(Path(args.config)), 'code': identity(),
        'encoder': encoder.identity, 'gpu': gpu, 'sources_sha256': config['probe_sources_sha256'],
        'shard': args.shard, 'shards': args.shards, 'training_updates': 0,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    failures = 0; completed = 0; parity = None
    with (out / 'features.jsonl').open('x') as handle:
        for i, source in enumerate(selected):
            try:
                record = old[source['video_uid'], 'base']
                path = Path(args.video_root) / source['relative_video_path']
                check(path, record['video_sha256'])
                frames, pts, fps = native_video(path, 1e-6)
                if len(frames) != 16 or fps != 8:
                    raise ValueError('native cadence changed')
                if parity is None:
                    check(record['feature_path'], record['feature_sha256'])
                    same = np.array_equal(encoder.encode(frames), np.load(record['feature_path'], allow_pickle=False))
                    parity = {'prior_base_feature_exact': bool(same), 'uid': source['video_uid']}
                    if not same:
                        raise ValueError('frozen encoder loading/parity failed')
                views, geometry = anchor_views(frames, source['video_uid'], config, pc)
                for view in ANCHORS:
                    tokens = encoder.encode(views[view])
                    fp = out / 'features' / f"{source['video_uid']}.{view}.npy"
                    with fp.open('xb') as f:
                        np.save(f, tokens, allow_pickle=False)
                    r = dict(source, view=view, status='ok', feature_path=str(fp), feature_sha256=digest(fp),
                             feature_shape=list(tokens.shape), video_sha256=record['video_sha256'],
                             pixels_sha256=hashlib.sha256(views[view].tobytes()).hexdigest(),
                             decoded_shape=list(frames.shape), fps=fps, pts=pts,
                             geometry=geometry if view == 'still_jitter8' else None)
                    handle.write(json.dumps(r, allow_nan=False) + '\n'); handle.flush(); completed += 1
            except Exception as exc:
                failures += 1
                handle.write(json.dumps(dict(source, status='failed', error=f'{type(exc).__name__}: {exc}')) + '\n')
                handle.flush()
                if i == 0:
                    raise
            if (i + 1) % 10 == 0 or i + 1 == len(selected):
                print(json.dumps({'shard': args.shard, 'sources': i + 1, 'expected_sources': len(selected),
                                  'features': completed, 'failures': failures}), flush=True)
    write_json(out / 'completion.json', {'status': 'finished' if not failures else 'failed', 'features': completed,
        'sources': len(selected), 'failed_sources': failures, 'parity': parity, 'ledger_sha256': digest(out / 'features.jsonl'),
        'wall_seconds': time.monotonic() - start, 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    if failures:
        raise RuntimeError('anchor feature failures; no silent exclusion or training')


def statistics(sources, pairs, latent):
    score = 1 / (1 + np.exp(-latent))
    index = {r['video_uid']: i for i, r in enumerate(sources)}
    ordered = [r for r in pairs if r['label'] != .5]; tied = [r for r in pairs if r['label'] == .5]
    margins = np.array([(score[index[r['a']], :3] - score[index[r['b']], :3]) * (2 * r['label'] - 1) for r in ordered])
    gaps = np.array([abs(score[index[r['a']], :3] - score[index[r['b']], :3]) for r in tied])
    change = score[:, 1:3] - score[:, :1]
    return {'sources': len(sources), 'ordered_pairs': len(ordered), 'human_ties': len(tied),
        'ordered_correct_base_alternating_aperiodic': (margins > 0).sum(0).tolist(),
        'human_tie_mean_gap_by_natural_view': gaps.mean(0).tolist(),
        'mean_by_view': dict(zip(VIEWS, score.mean(0).tolist())),
        'p95_by_view': dict(zip(VIEWS, np.quantile(score, .95, axis=0).tolist())),
        'native_base_std': float(score[:, 0].std()), 'native_base_range': [float(score[:, 0].min()), float(score[:, 0].max())],
        'native_jitter_delta': float(change.mean()), 'native_jitter_mae': float(abs(change).mean()),
        'native_largest_drop': float(change.min()),
        'weak_pan_above_still_fraction': float(np.mean(score[:, 4] > score[:, 3])),
        'strong_pan_above_weak_fraction': float(np.mean(score[:, 5] > score[:, 4])),
        'reversal_above_still_fraction': float(np.mean(score[:, 6] > score[:, 3])),
        'mean_motion_margins': {'pan8_minus_still': float((score[:, 4] - score[:, 3]).mean()),
                               'pan32_minus_pan8': float((score[:, 5] - score[:, 4]).mean()),
                               'reversal32_minus_still': float((score[:, 6] - score[:, 3]).mean())}}


def acceptance(stats, gates):
    correct = stats['ordered_correct_base_alternating_aperiodic']
    checks = {f'{v}_mean': stats['mean_by_view'][v] <= gates['validation_still_mean_max'] for v in ('still', 'still_jitter8')}
    checks.update({f'{v}_p95': stats['p95_by_view'][v] <= gates['validation_still_p95_max'] for v in ('still', 'still_jitter8')})
    for kind in ('weak_pan_above_still', 'strong_pan_above_weak', 'reversal_above_still'):
        checks[kind] = stats[kind + '_fraction'] >= gates[kind + '_fraction_min']
    checks.update(natural_ordering=correct[0] >= gates['validation_native_ordered_correct_min'],
                  jitter_ordering=min(correct[1:]) >= correct[0] - gates['max_cf_correct_loss_from_clean'],
                  native_invariance=stats['native_jitter_mae'] <= gates['native_jitter_mae_max'],
                  nonconstant=stats['native_base_std'] >= gates['native_score_std_min'])
    return {'checks': checks, 'all_passed': all(checks.values()), 'formal_goal_complete': False}


def train(args, config):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from dynamic_degree.learned_probe import MotionProbe
    from dynamic_degree.anchored_probe import anchored_losses

    start = time.monotonic(); gpu = gpu_precheck()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['optimization']['seed'])
    _, sources, pairs, features = previous(args.probe_root, config)
    receipts = []
    for p in sorted(Path(args.features).glob('shard-*/features.jsonl')):
        c = json.loads((p.parent / 'completion.json').read_text())
        provenance = json.loads((p.parent / 'provenance.json').read_text())
        if c['status'] != 'finished' or c['failed_sources'] or not c['parity']['prior_base_feature_exact']:
            raise ValueError('incomplete anchor feature extraction')
        check(p, c['ledger_sha256']); check(args.config, provenance['config_sha256'])
        if provenance['code'] != identity():
            raise ValueError('extraction code changed before training')
        receipts.append({'completion': c, 'provenance_sha256': digest(p.parent / 'provenance.json')})
        for r in read_rows(p):
            key = (r['video_uid'], r['view'])
            if key in features or r['status'] != 'ok':
                raise ValueError('duplicate or failed anchor feature')
            features[key] = r
    if len(receipts) != 4 or len(features) != 2160 or set(features) != {(r['video_uid'], v) for r in sources for v in VIEWS}:
        raise ValueError('complete 270-source/eight-view feature coverage required')
    out = fresh_output(args.output)
    write_json(out / 'provenance.json', {'config_sha256': digest(Path(args.config)), 'code': identity(), 'gpu': gpu,
        'torch': torch.__version__, 'numpy': np.__version__, 'initial_head_sha256': config['initial_joint_head_sha256'],
        'encoder_updates': 0, 'training_sources_sha256': config['probe_sources_sha256'], 'feature_shards': receipts,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    def load(role):
        cohort = [r for r in sources if r['role'] == role]
        array = np.empty((len(cohort) * len(VIEWS), 4608, 768), np.float16)
        for i, source in enumerate(cohort):
            for j, view in enumerate(VIEWS):
                r = features[source['video_uid'], view]; check(r['feature_path'], r['feature_sha256'])
                array[i * len(VIEWS) + j] = np.load(r['feature_path'], allow_pickle=False)
        if not np.isfinite(array).all():
            raise ValueError('nonfinite features')
        return cohort, torch.from_numpy(array).to('cuda:0', torch.float32)
    cohort, x = load('train')
    train_pairs = [r for r in pairs if r['role'] == 'train']
    index = {r['video_uid']: i for i, r in enumerate(cohort)}
    pair_index = torch.tensor([[index[r['a']], index[r['b']]] for r in train_pairs], device='cuda:0')
    labels = torch.tensor([r['label'] for r in train_pairs], device='cuda:0')
    model = MotionProbe().to('cuda:0')
    initial = torch.load(Path(args.probe_root) / 'training/joint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(initial, strict=True)
    opt = config['optimization']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    curve = []
    for step in range(opt['steps']):
        model.train(); optimizer.zero_grad(set_to_none=True)
        q = model(x).reshape(len(cohort), len(VIEWS))
        loss, parts = anchored_losses(q, pair_index, labels, config['loss'])
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward(); optimizer.step()
        curve.append({'step': step + 1, 'total': float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}})
        if (step + 1) % 10 == 0:
            print(json.dumps(curve[-1]), flush=True)
    model.eval().requires_grad_(False)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / 'anchored.pt')
    write_json(out / 'training_curves.json', curve)
    old = MotionProbe().to('cuda:0'); old.load_state_dict(initial, strict=True); old.eval().requires_grad_(False)
    del optimizer, q, loss, parts
    records = []; evaluations = {}
    for role in ('train', 'validation'):
        if role == 'validation':
            del x; torch.cuda.empty_cache(); cohort, x = load(role)
        role_pairs = [r for r in pairs if r['role'] == role]
        evaluations[role] = {}
        for arm, head in [('old_joint', old), ('anchored', model)]:
            with torch.inference_mode():
                q = head(x).reshape(len(cohort), len(VIEWS)).cpu().numpy().astype(float)
            evaluations[role][arm] = statistics(cohort, role_pairs, q)
            for source, values in zip(cohort, q):
                records.append(dict(video_uid=source['video_uid'], prompt_id=source['prompt_id'], role=role,
                                    arm=arm, views=VIEWS, latent=values.tolist(), scores=(1 / (1 + np.exp(-values))).tolist()))
    write_rows(out / 'scores.jsonl', records)
    control_scores = None
    if args.diagnostic_root:
        diagnostic = Path(args.diagnostic_root)
        old_scores = read_rows(diagnostic / 'scores/vjepa/scores.jsonl')
        control_scores = []
        with torch.inference_mode():
            for r in old_scores:
                check(r['feature_path'], r['feature_sha256'])
                tensor = torch.from_numpy(np.load(r['feature_path'], allow_pickle=False)).to('cuda:0', torch.float32)[None]
                q = float(model(tensor)[0]); value = float(1 / (1 + np.exp(-q)))
                control_scores.append(dict(case=r['family'], old_joint=r['joint']['score'], anchored=value, latent=q))
    write_json(out / 'summary.json', {'status': 'finished_fixed_model_training', 'evaluations': evaluations,
        'acceptance': acceptance(evaluations['validation']['anchored'], config['acceptance']),
        'trained_parameters': sum(p.numel() for p in model.parameters()), 'encoder_updates': 0, 'optimizer_steps': opt['steps'],
        'head_sha256': digest(out / 'anchored.pt'), 'scores_sha256': digest(out / 'scores.jsonl'),
        'user_requested_control_after_freeze': control_scores, 'posthoc_mapping_changed': False,
        'limitations': config['limitations'], 'wall_seconds': time.monotonic() - start,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'summary.json').read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract'); e.add_argument('--video-root', required=True)
    e.add_argument('--shard', type=int, required=True); e.add_argument('--shards', type=int, default=4)
    t = sub.add_parser('train'); t.add_argument('--features', required=True); t.add_argument('--diagnostic-root')
    for parser in (e, t):
        parser.add_argument('--probe-root', required=True); parser.add_argument('--output', required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-vjepa-model-retraining-anchored-v1':
        raise ValueError('unexpected model training protocol')
    globals()[args.command](args, config)


if __name__ == '__main__':
    main()
