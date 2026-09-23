"""One bounded frozen-video-encoder experiment: select, extract, train.

No main32/holdout training, no pretrained-network updates, no automatic downloads,
no checkpoint sweep. New outputs only; all failures remain in the feature ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .static_jitter import ROOT, digest


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def write_rows(path, rows):
    with Path(path).open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')


def fresh_output(path):
    path = Path(path).resolve()
    if path == ROOT or any(path.is_relative_to(ROOT / p) for p in ('data', 'results', 'splits', 'runs')):
        raise ValueError('output must not be workspace root or a frozen tree')
    path.mkdir(parents=True, exist_ok=False)
    return path


def code_identity():
    paths = [Path(__file__), Path(__file__).with_name('local_texture_jitter.py'),
             Path(__file__).with_name('official_video_jitter.py'), Path(__file__).with_name('static_jitter.py'),
             ROOT / 'packages/audit-models/src/vbench_audit_models/vjepa.py',
             ROOT / 'metrics/dynamic-degree/src/dynamic_degree/learned_probe.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def select_sources(pool, excluded, config):
    forbidden_prompts = {r['prompt_id'] for r in excluded}
    forbidden_uids = {r['video_uid'] for r in excluded}
    sources = [dict(r) for r in pool if r['dimension'] == 'dynamics_degree' and r['split'] == 'dev'
               and r['relative_video_path'].endswith('.mp4') and r['prompt_id'] not in forbidden_prompts
               and r['video_uid'] not in forbidden_uids]
    prompts = sorted({r['prompt_id'] for r in sources}, key=lambda p: hashlib.sha256(
        (config['selection_prefix'] + p).encode()).hexdigest())
    if len(prompts) != sum(config['prompt_split_counts'].values()):
        raise ValueError('eligible prompt count changed')
    roles, offset = {}, 0
    for role, count in config['prompt_split_counts'].items():
        roles.update({p: role for p in prompts[offset:offset + count]})
        offset += count
    for source in sources:
        source['role'] = roles[source['prompt_id']]
    if len(sources) != config['expected_sources'] or len({r['video_uid'] for r in sources}) != len(sources):
        raise ValueError('source coverage changed or duplicate UID')
    return sorted(sources, key=lambda r: (r['role'], r['video_uid']))


def select_pairs(rows, sources):
    by_uid = {r['video_uid']: r for r in sources}
    output = []
    for row in rows:
        # Do not consult test labels, other dimensions, or unused pair targets.
        if row['dimension'] != 'dynamics_degree' or row['split'] != 'dev':
            continue
        a, b = row['video_a_uid'], row['video_b_uid']
        if a not in by_uid or b not in by_uid:
            continue
        if by_uid[a]['role'] != by_uid[b]['role'] or by_uid[a]['prompt_id'] != row['prompt_id'] or by_uid[b]['prompt_id'] != row['prompt_id']:
            raise ValueError('pair crosses roles or prompts')
        if by_uid[a]['role'] == 'calibration_reserved':
            continue
        label = float(row['human_label'])
        if label not in (0., .5, 1.):
            raise ValueError('invalid dynamic preference')
        output.append({'a': a, 'b': b, 'label': label, 'prompt_id': row['prompt_id'],
                       'role': by_uid[a]['role'], 'label_source': row['label_source']})
    identities = {(r['a'], r['b']) for r in output}
    if len(identities) != len(output):
        raise ValueError('duplicate pair')
    return output


def select(args, config):
    for path, expected in [(args.pool, config['pool_sha256']), (args.pairs, config['pairs_sha256'])]:
        if digest(Path(path)) != expected:
            raise ValueError(f'input changed: {path}')
    if sorted(digest(Path(p)) for p in args.exclude) != sorted(config['exclude_sha256']):
        raise ValueError('exclusion manifests changed')
    with Path(args.pool).open() as handle:
        sources = select_sources(csv.DictReader(handle), [r for p in args.exclude for r in read_rows(p)], config)
    with Path(args.pairs).open() as handle:
        pairs = select_pairs(csv.DictReader(handle), sources)
    out = fresh_output(args.output)
    write_rows(out / 'sources.jsonl', sources)
    write_rows(out / 'pairs.jsonl', pairs)
    receipt = {'protocol': config['protocol'], 'config_sha256': digest(Path(args.config)),
               'code_sha256': code_identity(), 'sources_sha256': digest(out / 'sources.jsonl'),
               'pairs_sha256': digest(out / 'pairs.jsonl'), 'source_counts': dict(Counter(r['role'] for r in sources)),
               'pair_counts': {role: dict(Counter(str(r['label']) for r in pairs if r['role'] == role))
                               for role in ('train', 'validation')},
               'scores_read_for_selection': False, 'formal_holdout_labels_used': False,
               'holdout_media_opened': False, 'calibration_labels_exported': False,
               'prior_exposure': 'development videos; includes 63 previously used natural calibration sources',
               'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(out / 'selection.json', receipt)
    print(json.dumps(receipt), flush=True)


def augment(frames, uid, view, config):
    import cv2
    from .local_texture_jitter import displacement_field, alternating_phase

    c = config['augmentation']
    seed = int(hashlib.sha256((c['seed_prefix'] + uid + ':' + view).encode()).hexdigest()[:8], 16)
    n, h, w, _ = frames.shape
    field = displacement_field(h, w, scale=c['spatial_scale_pixels'], border=c['border_taper_pixels'], seed=seed)
    field *= c['amplitude_pixels']
    phase = alternating_phase(n, seed)
    if view == 'aperiodic':
        phase *= np.random.default_rng(seed ^ 0x5A17).uniform(*c['aperiodic_amplitude_range'], size=n).astype(np.float32)
        phase -= phase.mean()
        phase /= np.abs(phase).max()
    elif view != 'alternating':
        raise ValueError('unknown jitter view')
    y, x = np.mgrid[:h, :w].astype(np.float32)
    output, inside, jacobians = [], True, []
    dx_y, dx_x = np.gradient(field[..., 0])
    dy_y, dy_x = np.gradient(field[..., 1])
    for frame, a in zip(frames, phase):
        mx, my = x + a * field[..., 0], y + a * field[..., 1]
        inside &= bool(mx.min() >= 0 and mx.max() <= w - 1 and my.min() >= 0 and my.max() <= h - 1)
        output.append(cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
        jacobians.append(float(((1+a*dx_x)*(1+a*dy_y) - a*a*dx_y*dy_x).min()))
    result = np.stack(output)
    if not inside or len(result) != len(frames):
        raise ValueError('augmentation geometry/timeline failure')
    info = {'seed': seed, 'view': view, 'native_amplitude_pixels': c['amplitude_pixels'],
            'field_sha256': hashlib.sha256(field.tobytes()).hexdigest(), 'phase': phase.tolist(),
            'minimum_warp_jacobian': min(jacobians), 'jacobian_below_legacy_half': min(jacobians) < .5,
            'max_displacement_pixels': float(np.linalg.norm(field, axis=-1).max()),
            'inside': inside, 'quality_excluded': False, 'human_reviewed': False,
            'mean_pixel_absolute_change': float(np.abs(result.astype(float)-frames).mean())}
    return result, info


def gpu_precheck():
    import torch

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible or torch.cuda.device_count() != 1:
        raise ValueError('one UUID-bound visible GPU required')
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    row = next((line.split(',') for line in lines.splitlines() if visible in line), None)
    if row is None or int(row[0]) not in (4, 5, 6, 7) or int(row[2]) < 40960:
        raise ValueError('physical GPU 4-7 with >=40 GiB free required')
    return {'physical_index': int(row[0]), 'uuid': visible, 'free_mib_before': int(row[2]),
            'logical_device': 'cuda:0', 'name': torch.cuda.get_device_name(0)}


def extract(args, config):
    import cv2
    import torch
    from vbench_audit_models.vjepa import FrozenVJEPA
    from .official_video_jitter import native_video

    start = time.monotonic()
    gpu = gpu_precheck()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard index')
    selection = json.loads((Path(args.selection) / 'selection.json').read_text())
    if (selection['config_sha256'] != digest(Path(args.config))
            or selection['sources_sha256'] != digest(Path(args.selection) / 'sources.jsonl')):
        raise ValueError('selection/config identity mismatch')
    sources = [r for r in read_rows(Path(args.selection) / 'sources.jsonl') if r['role'] in config['extract_roles']]
    if len(sources) != config['expected_extract_sources']:
        raise ValueError('unexpected extraction coverage')
    sources = sources[args.shard::args.shards]
    out = fresh_output(args.output)
    (out / 'features').mkdir()
    encoder = FrozenVJEPA(args.source_root, args.checkpoint, config)
    write_json(out / 'provenance.json', {'protocol': config['protocol'], 'config_sha256': digest(Path(args.config)),
               'code_sha256': code_identity(), 'encoder': encoder.identity, 'gpu': gpu,
               'selection_sha256': digest(Path(args.selection) / 'selection.json'),
               'shard': args.shard, 'shards': args.shards, 'opencv': cv2.__version__, 'numpy': np.__version__,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    root, successes, failures, parity = Path(args.video_root).resolve(), 0, 0, None
    with (out / 'features.jsonl').open('x') as ledger:
        for i, source in enumerate(sources):
            row = dict(source)
            try:
                path = (root / source['relative_video_path']).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError('media path escapes official root')
                frames, pts, fps = native_video(path, 1e-6)
                if len(frames) != 16 or abs(fps - 8) > 1e-6 or frames.shape[1] != frames.shape[2]:
                    raise ValueError('expected original square 16-frame/8-FPS video')
                identity = {'video_sha256': digest(path), 'decoded_shape': list(frames.shape), 'fps': fps, 'pts': pts,
                            'media_duration_seconds': len(frames)/fps, 'native_frame_indices': list(range(len(frames)))}
                for view in ['base'] + config['augmentation']['views']:
                    edited, geometry = (frames, None) if view == 'base' else augment(frames, source['video_uid'], view, config)
                    features = encoder.encode(edited)
                    if parity is None:
                        again = encoder.encode(edited)
                        error = float(np.abs(features.astype(float)-again.astype(float)).max())
                        parity = {'repeat_max_abs_error': error, 'identical': bool(np.array_equal(features, again)),
                                  'tested_uid': source['video_uid'], 'view': view,
                                  'scope': 'same pinned encoder, input and preprocessing; not an external action benchmark'}
                        write_json(out / 'loading_check.json', parity)
                        if error != 0:
                            raise ValueError('feature repeatability failed')
                    feature_path = out / 'features' / f"{source['video_uid']}.{view}.npy"
                    with feature_path.open('xb') as handle:
                        np.save(handle, features, allow_pickle=False)
                    record = dict(row, **identity, view=view, geometry=geometry, status='ok',
                                  pixels_sha256=hashlib.sha256(edited.tobytes()).hexdigest(),
                                  feature_path=str(feature_path), feature_sha256=digest(feature_path),
                                  feature_shape=list(features.shape), feature_dtype=str(features.dtype))
                    ledger.write(json.dumps(record, allow_nan=False) + '\n')
                    ledger.flush()
                    successes += 1
            except Exception as exc:
                failures += 1
                ledger.write(json.dumps(dict(row, status='failed', error=f'{type(exc).__name__}: {exc}')) + '\n')
                ledger.flush()
                print(json.dumps({'failure_uid': source['video_uid'], 'error': str(exc)}), flush=True)
                if i == 0:
                    raise  # Do not repeat a loading/preprocessing failure over the cohort.
            if (i+1) % 10 == 0 or i+1 == len(sources):
                print(json.dumps({'shard': args.shard, 'sources_done': i+1, 'sources_total': len(sources), 'features': successes, 'failures': failures}), flush=True)
    write_json(out / 'completion.json', {'status': 'finished' if not failures else 'failed', 'features': successes,
               'failed_sources': failures, 'sources': len(sources), 'wall_seconds': time.monotonic()-start,
               'ledger_sha256': digest(out / 'features.jsonl'), 'parity': parity,
               'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    if failures:
        raise RuntimeError('incomplete feature cohort; no training permitted')


def statistics(sources, pairs, latent):
    index = {r['video_uid']: i for i, r in enumerate(sources)}
    scores = 1 / (1 + np.exp(-latent))
    labels = np.asarray([r['label'] for r in pairs])
    a, b = (np.asarray([index[r[k]] for r in pairs]) for k in ('a', 'b'))
    ordered = labels != .5
    differences = latent[a] - latent[b]
    margins = differences[ordered] * (2*labels[ordered, None]-1)
    def by_scale(x):
        delta = x[:, 1:] - x[:, :1]
        return {'base_mean': float(x[:, 0].mean()), 'cf_mean': float(x[:, 1:].mean()),
                'signed_change': float(delta.mean()), 'mean_absolute_change': float(abs(delta).mean()),
                'base_std': float(x[:, 0].std()), 'base_min': float(x[:, 0].min()), 'base_max': float(x[:, 0].max()),
                'max_absolute_change': float(abs(delta).max()),
                'p95_absolute_change': float(np.quantile(abs(delta), .95)),
                'per_prompt': {p: {'signed_change': float(delta[[r['prompt_id']==p for r in sources]].mean()),
                                  'mean_absolute_change': float(abs(delta[[r['prompt_id']==p for r in sources]]).mean())}
                               for p in sorted({r['prompt_id'] for r in sources})}}
    return {'sources': len(sources), 'pairs': len(pairs), 'ordered_pairs': int(ordered.sum()),
            'ties': int((~ordered).sum()), 'ordered_correct_by_view': (margins > 0).sum(axis=0).tolist(),
            'mean_signed_ordered_margin_by_view': margins.mean(axis=0).tolist(),
            'mean_absolute_tie_gap_by_view': abs(differences[~ordered]).mean(axis=0).tolist(),
            'latent': by_scale(latent), 'relative_sigmoid': by_scale(scores)}


def feasibility(natural, joint, gates):
    nc, jc = natural['ordered_correct_by_view'], joint['ordered_correct_by_view']
    nl, jl = natural['latent'], joint['latent']
    checks = {'clean_ordering': jc[0] >= gates['minimum_validation_ordered_correct'],
              'clean_ordering_retained': jc[0] >= nc[0]-gates['maximum_clean_correct_loss_vs_natural'],
              'jitter_ordering_retained': min(jc[1:]) >= jc[0]-gates['maximum_jitter_correct_loss_vs_clean'],
              'jitter_error_reduced': jl['mean_absolute_change'] <= gates['maximum_mae_ratio_vs_natural']*nl['mean_absolute_change'],
              'not_constant': jl['base_std'] >= gates['minimum_latent_std'],
              'range_retained': jl['base_std'] >= gates['minimum_std_ratio_vs_natural']*nl['base_std'],
              'motion_margin': joint['mean_signed_ordered_margin_by_view'][0] >= gates['minimum_mean_signed_ordered_margin']}
    return {'checks': checks, 'passed': all(checks.values()), 'formal_acceptance': 'NOT EVALUATED: no absolute calibration, no main32/test inference'}


def train(args, config):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from dynamic_degree.learned_probe import MotionProbe, motion_losses

    start, gpu = time.monotonic(), gpu_precheck()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['optimization']['seed'])
    selection_dir = Path(args.selection)
    selection = json.loads((selection_dir / 'selection.json').read_text())
    if (selection['sources_sha256'] != digest(selection_dir / 'sources.jsonl')
            or selection['pairs_sha256'] != digest(selection_dir / 'pairs.jsonl')
            or selection['config_sha256'] != digest(Path(args.config))):
        raise ValueError('selection identity mismatch')
    sources = read_rows(selection_dir / 'sources.jsonl')
    pairs = read_rows(selection_dir / 'pairs.jsonl')
    features, shard_receipts = {}, []
    for shard in sorted(Path(args.features).glob('shard-*')):
        completion = json.loads((shard / 'completion.json').read_text())
        provenance = json.loads((shard / 'provenance.json').read_text())
        if (completion['status'] != 'finished' or completion['ledger_sha256'] != digest(shard / 'features.jsonl')
                or provenance['config_sha256'] != digest(Path(args.config)) or provenance['code_sha256'] != code_identity()
                or provenance['selection_sha256'] != digest(selection_dir / 'selection.json')
                or not completion['parity']['identical']):
            raise ValueError('feature receipt mismatch/incomplete/parity failure')
        shard_receipts.append({'path': str(shard), 'completion': completion, 'provenance_sha256': digest(shard / 'provenance.json')})
        for row in read_rows(shard / 'features.jsonl'):
            key = row['video_uid'], row.get('view')
            if row['status'] != 'ok' or key in features:
                raise ValueError('failed/duplicate feature')
            features[key] = row
    expected = {(r['video_uid'], v) for r in sources if r['role'] in config['extract_roles']
                for v in ['base']+config['augmentation']['views']}
    if set(features) != expected or len(features) != config['expected_features']:
        raise ValueError('feature coverage mismatch')
    out = fresh_output(args.output)
    write_json(out / 'provenance.json', {'protocol': config['protocol'], 'config_sha256': digest(Path(args.config)),
               'code_sha256': code_identity(), 'gpu': gpu, 'torch': torch.__version__, 'numpy': np.__version__,
               'selection_sha256': digest(selection_dir / 'selection.json'), 'shards': shard_receipts,
               'optimization': config['optimization'], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})

    def load_features(role):
        selected = [r for r in sources if r['role'] == role]
        views = ['base']+config['augmentation']['views']
        array = np.empty((len(selected)*3, 4608, 768), dtype=np.float16)
        for i, source in enumerate(selected):
            for j, view in enumerate(views):
                record = features[source['video_uid'], view]
                path = Path(record['feature_path'])
                if digest(path) != record['feature_sha256']:
                    raise ValueError('feature bytes changed')
                array[i*3+j] = np.load(path, allow_pickle=False)
        if not np.isfinite(array).all():
            raise ValueError('nonfinite cached features')
        return selected, torch.from_numpy(array).to('cuda:0', torch.float32)

    train_sources, train_x = load_features('train')
    train_pairs = [r for r in pairs if r['role'] == 'train']
    uid_index = {r['video_uid']: i for i, r in enumerate(train_sources)}
    pair_indices = torch.tensor([[uid_index[r['a']], uid_index[r['b']]] for r in train_pairs], device='cuda:0')
    targets = torch.tensor([r['label'] for r in train_pairs], device='cuda:0')
    hc, opt = config['head'], config['optimization']
    initial = MotionProbe(hc['input_dim'], hc['hidden_dim'], hc['mlp_dim'])
    initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
    write_json(out / 'head.json', {'parameters': sum(p.numel() for p in initial.parameters()), 'encoder_updates': 0,
               'architecture': hc, 'fixed_steps': opt['steps'], 'validation_checkpoint_selection': False})
    models, training_curves = {}, {}
    # Both arms start from identical parameters, see exactly the same natural labels,
    # and receive exactly 100 optimizer updates. Validation is not read in this loop.
    for arm, weight in opt['arms'].items():
        model = MotionProbe(hc['input_dim'], hc['hidden_dim'], hc['mlp_dim']).to('cuda:0')
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
        x = train_x if weight else train_x[::3].contiguous()
        curve = []
        for step in range(opt['steps']):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            latent = model(x).reshape(len(train_sources), 3 if weight else 1)
            loss, parts = motion_losses(latent, pair_indices, targets, weight, opt['gauge_weight'])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training objective')
            loss.backward()
            optimizer.step()
            curve.append({'step': step+1, 'loss': float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}})
            if (step+1) % 10 == 0:
                print(json.dumps({'arm': arm, **curve[-1]}), flush=True)
        model.eval().requires_grad_(False)
        models[arm], training_curves[arm] = model, curve
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / f'{arm}.pt')
    write_json(out / 'training_curves.json', training_curves)
    del x, optimizer, latent, loss, parts
    initial.load_state_dict(initial_state, strict=True)
    models['untrained'] = initial.to('cuda:0').eval().requires_grad_(False)
    evaluations, score_rows = {}, []
    for role in ('train', 'validation'):
        if role == 'train':
            cohort, x = train_sources, train_x
        else:
            del train_x, x
            torch.cuda.empty_cache()
            cohort, x = load_features('validation')
        role_pairs = [r for r in pairs if r['role'] == role]
        evaluations[role] = {}
        for arm, model in models.items():
            with torch.inference_mode():
                q = model(x).reshape(len(cohort), 3).cpu().numpy().astype(float)
            evaluations[role][arm] = statistics(cohort, role_pairs, q)
            for source, values in zip(cohort, q):
                score_rows.append({'video_uid': source['video_uid'], 'prompt_id': source['prompt_id'],
                                   'role': role, 'arm': arm, 'views': ['base']+config['augmentation']['views'],
                                   'latent': values.tolist(), 'relative_sigmoid': (1/(1+np.exp(-values))).tolist()})
        evaluations[role]['constant_baseline'] = statistics(cohort, role_pairs, np.zeros((len(cohort), 3)))
    write_rows(out / 'scores.jsonl', score_rows)
    result = {'status': 'finished', 'evaluations': evaluations,
              'feasibility': feasibility(evaluations['validation']['natural_only'], evaluations['validation']['joint'], config['feasibility_gate']),
              'wall_seconds': time.monotonic()-start, 'scores_sha256': digest(out / 'scores.jsonl'),
              'head_sha256': {arm: digest(out / f'{arm}.pt') for arm in opt['arms']},
              'limitations': ['relative pairwise supervision, no absolute motion anchors',
                  '4 validation prompts only, previously exposed development population',
                  'possible content/generator confounding', 'new jitter views not human-reviewed',
                  'no calibrated comparison with Origin or the main32 10-percent gate',
                  'no formal holdout or calibration-reserved inference'],
              'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(out / 'summary.json', result)
    print(json.dumps({'status': result['status'], 'feasibility': result['feasibility'], 'wall_seconds': result['wall_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('select')
    p.add_argument('--pool', required=True)
    p.add_argument('--pairs', required=True)
    p.add_argument('--exclude', nargs='+', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--selection', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=4)
    p = sub.add_parser('train')
    p.add_argument('--selection', required=True)
    p.add_argument('--features', required=True)
    for p in sub.choices.values():
        p.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-vjepa-frozen-probe-v1':
        raise ValueError('unexpected experiment protocol')
    globals()[args.command](args, config)


if __name__ == '__main__':
    main()
