"""Reload both saved heads and replay every cached DEV/control prediction.

Read-only model verification: no encoder inference, optimizer, threshold tuning,
or access to TEST/calibration media. The output directory must be fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare(actual, reference, tolerance=1e-6):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('prediction shape or finite check failed')
    error = float(np.abs(actual - reference).max())
    if error > tolerance:
        raise ValueError(f'reloaded prediction disagrees: {error}')
    return error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'probe-root', 'diagnostic-root', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    root, probe, diagnostic, out = map(Path, (args.root, args.probe_root, args.diagnostic_root, args.output))
    out = out.resolve()
    repo = Path(__file__).resolve().parents[2]
    if out in (root.resolve(), repo) or any(out.is_relative_to(repo / n) for n in ('data', 'results', 'splits', 'runs')):
        raise ValueError('cannot write frozen trees')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from dynamic_degree.learned_probe import MotionProbe

    if torch.cuda.device_count() != 1:
        raise ValueError('mask exactly one GPU')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((root / 'code/configs/dynamic-static-jitter/vjepa-anchored-v1.json').read_text())
    summary = json.loads((root / 'training-chunked/summary.json').read_text())
    if sha(root / 'code/metrics/dynamic-degree/src/dynamic_degree/learned_probe.py') != config['head_implementation_sha256']:
        raise ValueError('model implementation changed')
    feature_index = {}
    for feature_root in (probe / 'features', root / 'features'):
        for path in sorted(feature_root.glob('shard-*/features.jsonl')):
            completion = json.loads((path.parent / 'completion.json').read_text())
            if completion['status'] != 'finished' or completion['failed_sources'] or sha(path) != completion['ledger_sha256']:
                raise ValueError('feature receipts failed')
            for r in rows(path):
                key = r['video_uid'], r['view']
                if key in feature_index or r['status'] != 'ok':
                    raise ValueError('duplicate or failed feature')
                feature_index[key] = r
    if len(feature_index) != 2160:
        raise ValueError('incomplete cached views')
    weights = {'old_joint': probe / 'training/joint.pt', 'anchored': root / 'training-chunked/anchored.pt'}
    expected = {'old_joint': config['initial_joint_head_sha256'], 'anchored': summary['head_sha256']}
    states, heads = {}, {}
    for arm, path in weights.items():
        if sha(path) != expected[arm]:
            raise ValueError('checkpoint hash mismatch')
        states[arm] = torch.load(path, map_location='cpu', weights_only=True)
        heads[arm] = MotionProbe().to('cuda:0')
        heads[arm].load_state_dict(states[arm], strict=True)
        heads[arm].eval().requires_grad_(False)
    parameter_changes = {k: float((v - states['old_joint'][k]).double().norm()) for k, v in states['anchored'].items()}
    if not all(np.isfinite(v) and v > 0 for v in parameter_changes.values()):
        raise ValueError('expected finite, genuinely retrained parameter tensors')
    records = rows(root / 'training-chunked/scores.jsonl')
    if len(records) != 540 or sha(root / 'training-chunked/scores.jsonl') != summary['scores_sha256']:
        raise ValueError('incomplete predictions')
    references = {}
    for r in records:
        for view, q, s in zip(r['views'], r['latent'], r['scores']):
            key = r['video_uid'], view
            references.setdefault(key, {})[r['arm']] = q, s
    if set(references) != set(feature_index) or any(set(r) != set(heads) for r in references.values()):
        raise ValueError('prediction/view coverage mismatch')
    keys = sorted(references)
    max_q_error = max_s_error = 0.
    values_verified = 0
    def tensor_for(record):
        if sha(record['feature_path']) != record['feature_sha256']:
            raise ValueError('feature bytes changed')
        array = np.load(record['feature_path'], allow_pickle=False)
        if array.shape != (4608, 768) or not np.isfinite(array).all():
            raise ValueError('invalid cached feature')
        return array
    with torch.inference_mode():
        for offset in range(0, len(keys), 24):
            selected = keys[offset:offset + 24]
            x = torch.from_numpy(np.stack([tensor_for(feature_index[k]) for k in selected])).to('cuda:0', torch.float32)
            for arm, head in heads.items():
                q = head(x).cpu().numpy().astype(float)
                reference = np.array([references[k][arm] for k in selected])
                max_q_error = max(max_q_error, compare(q, reference[:, 0]))
                max_s_error = max(max_s_error, compare(1 / (1 + np.exp(-q)), reference[:, 1]))
                values_verified += len(q)
            if (offset + len(selected)) % 480 == 0:
                print(json.dumps({'views_replayed': offset + len(selected)}), flush=True)
        controls = []
        reference_controls = {r['case']: r for r in summary['user_requested_control_after_freeze']}
        for r in rows(diagnostic / 'scores/vjepa/scores.jsonl'):
            x = torch.from_numpy(tensor_for(r)).to('cuda:0', torch.float32)[None]
            q = float(heads['anchored'](x)[0]); s = float(1 / (1 + np.exp(-q)))
            ref = reference_controls[r['family']]
            max_q_error = max(max_q_error, compare(q, ref['latent']))
            max_s_error = max(max_s_error, compare(s, ref['anchored']))
            controls.append({'case': r['family'], 'latent': q, 'score': s})
    if len(controls) != 5 or values_verified != 4320:
        raise ValueError('incomplete replay')
    receipt = {'status': 'passed', 'trained_head_sha256': expected['anchored'], 'old_head_sha256': expected['old_joint'],
        'parameter_tensor_l2_changes': parameter_changes, 'parameter_count': sum(v.numel() for v in states['anchored'].values()),
        'cached_views_checked': len(keys), 'dev_head_predictions_replayed': values_verified, 'controls_replayed': controls,
        'maximum_latent_error': max_q_error, 'maximum_score_error': max_s_error, 'tolerance': 1e-6,
        'training_updates': 0, 'encoder_inference': 0, 'calibration_or_test_media_read': False,
        'device': torch.cuda.get_device_name(0), 'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'script_sha256': sha(__file__), 'wall_seconds': time.monotonic() - start,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with (out / 'receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
