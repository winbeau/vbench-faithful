"""Fresh frozen-model inference of the anchored head on all 450 TEST sources.

Exactly the previous 1800 original/control/8px-CF inputs. No training, filtering,
new construction or posthoc mapping. Old Origin predictions are pinned and reused.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .expand_vjepa_validation import load_predictions, validate_combinations
from .static_jitter import ROOT, digest
from .validate_vjepa_motion import check_sha
from .vjepa_motion_probe import fresh_output, gpu_precheck, read_rows, write_json


def numpy_sha(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return hashlib.sha256(buffer.getbuffer()).hexdigest()


def previous_data(root, config):
    root = Path(root)
    path = root / 'inputs/inputs.jsonl'
    check_sha(path, config['inputs_sha256'])
    previous_config = root / 'code/configs/dynamic-static-jitter/vjepa-expansion450-v1.json'
    check_sha(previous_config, config['expansion_config_sha256'])
    inputs = read_rows(path)
    bases = {r['base_id'] for r in inputs}
    if len(inputs) != 1800 or len(bases) != 450 or len({r['prompt_id'] for r in inputs}) != 30:
        raise ValueError('fixed cohort coverage changed')
    validate_combinations(inputs, bases)
    for backend, shards in config['previous_scores_sha256'].items():
        for shard, checksum in shards.items():
            check_sha(root / 'scores' / backend / shard / 'scores.jsonl', checksum)
    output, _ = load_predictions(inputs, root / 'scores', previous_config)
    return inputs, output


def identity():
    files = [Path(__file__), ROOT / 'metrics/dynamic-degree/src/dynamic_degree/learned_probe.py',
             ROOT / 'packages/audit-models/src/vbench_audit_models/vjepa.py']
    files += [Path(__file__).with_name(name + '.py') for name in
              ('expand_vjepa_validation', 'validate_vjepa_motion', 'vjepa_motion_probe',
               'official_video_jitter', 'local_texture_jitter', 'static_jitter')]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'previous-root', 'probe-root', 'trained-root', 'video-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--shard', type=int, required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-vjepa-anchored-frozen-test450-v1' or args.shard not in range(4):
        raise ValueError('unexpected protocol/shard')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import cv2
    import torch
    from dynamic_degree.learned_probe import MotionProbe
    from vbench_audit_models.vjepa import FrozenVJEPA
    from .official_video_jitter import native_video

    start = time.monotonic(); gpu = gpu_precheck()
    torch.set_num_threads(3); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs, prior = previous_data(args.previous_root, config)
    bases = sorted({r['base_id'] for r in inputs})
    chosen = set(bases[args.shard::4])
    selected = [r for r in inputs if r['base_id'] in chosen]
    probe, trained = Path(args.probe_root), Path(args.trained_root)
    pc_path = ROOT / 'configs/dynamic-static-jitter/vjepa-probe-v1.json'
    check_sha(pc_path, config['probe_config_sha256'])
    pc = json.loads(pc_path.read_text())
    tc_path = trained / 'code/configs/dynamic-static-jitter/vjepa-anchored-v1.json'
    check_sha(tc_path, config['training_config_sha256'])
    check_sha(trained / 'training-chunked/summary.json', config['training_summary_sha256'])
    tc = json.loads(tc_path.read_text())
    check_sha(ROOT / 'metrics/dynamic-degree/src/dynamic_degree/learned_probe.py', tc['head_implementation_sha256'])
    old_pc = json.loads((Path(args.previous_root) / 'code/configs/dynamic-static-jitter/vjepa-validation-v1.json').read_text())
    for name, checksum in old_pc['implementation_sha256'].items():
        check_sha(ROOT / name, checksum)
    out = fresh_output(args.output)
    encoder = FrozenVJEPA(probe / 'assets/vjepa2', probe / 'assets/vjepa2_1_vitb_dist_vitG_384.pt', pc)
    heads = {}
    for arm, path, checksum in [('anchored', trained / 'training-chunked/anchored.pt', config['head_sha256']),
                                ('old_joint', probe / 'training/joint.pt', config['old_head_sha256'])]:
        check_sha(path, checksum)
        model = MotionProbe()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        heads[arm] = model.eval().requires_grad_(False).to('cuda:0')
    # Pre-test parity uses a prior training example, not a new test selection.
    reference = next(r for r in read_rows(probe / 'features/shard-0/features.jsonl') if r['view'] == 'base')
    video = Path(args.video_root) / reference['relative_video_path']
    check_sha(video, reference['video_sha256'])
    frames, _, _ = native_video(video, 1e-6)
    features = encoder.encode(frames)
    if numpy_sha(features) != reference['feature_sha256']:
        raise ValueError('encoder loading differs from prior feature cache')
    saved = {r['arm']: r for r in read_rows(trained / 'training-chunked/scores.jsonl')
             if r['video_uid'] == reference['video_uid']}
    parity = {}
    with torch.inference_mode():
        x = torch.from_numpy(features).to('cuda:0', torch.float32)[None]
        for arm, head in heads.items():
            error = abs(float(head(x)[0]) - saved[arm]['latent'][0])
            if error > config['inference']['head_tolerance']:
                raise ValueError('pretest head loading check failed')
            parity[arm] = error
    write_json(out / 'provenance.json', {'config_sha256': digest(Path(args.config)), 'code': identity(),
        'input_sha256': config['inputs_sha256'], 'head_sha256': config['head_sha256'],
        'old_head_sha256': config['old_head_sha256'], 'encoder': encoder.identity,
        'gpu': gpu, 'shard': args.shard, 'shards': 4, 'training_updates': 0, 'score_mapping_changed': False,
        'mode': 'fresh_decode_and_encoder_forward_every_input', 'origin': 'reuse_pinned_previous_scores',
        'pretest_loading_check': {'uid': reference['video_uid'], 'feature_exact': True, 'latent_error': parity},
        'expected_ids': [r['evaluation_id'] for r in selected], 'torch': torch.__version__,
        'numpy': np.__version__, 'opencv': cv2.__version__,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    failed = 0; feature_exact = 0; max_old_error = 0.
    with (out / 'scores.jsonl').open('x') as handle:
        for i, row in enumerate(selected):
            result = {k: row[k] for k in ('evaluation_id', 'base_id', 'family', 'seed', 'prompt_id', 'generator', 'construction_group')}
            try:
                path = Path(row['video']); check_sha(path, row['sha256'])
                frames, pts, fps = native_video(path, 1e-6)
                if (hashlib.sha256(frames.tobytes()).hexdigest() != row['decoded_pixels_sha256']
                        or list(frames.shape) != row['decoded_shape'] or len(frames) != 16 or abs(fps - 8) > 1e-6
                        or not np.allclose(pts, row['pts'], rtol=0, atol=1e-6)):
                    raise ValueError('native decoded pixels or timeline changed')
                features = encoder.encode(frames)
                checksum = numpy_sha(features)
                old = prior['vjepa'][row['evaluation_id']]
                if checksum != old['feature_sha256']:
                    raise ValueError('recomputed feature differs from prior run')
                feature_exact += 1
                with torch.inference_mode():
                    x = torch.from_numpy(features).to('cuda:0', torch.float32)[None]
                    for arm, head in heads.items():
                        q = float(head(x)[0])
                        if not np.isfinite(q):
                            raise ValueError('nonfinite model score')
                        result[arm] = {'latent': q, 'score': float(1 / (1 + np.exp(-q)))}
                error = abs(result['old_joint']['latent'] - old['joint']['latent'])
                if error > config['inference']['head_tolerance']:
                    raise ValueError('old-head replay differs from previous scores')
                max_old_error = max(max_old_error, error)
                result.update(status='ok', input_sha256=row['sha256'], feature_sha256=checksum,
                              pixels_sha256=row['decoded_pixels_sha256'], old_head_replay_error=error,
                              construction_status=row['status'], construction_reason=row.get('reason'))
            except Exception as exc:
                failed += 1
                result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            handle.write(json.dumps(result, allow_nan=False) + '\n'); handle.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(selected):
                print(json.dumps({'shard': args.shard, 'completed': i + 1, 'expected': len(selected), 'failed': failed}), flush=True)
            if i == 0 and failed:
                raise RuntimeError(result['error'])
    write_json(out / 'completion.json', {'status': 'finished' if not failed else 'failed', 'expected': len(selected),
        'completed': len(selected), 'failed': failed, 'feature_exact_to_previous': feature_exact,
        'old_head_max_latent_error': max_old_error, 'scores_sha256': digest(out / 'scores.jsonl'),
        'wall_seconds': time.monotonic() - start,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'completion.json').read_text(), flush=True)
    if failed:
        raise RuntimeError('retain failed inputs; no silent exclusion')


if __name__ == '__main__':
    main()
