"""Fresh-process checkpoint replay on all 480 DEV60 views and five controls.

Frozen cached encoder tokens only; no training, TEST450 or score mapping fit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .audit_vjepa_motion_probe import rows, sha
from .vjepa_motion_probe import fresh_output, gpu_precheck, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'probe-root', 'anchored-root', 'diagnostic-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from dynamic_degree.learned_probe import MotionProbe
    start = time.monotonic(); gpu = gpu_precheck()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root, probe, anchored = Path(args.root), Path(args.probe_root), Path(args.anchored_root)
    config = json.loads((root / 'code/configs/dynamic-static-jitter/vjepa-aligned-v1.json').read_text())
    summary = json.loads((root / 'training/summary.json').read_text())
    assert sha(root / 'training/scores.jsonl') == summary['scores_sha256']
    assert sha(probe / 'selection/sources.jsonl') == config['source_manifest_sha256']
    records = [r for r in rows(root / 'training/scores.jsonl') if r['role'] == 'validation']
    assert len(records) == 120
    features = {}
    for directory in (probe / 'features', anchored / 'features'):
        for path in sorted(directory.glob('shard-*/features.jsonl')):
            receipt = json.loads((path.parent / 'completion.json').read_text())
            assert receipt['status'] == 'finished' and sha(path) == receipt['ledger_sha256']
            for r in rows(path):
                key = (r['video_uid'], r['view']); assert key not in features
                features[key] = r
    models = {}
    for arm, path, checksum in [('previous_anchored', anchored / 'training-chunked/anchored.pt', config['initial_head_sha256']),
                                ('aligned', root / 'training/aligned.pt', summary['head_sha256'])]:
        assert sha(path) == checksum
        head = MotionProbe().to('cuda:0')
        head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        models[arm] = head.eval().requires_grad_(False)
    maximum_error = 0.; checked = 0; feature_shas = {}
    with torch.inference_mode():
        for offset in range(0, len(records), 3):
            # Three source rows contain exactly 24 views, matching training's tensor shape.
            batch = records[offset:offset + 3]; assert len({r['arm'] for r in batch}) == 1
            arrays = []; reference = []
            for r in batch:
                for view in r['views']:
                    f = features[r['video_uid'], view]
                    assert sha(f['feature_path']) == f['feature_sha256']
                    array = np.load(f['feature_path'], allow_pickle=False)
                    assert array.shape == (4608, 768) and np.isfinite(array).all()
                    arrays.append(array); feature_shas[r['video_uid'] + '/' + view] = f['feature_sha256']
                reference.extend(r['latent'])
            x = torch.from_numpy(np.stack(arrays)).to('cuda:0', torch.float32)
            result = models[batch[0]['arm']](x).cpu().numpy().astype(float)
            error = float(abs(result - np.array(reference)).max()); assert error <= 1e-6
            maximum_error = max(maximum_error, error); checked += len(result)
            del x
        controls = {r['case']: r for r in summary['user_requested_control_after_freeze']}
        control_errors = {}
        for r in rows(Path(args.diagnostic_root) / 'scores/vjepa/scores.jsonl'):
            assert sha(r['feature_path']) == r['feature_sha256']
            x = torch.from_numpy(np.load(r['feature_path'], allow_pickle=False)).to('cuda:0', torch.float32)[None]
            value = float(models['aligned'](x)[0]); error = abs(value - controls[r['family']]['latent'])
            assert error <= 1e-6; control_errors[r['family']] = error
    assert checked == 960 and len(feature_shas) == 480 and len(control_errors) == 5
    out = fresh_output(args.output)
    write_json(out / 'receipt.json', dict(status='passed', validation_latents_verified=checked,
        validation_unique_features_verified=len(feature_shas), max_abs_latent_error=maximum_error,
        user_control_errors=control_errors, checkpoint_sha256=summary['head_sha256'],
        summary_sha256=sha(root / 'training/summary.json'), verifier_sha256=sha(__file__),
        gpu=gpu, torch=torch.__version__, encoder_recomputed=False, training_updates=0, test450_opened=False,
        wall_seconds=time.monotonic() - start, completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print((out / 'receipt.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
