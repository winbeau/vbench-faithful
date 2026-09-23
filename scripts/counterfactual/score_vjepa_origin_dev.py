"""Measure unmodified Origin on the fixed DEV210/60 original MP4s only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .static_jitter import ROOT, digest
from .validate_vjepa_motion import check_sha
from .vjepa_motion_probe import fresh_output, gpu_precheck, read_rows, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'probe-root', 'video-root', 'upstream', 'raft-weight', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--shard', type=int, required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-vjepa-origin-scale-model-training-v1' or args.shard not in range(4):
        raise ValueError('unexpected protocol/shard')
    import torch
    from dynamic_degree.backends.vbench import OfficialDynamicEvaluator, official_result_payload
    start = time.monotonic(); gpu = gpu_precheck(); torch.set_num_threads(3)
    probe = Path(args.probe_root)
    source_path = probe / 'selection/sources.jsonl'
    check_sha(source_path, config['source_manifest_sha256'])
    sources = [r for r in read_rows(source_path) if r['role'] in ('train', 'validation')]
    if len(sources) != 270:
        raise ValueError('only original DEV210/60 allowed')
    features = {}
    for path in sorted((probe / 'features').glob('shard-*/features.jsonl')):
        c = json.loads((path.parent / 'completion.json').read_text())
        check_sha(path, c['ledger_sha256'])
        for r in read_rows(path):
            if r['view'] == 'base':
                features[r['video_uid']] = r
    if set(features) != {r['video_uid'] for r in sources}:
        raise ValueError('original feature provenance coverage mismatch')
    check_sha(args.raft_weight, config['raft_sha256'])
    evaluator = OfficialDynamicEvaluator(torch.device('cuda:0'), Path(args.raft_weight), Path(args.upstream))
    if evaluator.upstream_state.sha != config['origin_upstream_commit'] or evaluator.upstream_state.dirty:
        raise ValueError('Origin source identity mismatch')
    selected = sources[args.shard::4]; out = fresh_output(args.output)
    files = [Path(__file__), ROOT / 'metrics/dynamic-degree/src/dynamic_degree/backends/vbench.py',
             ROOT / 'configs/upstream.toml', ROOT / 'packages/audit-models/src/vbench_audit_models/raft.py']
    write_json(out / 'provenance.json', {'config_sha256': digest(Path(args.config)),
        'source_manifest_sha256': digest(source_path), 'code': {str(f.relative_to(ROOT)): digest(f) for f in files},
        'upstream': asdict(evaluator.upstream_state), 'raft_sha256': config['raft_sha256'], 'gpu': gpu,
        'shard': args.shard, 'shards': 4, 'expected_uids': [r['video_uid'] for r in selected],
        'training_updates': 0, 'torch': torch.__version__, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    failed = 0
    with (out / 'scores.jsonl').open('x') as f:
        for i, r in enumerate(selected):
            record = dict(r)
            try:
                video = Path(args.video_root) / r['relative_video_path']
                check_sha(video, features[r['video_uid']]['video_sha256'])
                value = evaluator.evaluate_video(video)
                diagnostics = official_result_payload(value)
                sampling = diagnostics['sampling']
                if sampling['source_fps'] != 8 or sampling['sampled_source_frame_indices'] != list(range(16)):
                    raise ValueError('native sampling changed')
                if i == 0 and bool(evaluator.dynamic.infer(str(video))) != value.official_video_boolean:
                    raise ValueError('upstream infer parity mismatch')
                record.update(status='ok', score=float(value.official_video_boolean), diagnostics=diagnostics,
                              video_sha256=features[r['video_uid']]['video_sha256'], reference_infer_parity=i == 0)
            except Exception as exc:
                failed += 1; record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            f.write(json.dumps(record, allow_nan=False) + '\n'); f.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(selected):
                print(json.dumps({'shard': args.shard, 'completed': i + 1, 'expected': len(selected), 'failed': failed}), flush=True)
            if i == 0 and failed:
                raise RuntimeError(record['error'])
    write_json(out / 'completion.json', {'status': 'finished' if not failed else 'failed', 'expected': len(selected),
        'completed': len(selected), 'failed': failed, 'scores_sha256': digest(out / 'scores.jsonl'),
        'wall_seconds': time.monotonic() - start, 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'completion.json').read_text(), flush=True)
    if failed:
        raise RuntimeError('Origin teacher incomplete; do not train')


if __name__ == '__main__':
    main()
