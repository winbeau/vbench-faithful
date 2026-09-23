"""Isolated logged phases for the bounded BMC extension; no background retraining."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--phase', choices=('prepare', 'build', 'score-summarize'), required=True)
    p.add_argument('--gpu-uuids')
    p.add_argument('--worker', action='store_true')
    args = p.parse_args(); root = Path(args.root).resolve(); logdir = root / (args.phase + '-v5-controller')
    if not args.worker:
        logdir.mkdir(parents=True, exist_ok=False)
        with (logdir / 'controller.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (logdir / 'launch.json').write_text(json.dumps({'pid': proc.pid, 'args': vars(args)}))
        print(json.dumps({'pid': proc.pid, 'logs': str(logdir)})); return
    code = root / 'code'; env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VBENCH_AUDIT_WORKSPACE=str(code),
               PYTHONPATH='.:packages/audit-core/src:packages/audit-models/src:metrics/dynamic-degree/src')
    base = [sys.executable, '-u', '-m', 'scripts.counterfactual.evaluate_dynamic_bmc']
    common = ['--config', 'configs/dynamic-generalization/bmc-v1.json', '--selection', str(root / 'selection-v5')]
    def stage(name, commands):
        workers = []
        for i, (command, overrides) in enumerate(commands):
            log = (logdir / f'{name}-{i}.log').open('x')
            child = subprocess.Popen(command, cwd=code, env={**env, **overrides}, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            workers.append((child, log)); print(json.dumps({'stage': name, 'shard': i, 'pid': child.pid}), flush=True)
        codes = []
        for child, log in workers:
            codes.append(child.wait()); log.close()
        if any(codes):
            raise RuntimeError(f'{name} incomplete: exit codes {codes}; partial outputs preserved')
    started = time.monotonic()
    try:
        if args.phase == 'prepare':
            stage('prepare', [(base + ['prepare'] + common + ['--download', str(root / 'download'), '--output', str(root / 'selection-v5')], {})])
        elif args.phase == 'build':
            stage('freeze', [(base + ['freeze'] + common + ['--review', 'configs/dynamic-generalization/review.bmc-v1.json'], {})])
            stage('build', [(base + ['build'] + common + ['--shard', str(i), '--shards', '2',
                  '--output', str(root / 'construction' / f'shard-{i}')], {}) for i in range(2)])
        else:
            if not args.gpu_uuids:
                raise ValueError('explicit GPU UUIDs required')
            gpus = args.gpu_uuids.split(',')
            if len(set(gpus)) != len(gpus):
                raise ValueError('each shard must use a different GPU')
            assets = Path('/data/chenjiayu/dynamic-structural-motion-20260922')
            score_common = ['--construction', str(root / 'construction'), '--probe-root', str(assets / 'vjepa-probe-v1'),
                '--trained-root', str(assets / 'vjepa-aligned-v1'),
                '--official-video-root', '/data/chenjiayu/wenbiao_zhao/vbench-official-v1',
                '--raft-weight', '/data/chenjiayu/.cache/vbench/raft_model/models/raft-things.pth',
                '--upstream', '/data/chenjiayu/dynamic-static-jitter-20260922/VBench']
            stage('score', [(base + ['score'] + common + score_common + ['--shard', str(i), '--shards', str(len(gpus)),
                '--output', str(root / 'scores' / f'shard-{i}')], {'CUDA_VISIBLE_DEVICES': gpu}) for i, gpu in enumerate(gpus)])
            stage('summarize', [(base + ['summarize'] + common + ['--construction', str(root / 'construction'),
                '--scores', str(root / 'scores'), '--output', str(root / 'analysis')], {})])
        result = {'status': 'finished', 'phase': args.phase, 'wall_seconds': time.monotonic() - started}
    except Exception as exc:
        result = {'status': 'failed', 'phase': args.phase, 'wall_seconds': time.monotonic() - started,
                  'error': f'{type(exc).__name__}: {exc}'}
    (logdir / 'completion.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)
    if result['status'] != 'finished':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
