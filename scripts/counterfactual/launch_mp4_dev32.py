"""Durable, four-isolated-GPU launcher for the predeclared DEV32 comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .static_jitter import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--legacy-boolean', action='store_true',
                        help='explicit historical reproduction only; default Repair is continuous intensity')
    parser.add_argument('--calibration-natural63', action='store_true',
                        help='score the separate prompt-disjoint natural calibration cohort, no interventions')
    args = parser.parse_args()
    if args.legacy_boolean and args.calibration_natural63:
        parser.error('natural scale calibration requires continuous intensity, never a binary Repair')
    task, output = Path(args.task_root), Path(args.output)
    root = Path(__file__).resolve().parents[2]
    construction = task / ('natural63-calibration-inputs-v1' if args.calibration_natural63
                           else 'local-texture-dev32-mp4-8px-v1')
    completion = json.loads((construction / 'completion.json').read_text())
    if completion['status'] != 'finished' or digest(construction / 'candidates.jsonl') != completion['manifest_sha256']:
        raise ValueError('completed, immutable construction required')
    devices = ['GPU-490b4a76-6210-31b9-4e03-838a113cf5f4', 'GPU-7d459912-b567-59fb-8abf-57c18713cb84',
               'GPU-8cf0627c-5919-bace-1c42-e5d627c36cb5', 'GPU-5e07f671-a7e1-b38d-4d14-30316dfd9ce8']
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'construction_manifest_sha256': completion['manifest_sha256'],
              'launcher_sha256': digest(Path(__file__)), 'children': []}
    start = time.monotonic()
    running = []
    for shard, device in enumerate(devices):
        command = [sys.executable, '-m', 'scripts.counterfactual.score_mp4_dev32',
          '--manifest', str(construction / 'candidates.jsonl'),
          '--sources', ('configs/dynamic-static-jitter/sources.natural63-calibration-v1.jsonl' if args.calibration_natural63
                        else 'configs/dynamic-static-jitter/sources.local-texture-dev32-mp4-v1.jsonl'),
          '--config', ('configs/dynamic-static-jitter/scoring.natural63-calibration-v1.json' if args.calibration_natural63
                       else 'configs/dynamic-static-jitter/scoring.dev32-cotracker3-boolean-v1.json' if args.legacy_boolean
                       else 'configs/dynamic-static-jitter/scoring.dev32-cotracker3-intensity-v1.json'),
          '--tracker-source-manifest', 'configs/dynamic-static-jitter/tracker.cotracker3-source-v1.json',
          '--tracker-root', '/data/chenjiayu/.cache/torch/hub/facebookresearch_co-tracker_main',
          '--tracker-weight', str(task / 'assets/cotracker3-scaled-offline-bf55ea50/scaled_offline.pth'),
          '--sam-root', str(task / 'deps/segment-anything'),
          '--sam-weight', '/data/chenjiayu/models/grounded_sam/sam_vit_h_4b8939.pth',
          '--sam-reference', str(task / 'code-sam-regions-v1/output/sam-regions-dev5-8px-v1/provenance.json'),
          '--sam-config', 'configs/dynamic-static-jitter/structure.sam-regions-dev-v1.json',
          '--raft-weight', '/data/chenjiayu/.cache/vbench/raft_model/models/raft-things.pth',
          '--upstream', '/data/chenjiayu/dynamic-static-jitter-20260922/VBench',
          '--output', str(output / f'shard-{shard}'), '--shard', str(shard), '--shards', '4']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   PYTHONPATH='metrics/dynamic-degree/src:packages/audit-models/src:packages/audit-core/src',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        log = (output / f'shard-{shard}.log').open('x')
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        record['children'].append({'shard': shard, 'pid': process.pid, 'device': device, 'command': command})
        running.append(process)
        (output / 'launch.json').write_text(json.dumps(record, indent=2))
    for child, process in zip(record['children'], running):
        child['returncode'] = process.wait()
        (output / 'launch.json').write_text(json.dumps(record, indent=2))
    record.update(status='finished', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  elapsed_seconds=time.monotonic() - start)
    (output / 'launch.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    return int(any(c['returncode'] for c in record['children']))


if __name__ == '__main__':
    raise SystemExit(main())
