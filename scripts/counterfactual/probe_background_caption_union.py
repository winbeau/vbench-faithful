"""Probe all caption-prompted object masks on hash-verified actual inputs.

Reuses raw independent GRiT descriptions, not the construction annotations or
the old selected box. Does not compute metric scores or update any default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .common import ROOT, sha256_file
from .subject_artifacts import new_output, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grit-probe', type=Path, nargs='+', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    protocol = json.loads(args.policy.read_text())
    for filename, expected in protocol['source_sha256'].items():
        if sha256_file(ROOT / filename) != expected:
            raise ValueError('union probe source changed after freeze')
    policy_path = ROOT / protocol['base_policy']
    if sha256_file(policy_path) != protocol['base_policy_sha256']:
        raise ValueError('fixed caption policy changed')
    policy = json.loads(policy_path.read_text())['caption_fallback']
    localizer_path = ROOT / 'configs/background-repair/holdout_protocol_v1.json'
    if sha256_file(localizer_path) != protocol['method_protocol_sha256']:
        raise ValueError('base model protocol changed')
    localizer = json.loads(localizer_path.read_text())['localizer']
    images = []
    seen = set()
    for filename in args.grit_probe:
        source = json.loads(filename.read_text())
        if (not source.get('completed') or source['construction_masks_read'] is not False
                or source['metric_scores_read'] is not False
                or source['model']['checkpoint_sha256'] != policy['checkpoint_sha256']):
            raise ValueError('compatible independent GRiT output required')
        for record in source['images']:
            if record['id'] in seen:
                raise ValueError('duplicate probe image')
            seen.add(record['id'])
            images.append(record)
    checkpoint = Path(localizer['sam_checkpoint'])
    if sha256_file(checkpoint) != localizer['sam_sha256']:
        raise ValueError('MobileSAM weights changed')
    import torch
    from background_consistency.inference import configure_inference
    from vbench_audit_models.caption_union import caption_sam_union
    from vbench_audit_models.foreground import load_mobile_sam_predictor
    configure_inference(0)
    torch.set_num_threads(2)
    predictor = load_mobile_sam_predictor(checkpoint, args.device)
    output = new_output(args.output)
    started = time.time()
    rows = []
    with torch.inference_mode():
        for record in images:
            filename = Path(record['path'])
            if sha256_file(filename) != record['sha256'] or record['result']['status'] != 'succeeded':
                raise ValueError('actual GRiT input pixels changed or failed')
            frame = np.asarray(Image.open(filename).convert('RGB'))
            mask, trace = caption_sam_union(predictor, frame, record['result']['postprocessed'], policy)
            mask_path = output / (record['id'] + '.npz')
            write_npz(mask_path, selected=mask)
            result = {key: record[key] for key in ('id', 'video_uid', 'variant', 'frame_index', 'path', 'sha256')}
            result.update(selection=trace, mask={'path': mask_path.name, 'sha256': sha256_file(mask_path)})
            rows.append(result)
            print(json.dumps({'id': record['id'], 'objects': len(trace['selected_indices']),
                              'fraction': trace['foreground_fraction']}), flush=True)
    write_json(output / 'probe.json', {'completed': True, 'images': rows,
        'policy_sha256': sha256_file(args.policy), 'protocol': protocol,
        'grit_probe_sha256': {str(p): sha256_file(p) for p in args.grit_probe},
        'sam_weights_sha256': localizer['sam_sha256'], 'runner_sha256': sha256_file(Path(__file__)),
        'elapsed_seconds': time.time() - started, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'construction_masks_read': False, 'metric_scores_read': False,
        'scope': 'Caption-only union probe on every supplied actual frame, regardless of any prior COCO detection. No new score run or default promotion.'})


if __name__ == '__main__':
    main()
