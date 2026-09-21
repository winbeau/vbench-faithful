"""Inspect the existing GRiT object and caption heads on image-only inputs.

Records each head's actual ROI size before mapping object boxes to native RGB.
No construction target/mask or metric score is read, and no weight is downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import MethodType

import numpy as np
from PIL import Image

from .common import sha256_file
from .subject_artifacts import new_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('unexpected local GRiT checkpoint')
    inputs = json.loads(args.images.read_text())
    import torch
    from vbench_audit_models.grit import GritEvidenceModel
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    output = new_output(args.output); started = time.time()
    model = GritEvidenceModel('color', args.checkpoint, device=args.device)
    roi = model.model.demo.predictor.model.roi_heads
    original = roi._forward_box
    shapes = []

    def observe_size(_roi, *values, **kwargs):
        result = original(*values, **kwargs)
        shapes.append({'head': 'object' if kwargs.get('det_box', False) else 'primary',
                       'image_size': list(result[0].image_size)})
        return result

    roi._forward_box = MethodType(observe_size, roi)
    results = []
    for item in inputs:
        path = Path(item['path'])
        if sha256_file(path) != item['sha256']:
            raise ValueError('probe RGB image changed')
        image = np.asarray(Image.open(path).convert('RGB'))
        shapes.clear()
        result = model.detect(image)
        row = {**item, 'result': result, 'roi_sizes': list(shapes), 'native_objects': []}
        if result['status'] == 'succeeded':
            shape = [x for x in shapes if x['head'] == 'object']
            if len(shape) != 1:
                raise ValueError('missing or ambiguous object head coordinate system')
            h, w = shape[0]['image_size']
            scale = np.asarray([image.shape[1]/w, image.shape[0]/h]*2)
            for obj in result['objects']:
                box = np.asarray(obj['box'])*scale
                if (not np.isfinite(box).all() or (box < 0).any()
                        or box[2] > image.shape[1]+1e-4 or box[3] > image.shape[0]+1e-4):
                    raise ValueError('object box cannot be mapped to native pixels')
                row['native_objects'].append({**obj, 'box': box.tolist()})
        results.append(row)
        write_json(output/'progress.json', {'processed': len(results), 'total': len(inputs),
            'failures': sum(x['result']['status'] != 'succeeded' for x in results), 'elapsed_seconds': time.time()-started})
        print(json.dumps({'id': item['id'], 'status': result['status'],
                          'objects': [o['text'] for o in row['native_objects']]}), flush=True)
    write_json(output/'probe.json', {'images': results, 'input_manifest_sha256': sha256_file(args.images),
        'model': model.provenance, 'runner_sha256': sha256_file(Path(__file__)),
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'elapsed_seconds': time.time()-started,
        'strict_torch_determinism': False, 'cudnn_deterministic': True, 'seed': 0,
        'metric_scores_read': False, 'construction_masks_read': False, 'completed': True})


if __name__ == '__main__':
    main()
