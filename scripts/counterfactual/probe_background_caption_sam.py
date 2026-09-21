"""Prompt MobileSAM with object-head-filtered native GRiT caption boxes."""
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
    parser.add_argument('--grit-probe', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method-protocol', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    source = json.loads(args.grit_probe.read_text())
    localizer = json.loads(args.method_protocol.read_text())['localizer']
    if (not source.get('completed') or source['construction_masks_read'] is not False
            or source['metric_scores_read'] is not False
            or source['model']['checkpoint_sha256'] != policy['checkpoint_sha256']):
        raise ValueError('input-only compatible GRiT probe required')
    checkpoint = Path(localizer['sam_checkpoint'])
    if sha256_file(checkpoint) != localizer['sam_sha256']:
        raise ValueError('MobileSAM weights changed')
    import torch
    from background_consistency.inference import configure_inference
    from vbench_audit_models.caption_foreground import select_caption_box
    from vbench_audit_models.foreground import load_mobile_sam_predictor
    configure_inference(0); torch.set_num_threads(2)
    predictor = load_mobile_sam_predictor(checkpoint, args.device)
    output = new_output(args.output); started = time.time(); results = []
    for record in source['images']:
        path = Path(record['path'])
        if sha256_file(path) != record['sha256'] or record['result']['status'] != 'succeeded':
            raise ValueError('GRiT input changed or failed')
        image = np.asarray(Image.open(path).convert('RGB'))
        box, trace = select_caption_box(record['result']['postprocessed'], image.shape[:2], policy)
        mask = np.zeros(image.shape[:2], np.uint8)
        with torch.inference_mode():
            if box is not None:
                predictor.set_image(np.ascontiguousarray(image), image_format='RGB')
                prediction, quality, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
                if prediction.shape != (1, *image.shape[:2]) or not np.isin(prediction, (0, 1)).all():
                    raise ValueError('caption-prompted SAM returned invalid mask')
                mask = prediction[0].astype(np.uint8)
                trace['sam_predicted_iou'] = float(np.asarray(quality).ravel()[0])
        identity = record['id']
        path = output/(identity+'.npz'); write_npz(path, selected=mask)
        overlay = image.copy(); used = mask.astype(bool)
        overlay[used] = (.5*overlay[used]+.5*np.array([240,40,60])).astype(np.uint8)
        Image.fromarray(overlay).save(output/(identity+'.png'))
        result = {k:record[k] for k in ['id','video_uid','variant','frame_index','path','sha256']}
        result.update(selection=trace, selected_fraction=float(mask.mean()),
                      mask={'path':path.name,'sha256':sha256_file(path)})
        results.append(result)
        print(json.dumps({'id':identity,'fraction':result['selected_fraction'],'selected':trace['selected_index']}),flush=True)
    write_json(output/'probe.json', {'completed':True,'images':results,'policy':policy,
        'policy_sha256':sha256_file(args.policy),'grit_probe_sha256':sha256_file(args.grit_probe),
        'sam_weights_sha256':localizer['sam_sha256'],
        'selector_source_sha256':sha256_file(ROOT/'packages/audit-models/src/vbench_audit_models/caption_foreground.py'),
        'runner_sha256':sha256_file(Path(__file__)),'elapsed_seconds':time.time()-started,
        'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'construction_masks_read':False,'metric_scores_read':False})


if __name__ == '__main__':
    main()
