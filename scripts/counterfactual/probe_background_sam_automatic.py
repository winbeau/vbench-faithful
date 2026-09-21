"""Probe existing MobileSAM automatic proposals on an explicit image-only list.

This input-quality diagnostic reads neither construction masks nor metric scores.
Every proposal and its selection trace is retained, including empty outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .common import ROOT, sha256_file
from .subject_artifacts import new_output, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--method-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--semantic-policy', type=Path, help='Optional explicit input-only CLIP object/scene proposal gate.')
    args = parser.parse_args()
    inputs = json.loads(args.images.read_text())
    policy = json.loads(args.policy.read_text())
    localizer = json.loads(args.method_protocol.read_text())['localizer']
    assets = json.loads((ROOT/'configs/subject-repair/assets.lock.json').read_text())
    checkpoint = Path(localizer['sam_checkpoint'])
    if sha256_file(checkpoint) != localizer['sam_sha256']:
        raise ValueError('MobileSAM weights changed')

    import mobile_sam
    import torch
    from background_consistency.inference import configure_inference
    from vbench_audit_models.automatic_foreground import build_automatic_generator, select_automatic_mask
    from vbench_audit_models.foreground import load_mobile_sam_predictor

    configure_inference(0); torch.set_num_threads(2)
    source = Path(mobile_sam.__file__).resolve().parent.parent
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != assets['mobilesam']['revision']:
        raise ValueError('unexpected MobileSAM source revision')
    output = new_output(args.output)
    started = time.time()
    generator = build_automatic_generator(load_mobile_sam_predictor(checkpoint, args.device).model, policy)
    classifier = None
    semantic = None
    if args.semantic_policy:
        from background_consistency.models import build_model
        from vbench_audit_models.semantic_foreground import ClipProposalClassifier, select_semantic_mask
        semantic = json.loads(args.semantic_policy.read_text())
        for path, expected in semantic['source_sha256'].items():
            if sha256_file(ROOT/path) != expected:
                raise ValueError('semantic probe source changed')
        encoder = build_model(json.loads(args.method_protocol.read_text())['clip'], device=args.device)
        classifier = ClipProposalClassifier(encoder.model, encoder.preprocess, encoder.module.clip.tokenize, semantic, device=args.device)
    records = []
    for item in inputs:
        path = Path(item['path'])
        if sha256_file(path) != item['sha256']:
            raise ValueError('input image changed')
        frame = np.asarray(Image.open(path).convert('RGB'))
        with torch.inference_mode():
            proposals = generator.generate(np.ascontiguousarray(frame))
        if classifier:
            classifications = classifier.classify(frame, proposals)
            selected, trace = select_semantic_mask(proposals, classifications, frame.shape[:2], policy, semantic)
        else:
            selected, trace = select_automatic_mask(proposals, frame.shape[:2], policy)
        identity = item['id']
        if Path(identity).name != identity:
            raise ValueError('unsafe image identifier')
        target = output/(identity+'.npz')
        proposal_masks = (np.stack([row['segmentation'] for row in proposals]).astype(np.uint8)
                          if proposals else np.empty((0, *frame.shape[:2]), np.uint8))
        write_npz(target, proposals=proposal_masks, selected=selected)
        overlay = frame.copy()
        overlay[selected.astype(bool)] = (.5*overlay[selected.astype(bool)] + .5*np.array([240, 40, 60])).astype(np.uint8)
        Image.fromarray(overlay).save(output/(identity+'.png'))
        row = {**item, 'shape': list(frame.shape), 'proposal_count': len(proposals),
               'selected_fraction': float(selected.mean()), 'selection': trace,
               'masks': {'path': target.name, 'sha256': sha256_file(target)}}
        records.append(row)
        write_json(output/'progress.json', {'completed': len(records), 'total': len(inputs),
                                         'elapsed_seconds': time.time()-started})
        print(json.dumps({'id': identity, 'proposal_count': len(proposals),
                          'selected_fraction': row['selected_fraction']}), flush=True)
    write_json(output/'probe.json', {
        'policy': policy, 'policy_sha256': sha256_file(args.policy),
        'semantic_policy': semantic,
        'semantic_policy_sha256': sha256_file(args.semantic_policy) if semantic else None,
        'semantic_encoder': encoder.provenance if semantic else None,
        'input_manifest_sha256': sha256_file(args.images), 'images': records,
        'weights_sha256': localizer['sam_sha256'], 'mobilesam_revision': revision,
        'mobilesam_source_sha256': {str(path.relative_to(source)): sha256_file(path)
                                  for path in sorted((source/'mobile_sam').rglob('*.py'))},
        'adapter_sha256': sha256_file(ROOT/'packages/audit-models/src/vbench_audit_models/automatic_foreground.py'),
        'runner_sha256': sha256_file(Path(__file__)), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'elapsed_seconds': time.time()-started, 'completed': True,
        'construction_masks_read': False, 'metric_scores_read': False,
        'interpretation': 'Input-only development probe; selection quality requires visual review, not area alone.'})


if __name__ == '__main__':
    main()
