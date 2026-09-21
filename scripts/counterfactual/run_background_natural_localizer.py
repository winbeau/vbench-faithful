"""Evaluate the frozen caption-localizer candidate on every native dev video.

The earlier 680-video result is a checked reference, not a source of new masks.
No preference labels or construction artifacts are read while scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from .common import ROOT, sha256_file
from .run_background_holdout import frozen_protocol
from .score_background_refined import development_provider
from .subject_artifacts import new_output, read_jsonl, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, default=Path('/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    method_path, candidate_path = (ROOT/protocol[k] for k in ['method_protocol', 'candidate_protocol'])
    if (sha256_file(args.reference) != protocol['reference_sha256']
            or sha256_file(method_path) != protocol['method_protocol_sha256']
            or sha256_file(candidate_path) != protocol['candidate_protocol_sha256']):
        raise ValueError('development reference or method changed')
    method = frozen_protocol(method_path)
    manifest = ROOT/method['manifest']
    if sha256_file(manifest) != protocol['manifest_sha256']:
        raise ValueError('natural input manifest changed')
    entries = [r for r in read_jsonl(manifest) if r['split'] == 'dev']
    previous = read_jsonl(args.reference); reference = {r['video_uid']: r for r in previous}
    if (len(entries) != 680 or len(previous) != 680 or set(reference) != {r['video_uid'] for r in entries}
            or any(r['status'] != 'completed' for r in previous)):
        raise ValueError('complete matched development-only reference required')
    for entry in entries:
        if any(reference[entry['video_uid']][k] != value for k, value in entry.items()):
            raise ValueError('reference identity mismatch')

    import numpy as np
    import torch
    from PIL import Image, ImageSequence
    from background_consistency.calibration import patch_score_grid
    from background_consistency.inference import configure_inference
    from background_consistency.models import build_model
    from background_consistency.patches import background_patch_weights, patch_views, pool_background_tokens
    configure_inference(0); torch.set_num_threads(2)
    output = new_output(args.output); started = time.time()
    encoder = build_model(method['clip'], device=args.device)
    candidate = json.loads(candidate_path.read_text())
    if not candidate.get('caption_fallback') or candidate.get('automatic_fallback'):
        raise ValueError('this dev comparison requires the explicit caption fallback')
    provider = development_provider(method, candidate, device=args.device)
    run = {'protocol_sha256':sha256_file(args.protocol),'candidate_protocol_sha256':sha256_file(candidate_path),
        'reference_sha256':sha256_file(args.reference),'video_count':len(entries),'started_unix':started,
        'source_sha256':method['source_sha256'],'additional_source_sha256':candidate['source_sha256'],
        'runner_sha256':sha256_file(Path(__file__)),'localizer':provider.provenance,'encoder':encoder.provenance,
        'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':args.device,'torch_version':torch.__version__,
        'primary_method':method['primary_method'],'cohort':'all 680 native official background dev videos',
        'labels_read':False,'construction_artifacts_read':False}
    write_json(output/'run.json',run); write_json(output/'protocol.json',protocol)
    counts = Counter()
    for i, entry in enumerate(entries):
        uid = entry['video_uid']; old = reference[uid]
        row = {**entry,'scores':{}}
        try:
            path = args.video_root/entry['relative_video_path']
            if sha256_file(path) != old['video_sha256']:
                raise ValueError('reference original video bytes changed')
            row['video_sha256'] = old['video_sha256']
            frames = encoder.decode(path)
            frame_hash = hashlib.sha256(frames.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            if frame_hash != old['frame_array_sha256']:
                raise ValueError('reference decode pixels changed')
            row['scores'] = {'official':old['scores']['official'], 'frozen19':old['scores'][method['primary_method']]}
            masks = provider.masks_for(frames)
            diagnostics = provider.last_diagnostics
            cls, patches, views, fractions, valid = patch_views(encoder, frames, masks)
            grid = patch_score_grid(cls,views,valid)
            parity = abs(grid['official']-old['scores']['official']['score'])
            if parity > 1e-6 or old['parity_absolute_error'] > 1e-6:
                raise ValueError('new/reference/upstream origin parity failed')
            # A fallback frame had exactly zero base pixels. Reconstruct that
            # independent COCO80 result without a second model forward.
            base_masks = masks.copy()
            for trace in diagnostics['caption_fallback_frames']:
                base_masks[trace['frame']] = 0
            patch_grid = (int(encoder.model.visual.input_resolution)//int(encoder.model.visual.conv1.kernel_size[0]),)*2
            if np.prod(patch_grid) != patches.shape[1]:
                raise ValueError('unexpected native CLIP token grid')
            base_views, base_valid = {'global':views['global']}, {'global':valid['global']}
            for support in ['frame','union']:
                weights = background_patch_weights(encoder,base_masks,patch_grid,temporal_union=support=='union')
                base_views[support], _, base_valid[support] = pool_background_tokens(patches,weights)
            base_grid = patch_score_grid(cls,base_views,base_valid)
            mask_path = output/'scoring_masks'/f'{uid}.npz'
            write_npz(mask_path,masks=masks,coco80_masks=base_masks,
                metadata_json=json.dumps({'role':'scoring','construction_masks_reused':False,
                    'frame_array_sha256':frame_hash,'candidate_protocol_sha256':run['candidate_protocol_sha256']}))
            feature_path = output/'features'/f'{uid}.npz'
            write_npz(feature_path,cls=cls.cpu().numpy(),patches=patches.cpu().numpy(),
                      **{k:v.cpu().numpy() for k,v in views.items()})
            duration = None
            if path.suffix.lower() == '.gif':
                with Image.open(path) as image:
                    durations = [f.info.get('duration') for f in ImageSequence.Iterator(image)]
                if durations and all(x is not None for x in durations):
                    duration = sum(durations)/1000
            else:
                import decord
                reader = decord.VideoReader(str(path),num_threads=1)
                duration = len(reader)/reader.get_avg_fps()
            row['scores'].update({name:{'status':'succeeded','score':values[method['primary_method']]}
                                  for name,values in [('coco80',base_grid),('caption',grid)]})
            row.update(status='completed',score_grids={'caption':grid,'coco80':base_grid},num_frames=len(frames),
                frame_array_sha256=frame_hash,reference_origin_absolute_error=parity,
                upstream_score=old['upstream_score'],reference_upstream_absolute_error=old['parity_absolute_error'],
                localizer_diagnostics=diagnostics,media_duration_seconds=duration,
                scoring_mask={'path':str(mask_path.relative_to(output)),'sha256':sha256_file(mask_path)},
                features={'path':str(feature_path.relative_to(output)),'sha256':sha256_file(feature_path)})
        except Exception as exc:
            row.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
        counts[row['status']] += 1
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        progress = {'processed':i+1,'total':len(entries),'counts':dict(counts),'elapsed_seconds':time.time()-started}
        write_json(output/'progress.json',progress)
        if (i+1)%10 == 0 or row['status']=='failed':print(json.dumps(progress),flush=True)
    run.update(completed=True,finished_unix=time.time(),counts=dict(counts),scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json',run)
    return int(counts['failed']>0)


if __name__ == '__main__':
    raise SystemExit(main())
