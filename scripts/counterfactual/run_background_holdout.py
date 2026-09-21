"""Score the frozen background method on held-out natural or intervention inputs.

The protocol pins method sources before this script can run. Human preference
labels are never opened here. Every actual video version gets fresh scoring
localization; construction masks are never read by the scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def frozen_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol.get('status') != 'frozen_before_test_scoring':
        raise ValueError('heldout method has not been frozen')
    for name, expected in protocol['source_sha256'].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError('frozen scoring source changed: '+name)
    for name, expected in protocol['configuration_sha256'].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError('frozen input/method configuration changed: '+name)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('natural', 'interventions'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--video-root', type=Path, default=Path('/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid shard')
    protocol = frozen_protocol(args.protocol)
    expected = {r['video_uid']: r for r in read_jsonl(ROOT/protocol['manifest']) if r['split'] == 'test'}
    if len(expected) != protocol['natural_video_count']:
        raise ValueError('heldout population mismatch')
    if args.mode == 'natural':
        entries = list(expected.values())
    else:
        if args.dataset is None or args.verification is None:
            parser.error('interventions require a complete dataset and pixel verification')
        construction = json.loads((args.dataset/'run.json').read_text())
        verification = json.loads(args.verification.read_text())
        index_hash = sha256_file(args.dataset/'index.jsonl')
        if (not construction.get('completed') or construction['index_sha256'] != index_hash
                or construction['protocol_sha256'] != protocol['construction_protocol_sha256']
                or verification['dataset_index_sha256'] != index_hash
                or verification['protocol_sha256'] != construction['protocol_sha256']):
            raise ValueError('unverified or mismatched heldout construction')
        entries = read_jsonl(args.dataset/'index.jsonl')
        if len(entries) != len(expected) or {r['video_uid'] for r in entries} != set(expected):
            raise ValueError('intervention population mismatch')
        verified = {r['video_uid']: r['status'] for r in verification['results']}
        if len(verified) != len(entries) or len(verification['results']) != len(entries):
            raise ValueError('pixel verification coverage mismatch')
        for entry in entries:
            allowed = {'accepted': 'accepted_replay_verified', 'construction_rejected': 'rejection_verified', 'failed': 'construction_failed'}
            if verified.get(entry['video_uid']) != allowed[entry['status']]:
                raise ValueError('candidate was not verified against its construction status')

    import numpy as np
    import torch
    from background_consistency.calibration import REPAIR_GAIN, patch_score_grid
    from background_consistency.inference import configure_inference
    from background_consistency.models import build_model
    from background_consistency.patches import patch_views
    from background_consistency.runtime import build_foreground_provider
    if REPAIR_GAIN != protocol['repair_gain']:
        raise ValueError('frozen repair gain mismatch')
    configure_inference(0); torch.set_num_threads(2)
    output = new_output(args.output)
    started = time.time()
    encoder = build_model(protocol['clip'], device=args.device)
    provider = build_foreground_provider(protocol['localizer'], device=args.device)
    selected = entries[args.shard_index::args.num_shards]
    run = {'mode': args.mode, 'protocol_sha256': sha256_file(args.protocol), 'video_count': len(selected),
        'started_unix': started, 'num_shards': args.num_shards, 'shard_index': args.shard_index,
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
        'torch_version': torch.__version__, 'encoder': encoder.provenance, 'localizer': provider.provenance,
        'source_sha256': protocol['source_sha256'], 'primary_method': protocol['primary_method']}
    if args.mode == 'interventions':
        run.update(dataset_index_sha256=index_hash, verification_sha256=sha256_file(args.verification))
    write_json(output/'run.json', run); write_json(output/'protocol.json', protocol)

    def evaluate(frames, uid, variant, input_sha):
        masks = provider.masks_for(frames)
        cls, patches, views, fractions, valid = patch_views(encoder, frames, masks)
        scores = patch_score_grid(cls, views, valid)
        if set(scores) != set(protocol['methods']):
            raise ValueError('frozen score grid changed')
        identity = uid+'__'+variant.replace('/', '__')
        mask_path = output/'scoring_masks'/f'{identity}.npz'
        frame_hash = hashlib.sha256(frames.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        write_npz(mask_path, masks=masks, metadata_json=json.dumps({**provider.provenance,
            'frame_array_sha256': frame_hash, 'input_sha256': input_sha, 'protocol_sha256': run['protocol_sha256']}, sort_keys=True))
        feature_path = output/'features'/f'{identity}.npz'
        write_npz(feature_path, cls=cls.cpu().numpy(), patches=patches.cpu().numpy(),
            **{k: v.cpu().numpy() for k, v in views.items()},
            **{k+'_background_fraction': v.cpu().numpy() for k, v in fractions.items()})
        return {'scores': {k: {'status': 'succeeded', 'score': v} for k, v in scores.items()},
            'num_frames': len(frames), 'frame_array_sha256': frame_hash,
            'scoring_mask': {'path': str(mask_path.relative_to(output)), 'sha256': sha256_file(mask_path)},
            'features': {'path': str(feature_path.relative_to(output)), 'sha256': sha256_file(feature_path)},
            'localizer_diagnostics': provider.last_diagnostics,
            'background_fraction': {k: v.cpu().tolist() for k, v in fractions.items()}}

    completed = failures = 0
    for i, entry in enumerate(selected):
        uid = entry['video_uid']; row = dict(entry)
        try:
            if args.mode == 'natural':
                path = args.video_root/entry['relative_video_path']
                row['video_sha256'] = sha256_file(path)
                frames = encoder.decode(path)
                row.update(evaluate(frames, uid, 'natural', row['video_sha256']))
                if path.suffix == '.gif':
                    from PIL import Image, ImageSequence
                    with Image.open(path) as image:
                        row['media_duration_seconds'] = sum(f.info.get('duration', 100) for f in ImageSequence.Iterator(image))/1000
                else:
                    import decord
                    reader = decord.VideoReader(str(path), num_threads=1)
                    row['media_duration_seconds'] = len(reader)/reader.get_avg_fps()
                upstream = encoder.upstream_score(path)
                row.update(upstream_score=upstream, parity_absolute_error=abs(upstream-row['scores']['official']['score']))
                if row['parity_absolute_error'] > 1e-6:
                    raise ValueError('heldout origin does not match pinned upstream')
                row['status'] = 'completed'
            else:
                path = artifact_path(args.dataset, entry['manifest'])
                if sha256_file(path) != entry['manifest_sha256']:
                    raise ValueError('heldout construction manifest changed')
                source = json.loads(path.read_text())
                if source['base'] != expected[uid] or source['status'] != entry['status']:
                    raise ValueError('heldout source identity mismatch')
                row.update(base=source['base'], construction_status=source['status'], variants={})
                if source['status'] == 'accepted':
                    for variant, files in source['variants'].items():
                        frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0, 3, 1, 2)
                        row['variants'][variant] = evaluate(frames, uid, variant, entry['manifest_sha256'])
                    row['status'] = 'completed'
                else:
                    row['status'] = 'construction_rejected' if source['status'] == 'construction_rejected' else 'construction_failed'
            completed += row['status'] == 'completed'
        except Exception as exc:
            row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
            failures += 1
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        progress = {'processed': i+1, 'total': len(selected), 'completed': completed,
                    'failures': failures, 'elapsed_seconds': time.time()-started}
        write_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
    run.update(completed=True, finished_unix=time.time(), runtime_failures=failures,
               scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
