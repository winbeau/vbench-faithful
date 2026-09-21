"""Evaluate v4 CLIP background patch pooling using verified independent masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .analyze_background_development import checked_rows
from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('natural', 'native', 'diagnostic'), required=True)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--video-root', type=Path, default=Path('/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid shard')
    protocol_path = ROOT/'configs/background-repair/development_patches_v4.json'
    protocol = json.loads(protocol_path.read_text())
    parent_path = protocol_path.parent/protocol['parent_protocol']
    if sha256_file(parent_path) != protocol['parent_protocol_sha256']:
        raise ValueError('parent protocol changed')
    parent = json.loads(parent_path.read_text())
    rows, roots, runs = [], {}, []
    directories = [args.previous_run] if (args.previous_run/'scores.jsonl').is_file() else sorted(args.previous_run.glob('shard*'))
    for directory in directories:
        chunk, run = checked_rows(directory)
        if run['protocol_sha256'] != protocol['parent_protocol_sha256']:
            raise ValueError('previous run uses a different method protocol')
        for name in ('algorithms.py', 'models.py', 'runtime.py'):
            key = 'metrics/background-consistency/src/background_consistency/'+name
            if run['source_sha256'][key] != sha256_file(ROOT/key):
                raise ValueError('baseline encoder/localizer implementation changed')
        runs.append({'directory': str(directory), 'run_sha256': sha256_file(directory/'run.json'), **run})
        for row in chunk:
            uid = row['video_uid']
            if uid in roots:
                raise ValueError('duplicate cached input')
            roots[uid] = directory
        rows.extend(chunk)
    expected_count = 288 if args.cohort == 'diagnostic' else 680
    if len(rows) != expected_count:
        raise ValueError('incomplete development cohort')
    if args.cohort == 'natural':
        expected = {r['video_uid']: r for r in read_jsonl(protocol_path.parent/parent['manifest_file']) if r['split'] == 'dev'}
        if set(roots) != set(expected):
            raise ValueError('unexpected or held-out natural input')
        if any(any(row[k] != value for k, value in expected[row['video_uid']].items()) for row in rows):
            raise ValueError('natural input identity mismatch')
    else:
        if args.dataset is None:
            parser.error('interventions require --dataset')
        entries = read_jsonl(args.dataset/'index.jsonl')
        if len(entries) != expected_count or {r['video_uid'] for r in entries} != set(roots):
            raise ValueError('intervention input identity mismatch')

    import numpy as np
    import torch
    from background_consistency.algorithms import temporal_score
    from background_consistency.candidates import all_pairs_same_precision
    from background_consistency.inference import configure_inference
    from background_consistency.models import build_model
    from background_consistency.patches import patch_views

    configure_inference(0); torch.set_num_threads(2)
    output = new_output(args.output)
    started = time.time()
    encoder = build_model(parent['clip'], device=args.device)
    selected = rows[args.shard_index::args.num_shards]
    run = {'protocol_sha256': sha256_file(protocol_path), 'cohort': args.cohort,
        'video_count': len(selected), 'started_unix': started,
        'num_shards': args.num_shards, 'shard_index': args.shard_index,
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
        'encoder': encoder.provenance, 'parent_runs': runs,
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p)
            for directory in (ROOT/'metrics/background-consistency/src', ROOT/'scripts/counterfactual') for p in directory.rglob('*.py')}}
    if args.dataset:
        run['dataset_index_sha256'] = sha256_file(args.dataset/'index.jsonl')
    write_json(output/'run.json', run); write_json(output/'protocol.json', protocol)

    def evaluate(frames, old, uid, variant, first_png_sha=None):
        ref = old['scoring_mask']
        path = Path(ref['path'])
        if not path.is_absolute():
            path = artifact_path(roots[uid], ref['path'])
        if sha256_file(path) != ref['sha256']:
            raise ValueError('independent scoring mask changed')
        frame_hash = hashlib.sha256(frames.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            masks = data['masks']
            meta = json.loads(str(data['metadata_json']))
        if meta.get('role') != 'scoring' or meta.get('construction_masks_reused') is not False:
            raise ValueError('mask was not independently predicted for scoring')
        if meta['weights_sha256'] != parent['localizer']['sam_sha256']:
            raise ValueError('cached scoring model identity changed')
        if 'frame_array_sha256' in meta:
            if meta['frame_array_sha256'] != frame_hash:
                raise ValueError('scoring mask belongs to different pixels')
        elif first_png_sha is None or meta.get('source_frame_sha256') != first_png_sha:
            raise ValueError('diagnostic scoring mask does not match the hashed variant')
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.shape != (len(frames), *frames.shape[-2:]):
            raise ValueError('scoring mask geometry mismatch')
        cls, patches, views, fractions, valid = patch_views(encoder, frames, masks)
        parity = abs(temporal_score(cls)-old['scores']['official']['score'])
        if parity > 1e-6:
            raise ValueError(f'patch extraction changed origin: {parity}')
        scores = {}
        for support, features in views.items():
            scores[f'patch_{support}_official'] = temporal_score(features, valid=valid[support].tolist())
            scores[f'patch_{support}_all_pairs'] = all_pairs_same_precision(features, valid=valid[support].tolist())
        identity = uid+'__'+variant.replace('/', '__')
        feature_path = output/'features'/f'{identity}.npz'
        write_npz(feature_path, cls=cls.cpu().numpy(), patches=patches.cpu().numpy(),
            **{k: v.cpu().numpy() for k, v in views.items()},
            **{k+'_background_fraction': v.cpu().numpy() for k, v in fractions.items()})
        return {'scores': {**old['scores'], **{k: {'status': 'succeeded', 'score': v} for k, v in scores.items()}},
            'features': {'path': str(feature_path.relative_to(output)), 'sha256': sha256_file(feature_path)},
            'scoring_mask': {'path': str(path), 'sha256': ref['sha256']},
            'frame_array_sha256': frame_hash, 'num_frames': len(frames),
            'origin_reencode_absolute_error': parity,
            'background_fraction': {k: v.cpu().tolist() for k, v in fractions.items()}}

    completed = failures = 0
    for i, old in enumerate(selected):
        uid = old['video_uid']
        row = {k: v for k, v in old.items() if k not in ('scores', 'variants', 'features', 'background_fraction', 'scoring_mask')}
        try:
            if args.cohort == 'natural':
                path = args.video_root/old['relative_video_path']
                if sha256_file(path) != old['video_sha256']:
                    raise ValueError('natural video content changed')
                row.update(evaluate(encoder.decode(path), old, uid, 'natural'))
                row['status'] = 'completed'
            else:
                row['variants'] = {}
                path = artifact_path(args.dataset, old['manifest'])
                if sha256_file(path) != old['manifest_sha256']:
                    raise ValueError('construction manifest changed')
                source = json.loads(path.read_text())
                if source['status'] == 'accepted':
                    if old['status'] != 'completed' or set(old['variants']) != set(source['variants']):
                        raise ValueError('incomplete cached intervention grid')
                    for variant, files in source['variants'].items():
                        frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0, 3, 1, 2)
                        row['variants'][variant] = evaluate(frames, old['variants'][variant], uid, variant, files[0]['sha256'])
                    row['status'] = 'completed'
                else:
                    row['status'] = old['status']
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
