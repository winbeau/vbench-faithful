"""Run the finite v3 background localizer experiment, on development data only."""
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


def previous_rows(root):
    rows, runs = [], []
    directories = [root] if (root/'scores.jsonl').is_file() else sorted(root.glob('shard*'))
    for directory in directories:
        if not directory.is_dir():
            continue
        chunk, run = checked_rows(directory)
        rows.extend(chunk)
        runs.append({'directory': str(directory), 'run_sha256': sha256_file(directory/'run.json'), **run})
    return rows, runs


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
    config = ROOT/'configs/background-repair'
    protocol_path = config/'development_localizer_v3.json'
    protocol = json.loads(protocol_path.read_text())
    parent_path = config/protocol['parent_protocol']
    if sha256_file(parent_path) != protocol['parent_protocol_sha256']:
        raise ValueError('parent protocol changed')
    parent = json.loads(parent_path.read_text())
    rows, source_runs = previous_rows(args.previous_run)
    expected_count = 288 if args.cohort == 'diagnostic' else 680
    ids = [r.get('video_uid', r.get('base', {}).get('video_uid')) for r in rows]
    if len(rows) != expected_count or len(set(ids)) != expected_count or None in ids:
        raise ValueError('incomplete or duplicate development population')
    for run in source_runs:
        if run['protocol_sha256'] != protocol['parent_protocol_sha256']:
            raise ValueError('unexpected parent scoring protocol')
        for name in ('algorithms.py', 'models.py', 'runtime.py'):
            key = 'metrics/background-consistency/src/background_consistency/'+name
            if run['source_sha256'][key] != sha256_file(ROOT/key):
                raise ValueError('baseline representation changed')
    if args.cohort == 'natural':
        expected = {r['video_uid']: r for r in read_jsonl(config/parent['manifest_file']) if r['split'] == 'dev'}
        if set(ids) != set(expected):
            raise ValueError('unexpected or held-out natural identity')
        for row in rows:
            if any(row[k] != expected[row['video_uid']][k] for k in expected[row['video_uid']]):
                raise ValueError('natural input identity mismatch')
    else:
        if args.dataset is None:
            parser.error('interventions require --dataset')
        index = read_jsonl(args.dataset/'index.jsonl')
        if {r['video_uid'] for r in index} != set(ids) or len(index) != expected_count:
            raise ValueError('dataset population differs from previous scores')
        if args.cohort == 'native' and any(run['dataset_index_sha256'] != sha256_file(args.dataset/'index.jsonl') for run in source_runs):
            raise ValueError('native dataset changed')

    import numpy as np
    import torch
    from background_consistency.algorithms import temporal_score
    from background_consistency.candidates import all_pairs_same_precision, box_masks, candidate_scores, encode_candidate_views
    from background_consistency.inference import configure_inference
    from background_consistency.models import build_model
    from background_consistency.runtime import build_foreground_provider

    configure_inference(0)
    torch.set_num_threads(2)
    output = new_output(args.output)
    started = time.time()
    encoder = build_model(parent['clip'], device=args.device)
    provider = build_foreground_provider({**parent['localizer'], 'threshold': protocol['detector_threshold']}, device=args.device)
    selected = rows[args.shard_index::args.num_shards]
    run = {'protocol_sha256': sha256_file(protocol_path), 'cohort': args.cohort,
        'video_count': len(selected), 'num_shards': args.num_shards, 'shard_index': args.shard_index,
        'started_unix': started, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device': args.device, 'encoder': encoder.provenance, 'localizer': provider.provenance,
        'parent_runs': source_runs, 'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p)
            for directory in (ROOT/'metrics/background-consistency/src', ROOT/'packages/audit-models/src', ROOT/'scripts/counterfactual')
            for p in directory.rglob('*.py')}}
    if args.dataset:
        run['dataset_index_sha256'] = sha256_file(args.dataset/'index.jsonl')
    write_json(output/'run.json', run)
    write_json(output/'protocol.json', protocol)

    def evaluate(frames, old, identity, input_identity):
        # A fresh origin encode verifies that every candidate uses the exact
        # same real input and precision as the previously measured baseline.
        global_features = encoder.features(frames)
        parity = abs(temporal_score(global_features)-old['scores']['official']['score'])
        if parity > 1e-6:
            raise ValueError(f'v3 origin differs from v2: {parity}')
        masks = provider.masks_for(frames)
        boxes = box_masks(provider.last_diagnostics['detections'], masks.shape)
        views, fractions = encode_candidate_views(encoder, frames, {'sam03': masks, 'box03': boxes})
        scores = candidate_scores(views, fractions, min_background_fraction=protocol['min_background_fraction'])
        scores['aggregation_fp16'] = all_pairs_same_precision(global_features)
        path = output/'features'/f'{identity}.npz'
        write_npz(path, **{'global': global_features.cpu().numpy()},
            **{k: v.cpu().numpy() for k, v in views.items()},
            **{k+'_background_fraction': v for k, v in fractions.items()})
        mask_path = output/'scoring_masks'/f'{identity}.npz'
        write_npz(mask_path, sam03=masks, box03=boxes, metadata_json=json.dumps({**provider.provenance,
            'frame_array_sha256': hashlib.sha256(frames.to(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
            'input_identity': input_identity}, sort_keys=True))
        return {'scores': {**old['scores'], **{k: {'status': 'succeeded', 'score': v} for k, v in scores.items()}},
            'origin_reencode_absolute_error': parity, 'num_frames': len(frames),
            'features': {'path': str(path.relative_to(output)), 'sha256': sha256_file(path)},
            'scoring_mask': {'path': str(mask_path.relative_to(output)), 'sha256': sha256_file(mask_path)},
            'background_fraction': {k: v.tolist() for k, v in fractions.items()},
            'localizer_diagnostics': provider.last_diagnostics}

    failures = completed = 0
    for i, old in enumerate(selected):
        uid = old.get('video_uid', old.get('base', {}).get('video_uid'))
        row = {k: v for k, v in old.items() if k not in ('variants', 'scores', 'features', 'scoring_mask', 'background_fraction', 'localizer_diagnostics')}
        try:
            if args.cohort == 'natural':
                path = args.video_root/old['relative_video_path']
                if sha256_file(path) != old['video_sha256']:
                    raise ValueError('natural video changed')
                row.update(evaluate(encoder.decode(path), old, uid, old['video_sha256']))
                row['status'] = 'completed'
            else:
                row['variants'] = {}
                path = artifact_path(args.dataset, old['manifest'])
                if sha256_file(path) != old['manifest_sha256']:
                    raise ValueError('construction manifest changed')
                manifest = json.loads(path.read_text())
                if manifest['status'] == 'accepted':
                    if old['status'] != 'completed' or set(old['variants']) != set(manifest['variants']):
                        raise ValueError('incomplete previous intervention scores')
                    for variant, files in manifest['variants'].items():
                        frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0, 3, 1, 2)
                        identity = uid+'__'+variant.replace('/', '__')
                        row['variants'][variant] = evaluate(frames, old['variants'][variant], identity, old['manifest_sha256'])
                    row['status'] = 'completed'
                else:
                    row['status'] = old['status']
            if row['status'] == 'completed':
                completed += 1
        except Exception as exc:
            row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
            failures += 1
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        progress = {'processed': i+1, 'total': len(selected), 'completed': completed,
                    'failures': failures, 'elapsed_seconds': time.time()-started}
        write_json(output/'progress.json', progress)
        print(json.dumps(progress), flush=True)
    run.update(completed=True, finished_unix=time.time(), runtime_failures=failures,
               scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
