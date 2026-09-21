"""Fixed CLS/missing-evidence candidate on the reused test and background sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .subject_artifacts import new_output, read_jsonl, read_png_sequence, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('natural-test', 'background'), required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--video-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dino-repo', required=True)
    parser.add_argument('--dino-weight', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.kind == 'background' and args.dataset is None:
        parser.error('background requires --dataset')
    if args.kind == 'natural-test' and args.video_root is None:
        parser.error('natural-test requires --video-root')
    protocol_path = ROOT/'configs/subject-repair/cls_candidate_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    natural = json.loads((ROOT/'configs/subject-repair/natural1440_protocol_v2.json').read_text())
    if sha256_file(args.dino_weight) != natural['dino_weight_sha256']:
        raise ValueError('DINO checkpoint changed')
    records, origins = [], {}
    for directory in sorted(args.source_run.glob('shard*')):
        if not directory.is_dir():
            continue
        previous = json.loads((directory/'run.json').read_text())
        if not previous.get('completed') or sha256_file(directory/'scores.jsonl') != previous['scores_sha256']:
            raise ValueError('source run incomplete or changed')
        for row in read_jsonl(directory/'scores.jsonl'):
            if args.kind == 'natural-test' and row['split'] != 'test':
                continue
            uid = row['video_uid'] if args.kind == 'natural-test' else row['base']['video_uid']
            if uid in origins:
                raise ValueError('duplicate source video')
            records.append(row); origins[uid] = directory
    expected = protocol['natural_test_videos'] if args.kind == 'natural-test' else protocol['background_candidates']
    if len(records) != expected:
        raise ValueError('incorrect followup cohort')
    if args.kind == 'background':
        index = {r['video_uid']: r for r in read_jsonl(args.dataset/'index.jsonl')}
        if set(index) != set(origins):
            raise ValueError('dataset cohort mismatch')
    output = new_output(args.output)
    write_json(output/'protocol.json', protocol)
    import numpy as np
    import torch
    from subject_consistency.isolation import isolate_subject_inputs
    from subject_consistency.metric import build_dino_config, upstream_path
    from subject_consistency.models import OfficialDinoFeatureExtractor, native_rgb_uint8
    from subject_consistency.subject_evidence import subject_consistency_from_vectors

    deterministic_setup(); torch.set_num_threads(2)
    started = time.time()
    extractor = OfficialDinoFeatureExtractor(args.device, build_dino_config(args.dino_repo, args.dino_weight), upstream_path())
    run = {'kind': args.kind, 'protocol_sha256': sha256_file(protocol_path), 'started_unix': started,
           'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'upstream': vars(extractor.upstream_state),
           'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                            [Path(__file__), *(ROOT/'metrics/subject-consistency/src').rglob('*.py')]}}
    write_json(output/'run.json', run)

    def candidate(value, frames, origin):
        mask_path = origin/value['scoring_mask_file']['path']
        if sha256_file(mask_path) != value['scoring_mask_file']['sha256']:
            raise ValueError('scoring mask changed')
        with np.load(mask_path, allow_pickle=False) as data:
            masks = torch.from_numpy(data['masks']).float(); present = torch.from_numpy(data['present']).bool()
        isolated = isolate_subject_inputs(frames, masks, present, view='full')
        count = int(isolated.present.any(1).sum())
        fallback = count < 2
        if fallback:
            conditional = None
            if value['scores']['official']['status'] != 'succeeded':
                raise ValueError('unavailable Official fallback')
            final = value['scores']['official']['score']
        else:
            features = extractor.features_from_frames(isolated.frames)
            conditional = subject_consistency_from_vectors(features, isolated.present.any(1), missing_policy='exclude').score
            final = conditional
        value['scores'][protocol['primary_method']] = {'status': 'succeeded', 'score': final,
            'diagnostics': {'num_present_frames': count, 'num_frames': len(frames), 'fallback_official': fallback,
                            'conditional_cls_score': conditional}}

    failures = 0
    for ordinal, row in enumerate(records, start=1):
        try:
            if args.kind == 'natural-test':
                uid = row['video_uid']; path = args.video_root/row['relative_video_path']
                if sha256_file(path) != row['source_video_sha256']:
                    raise ValueError('source video changed')
                frames = native_rgb_uint8(extractor.module.load_video(str(path)))
                if hashlib.sha256(frames.numpy().tobytes()).hexdigest() != row['decoded_rgb_sha256']:
                    raise ValueError('decode changed')
                candidate(row, frames, origins[uid])
            elif row['construction_status'] == 'accepted':
                uid = row['base']['video_uid']; entry = index[uid]
                path = args.dataset/entry['manifest']
                if sha256_file(path) != entry['manifest_sha256']:
                    raise ValueError('dataset manifest changed')
                manifest = json.loads(path.read_text())
                if manifest['base'] != row['base']:
                    raise ValueError('base identity changed')
                for name, value in row['variants'].items():
                    frames = torch.from_numpy(read_png_sequence(args.dataset, manifest['variants'][name])).permute(0,3,1,2)
                    candidate(value, frames, origins[uid])
            row['candidate_status'] = 'completed' if row['status'] == 'completed' else row['status']
        except Exception as exc:
            failures += 1; row.update(candidate_status='failed', candidate_failure_reason=f'{type(exc).__name__}: {exc}')
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        if ordinal % 20 == 0 or ordinal == len(records):
            progress = {'processed': ordinal, 'total': len(records), 'runtime_failures': failures, 'elapsed_s': time.time()-started}
            write_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
    run.update(completed=True, runtime_failures=failures, finished_unix=time.time(), scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
