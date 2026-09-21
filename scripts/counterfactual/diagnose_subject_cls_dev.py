"""Development-only CLS-vs-patch readout ablation with unchanged coarse masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .subject_artifacts import new_output, read_jsonl, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-run', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dino-repo', required=True)
    parser.add_argument('--dino-weight', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    original, origins = {}, {}
    for directory in sorted(args.natural_run.glob('shard*')):
        if not directory.is_dir():
            continue
        run = json.loads((directory/'run.json').read_text())
        if not run.get('completed') or sha256_file(directory/'scores.jsonl') != run['scores_sha256']:
            raise ValueError('incomplete or changed natural run')
        for row in read_jsonl(directory/'scores.jsonl'):
            if row['split'] != 'dev':
                continue
            uid = row['video_uid']
            if uid in original:
                raise ValueError('duplicate video')
            original[uid] = row; origins[uid] = directory
    if len(original) != 580:
        raise ValueError('expected all 580 development videos; no test inference allowed')
    config = json.loads((ROOT/'configs/subject-repair/natural1440_protocol_v2.json').read_text())
    if sha256_file(args.dino_weight) != config['dino_weight_sha256']:
        raise ValueError('DINO weight changed')
    methods = ['official', 'aggregation', 'preencode_full_zero', 'preencode_full_exclude',
               'preencode_crop_zero', 'preencode_crop_exclude']
    methods += ['cls_'+view+'_'+policy+suffix for view in ('full', 'crop')
                for policy in ('zero', 'exclude') for suffix in ('', '_local_global')]
    output = new_output(args.output)
    protocol = {'stage': 'post_hoc_development_representation_ablation', 'n_videos': 580, 'split': 'dev only',
        'test_policy': 'Previously exposed test is not used for new scores, calibration, or selection.',
        'changed_component': 'DINO CLS readout instead of masked patch pooling; same isolated encoder pixels and masks.',
        'views': ['full', 'crop'], 'missing_policies': ['zero', 'exclude'],
        'aggregation': ['all_pairs', '0.5 * adjacent + 0.5 * all_pairs'],
        'methods': methods, 'tie_margin': 0., 'parameters_fitted': False,
        'fixed_representation_parameters': config['fixed_representation_parameters'],
        'natural_protocol_sha256': sha256_file(ROOT/'configs/subject-repair/natural1440_protocol_v2.json'),
        'dino_weight_sha256': sha256_file(args.dino_weight),
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
            [Path(__file__), *(ROOT/'metrics/subject-consistency/src').rglob('*.py')]}}
    write_json(output/'protocol.json', protocol)
    import numpy as np
    import torch
    from subject_consistency.isolation import isolate_subject_inputs
    from subject_consistency.metric import build_dino_config, upstream_path
    from subject_consistency.models import OfficialDinoFeatureExtractor, native_rgb_uint8
    from subject_consistency.subject_evidence import subject_consistency_from_vectors

    deterministic_setup(); torch.set_num_threads(2)
    started = time.time()
    extractor = OfficialDinoFeatureExtractor(args.device, build_dino_config(args.dino_repo,args.dino_weight), upstream_path())
    run = {'started_unix': started, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'upstream': vars(extractor.upstream_state), 'protocol_sha256': sha256_file(output/'protocol.json')}
    write_json(output/'run.json', run)
    failures = 0
    for ordinal, (uid, row) in enumerate(sorted(original.items()), start=1):
        try:
            path = args.video_root/row['relative_video_path']
            if sha256_file(path) != row['source_video_sha256']:
                raise ValueError('video changed')
            frames = native_rgb_uint8(extractor.module.load_video(str(path)))
            if hashlib.sha256(frames.numpy().tobytes()).hexdigest() != row['decoded_rgb_sha256']:
                raise ValueError('decode changed')
            mask_path = origins[uid]/row['scoring_mask_file']['path']
            if sha256_file(mask_path) != row['scoring_mask_file']['sha256']:
                raise ValueError('scoring masks changed')
            with np.load(mask_path, allow_pickle=False) as data:
                masks = torch.from_numpy(data['masks']).float()
                present = torch.from_numpy(data['present']).bool()
            for view in ('full', 'crop'):
                isolated = isolate_subject_inputs(frames, masks, present, view=view)
                features = extractor.features_from_frames(isolated.frames)
                write_npz(output/'features'/f'{uid}__{view}.npz', features=features.cpu().numpy(),
                          present=isolated.present.numpy())
                for policy in ('zero', 'exclude'):
                    name = 'cls_'+view+'_'+policy
                    try:
                        d = subject_consistency_from_vectors(features, isolated.present.any(1), missing_policy=policy)
                        row['scores'][name] = {'status': 'succeeded', 'score': d.score}
                        row['scores'][name+'_local_global'] = {'status': 'succeeded', 'score': .5*(d.score+d.adjacent_score)}
                    except ValueError as exc:
                        for key in (name, name+'_local_global'):
                            row['scores'][key] = {'status': 'failed', 'score': None, 'failure_reason': str(exc)}
            row['cls_run_status'] = 'completed'
        except Exception as exc:
            failures += 1
            row.update(cls_run_status='failed', cls_failure_reason=f'{type(exc).__name__}: {exc}')
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        if ordinal % 20 == 0:
            progress = {'completed_videos': ordinal, 'total': 580, 'runtime_failures': failures, 'elapsed_s': time.time()-started}
            write_json(output/'progress.json', progress); print(json.dumps(progress), flush=True)
    run.update(completed=True, runtime_failures=failures, finished_unix=time.time(),
               scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    # Preference labels need not be available on the remote inference snapshot.
    print(json.dumps({'completed': 580, 'runtime_failures': failures, 'elapsed_s': time.time()-started}), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
