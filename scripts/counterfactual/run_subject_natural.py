"""Full official-subject experiment with explicitly automatic scoring prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from .common import ROOT, probe_video, sha256_file
from .generate_subject_masks import deterministic_setup
from .score_region_discrimination import ClipExtractor
from .subject_artifacts import new_output, read_jsonl, write_json, write_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'protocol', 'video-root', 'output', 'detector-checkpoint', 'mobilesam-checkpoint'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dino-repo', required=True)
    parser.add_argument('--dino-weight', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true', help='Metadata-only three-class smoke; never a substitute for full coverage.')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid shard index')
    protocol = json.loads(args.protocol.read_text())
    if sha256_file(args.manifest) != protocol['manifest_sha256']:
        raise ValueError('manifest differs from preregistration')
    bases = read_jsonl(args.manifest)
    if len(bases) != 1440 or len({b['video_uid'] for b in bases}) != 1440:
        raise ValueError('expected all 1440 unique official videos')
    if args.smoke:
        bases = [next(b for b in bases if b['subject_en'] == name) for name in ('person', 'bird', 'car')]
    bases = bases[args.shard_index::args.num_shards]
    for base in bases:
        path = args.video_root / base['relative_video_path']
        if not path.is_file():
            raise ValueError(f'exact official input missing: {path}')
    import numpy as np
    import torch
    import torchvision
    from subject_consistency.automatic_localizer import CocoSubjectBoxDetector, AutomaticMobileSamSubjectMaskProvider
    from subject_consistency.localizer import load_mobile_sam_predictor
    from subject_consistency.backends.vbench import official_diagnostics
    from subject_consistency.metric import build_dino_config, evaluate_masked_batch, subject_consistency_diagnostics, upstream_path
    from subject_consistency.models import OfficialDinoPatchExtractor, native_rgb_uint8

    deterministic_setup()
    torch.set_num_threads(2)
    assets = json.loads((ROOT/'configs/subject-repair/assets.lock.json').read_text())
    if sha256_file(args.mobilesam_checkpoint) != assets['mobilesam']['checkpoint_sha256']:
        raise ValueError('unexpected MobileSAM checkpoint')
    output = new_output(args.output)
    started = time.time()
    detector = CocoSubjectBoxDetector(args.detector_checkpoint,
        expected_sha256=protocol['box_detector']['weight_sha256'], device=args.device,
        threshold=protocol['box_detector']['threshold'], size=protocol['box_detector']['size'])
    predictor = load_mobile_sam_predictor(args.mobilesam_checkpoint, args.device)
    provider = AutomaticMobileSamSubjectMaskProvider(detector, predictor,
        weights_sha256=assets['mobilesam']['checkpoint_sha256'])
    config = build_dino_config(args.dino_repo, args.dino_weight)
    if sha256_file(args.dino_weight) != protocol['dino_weight_sha256']:
        raise ValueError('unexpected DINO checkpoint')
    extractor = OfficialDinoPatchExtractor(args.device, config, upstream_path())
    write_json(output/'protocol.json', protocol)
    run = {'stage': protocol['stage'], 'mode': 'smoke' if args.smoke else 'full',
        'shard_index': args.shard_index, 'num_shards': args.num_shards, 'video_count': len(bases),
        'manifest_sha256': sha256_file(args.manifest), 'protocol_sha256': sha256_file(args.protocol),
        'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
        'gpu_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
        'dino_weight_sha256': sha256_file(args.dino_weight), 'localizer': provider.provenance,
        'upstream': vars(extractor.upstream_state), 'started_unix': started,
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for directory in
            (ROOT/'scripts/counterfactual', ROOT/'metrics/subject-consistency/src', ROOT/'packages/audit-core/src')
            for p in sorted(directory.rglob('*.py'))}}
    write_json(output/'run.json', run)
    failures = 0
    media_seconds = 0.
    for ordinal, base in enumerate(bases):
        before = time.time()
        video = args.video_root / base['relative_video_path']
        uid = base['video_uid']
        row = {**base, 'scores': {}, 'source_video_sha256': sha256_file(video)}
        try:
            frames = native_rgb_uint8(extractor.module.load_video(str(video)))
            meta = probe_video(video)
            media_seconds += meta.duration_s
            row.update(num_frames=len(frames), native_shape=list(frames.shape), media_duration_s=meta.duration_s,
                       decoded_rgb_sha256=hashlib.sha256(frames.numpy().tobytes()).hexdigest())
            features = extractor.features_from_frames(frames)
            write_npz(output/'global_features'/f'{uid}.npz', features=features.cpu().numpy())
            for name, fn in [('official', official_diagnostics), ('aggregation', subject_consistency_diagnostics)]:
                value = fn(features)
                row['scores'][name] = {'status': 'succeeded', 'score': value.final_score}
            masks = provider.masks_for(video, frames, base['subject_en'])
            row['localizer_diagnostics'] = provider.last_diagnostics
            mask_path = output/'scoring_masks'/f'{uid}.npz'
            write_npz(mask_path, masks=masks.instance_masks.numpy().astype('uint8'),
                      present=masks.instance_present.numpy(),
                      metadata_json=json.dumps({**provider.provenance, 'video_uid': uid,
                                               'source_video_sha256': row['source_video_sha256']}, sort_keys=True))
            row['scoring_mask_file'] = {'path': str(mask_path.relative_to(output)), 'sha256': sha256_file(mask_path)}
            class FixedMasks:
                def masks_for(self, *unused):
                    return masks
            for mode in protocol['representations']:
                adapter = ClipExtractor(extractor, frames)
                for policy in ('zero', 'exclude'):
                    value = evaluate_masked_batch([Path(uid)], {uid: {'subject_en': base['subject_en']}},
                        args.device, {}, FixedMasks(), extractor=adapter, encoding_mode=mode, missing_policy=policy)[0]
                    row['scores'][mode+'_'+policy] = {key: value[key] for key in ('status', 'score', 'failure_reason', 'diagnostics')}
            row['status'] = 'completed'
        except Exception as exc:
            row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
            failures += 1
        row['elapsed_s'] = time.time()-before
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        progress = {'completed_videos': ordinal+1, 'total_videos': len(bases), 'failures': failures,
                    'elapsed_s': time.time()-started, 'last_video_uid': uid, 'last_clip_s': row['elapsed_s']}
        write_json(output/'progress.json', progress)
        print(json.dumps(progress), flush=True)
    run.update(finished_unix=time.time(), wall_seconds=time.time()-started,
               media_duration_seconds=media_seconds, failed_videos=failures, completed=True,
               scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
