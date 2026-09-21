"""Frozen developmental replay with clip-local tracking and direct-box ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types

from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .score_region_discrimination import ClipExtractor
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def upstream_score_from_frames(module, model, frames, device):
    """Execute pinned upstream scoring with the exact lossless decoded pixels.

    Clone function globals locally to replace only its file decoder. This does
    not edit the upstream checkout or shared module globals. Upstream receives
    float32, as its own load_video does, and runs its own transform and model.
    """
    namespace = dict(module.subject_consistency.__globals__)
    namespace['load_video'] = lambda path: frames.float()
    namespace['tqdm'] = lambda iterable, **kwargs: iterable
    reference = types.FunctionType(module.subject_consistency.__code__, namespace,
                                   module.subject_consistency.__name__,
                                   module.subject_consistency.__defaults__,
                                   module.subject_consistency.__closure__)
    _, rows = reference(model, ['actual_lossless_clip'], device, False)
    return float(rows[0]['video_results'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'output', 'protocol', 'detector-checkpoint', 'mobilesam-checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-run', type=Path, help='Historical development scores for exact replay; omit for frozen follow-up')
    parser.add_argument('--dino-repo', required=True)
    parser.add_argument('--dino-weight', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid shard')
    config = json.loads(args.protocol.read_text())
    for relative, expected_sha in config.get('method_source_sha256', {}).items():
        if sha256_file(artifact_path(ROOT, relative)) != expected_sha:
            raise ValueError('frozen repair method changed')
    candidate_name = config.get('candidate_name', 'tracked')
    if candidate_name not in ('tracked', 'anchor', 'hybrid'):
        raise ValueError('unknown candidate label')
    sources = {}
    for directory in sorted(args.source_run.glob('shard*')) if args.source_run else []:
        if not directory.is_dir():
            continue
        run = json.loads((directory/'run.json').read_text())
        if not run.get('completed') or sha256_file(directory/'scores.jsonl') != run['scores_sha256']:
            raise ValueError('source run incomplete or changed')
        for row in read_jsonl(directory/'scores.jsonl'):
            uid = row['base']['video_uid']
            if uid in sources:
                raise ValueError('duplicate source input')
            sources[uid] = row
    index = read_jsonl(args.dataset/'index.jsonl')
    integrity = None
    if args.source_run:
        expected = {uid:row['base'] for uid,row in sources.items()}
    else:
        manifest_path = artifact_path(ROOT, config['input_manifest'])
        if sha256_file(manifest_path) != config['input_manifest_sha256']:
            raise ValueError('frozen input manifest changed')
        bases = read_jsonl(manifest_path)
        expected = {row['video_uid']:row for row in bases}
        if len(expected) != len(bases) or sha256_file(args.dataset/'index.jsonl') != config['dataset_index_sha256']:
            raise ValueError('follow-up construction cohort changed')
        if config.get('defer_pixel_replay_to_analysis'):
            # Scoring reads hashed actual pixels; the independent CPU replay
            # may run concurrently. Final analysis must supply its receipt.
            integrity = {'status':'pending_analysis_gate','index_sha256':config['dataset_index_sha256']}
        elif config.get('integrity_receipt_sha256'):
            receipt = args.dataset/'integrity.json'
            if sha256_file(receipt) != config['integrity_receipt_sha256']:
                raise ValueError('pixel integrity receipt changed')
            integrity = json.loads(receipt.read_text())
            if (integrity['index_sha256'] != config['dataset_index_sha256']
                    or integrity['outside_mask_changed_pixels'] != 0 or integrity['bases'] != len(bases)):
                raise ValueError('pixel integrity receipt does not cover cohort')
        else:
            from .build_region_discrimination import verify
            integrity = verify(args.dataset)
    if (len(index) != config['cohort_candidates'] or len({r['video_uid'] for r in index}) != len(index)
            or {r['video_uid'] for r in index} != set(expected)):
        raise ValueError('source/dataset cohort mismatch')
    if sum(r['status'] == 'accepted' for r in index) != config['expected_constructed']:
        raise ValueError('construction cohort changed')
    assets = json.loads((ROOT/'configs/subject-repair/assets.lock.json').read_text())
    natural = json.loads((ROOT/'configs/subject-repair/natural1440_protocol_v2.json').read_text())
    if sha256_file(args.mobilesam_checkpoint) != assets['mobilesam']['checkpoint_sha256']:
        raise ValueError('MobileSAM checkpoint changed')
    if sha256_file(args.dino_weight) != natural['dino_weight_sha256']:
        raise ValueError('DINO checkpoint changed')
    import numpy as np
    import torch
    from subject_consistency.automatic_localizer import CocoSubjectBoxDetector
    from subject_consistency.localizer import load_mobile_sam_predictor
    from subject_consistency.temporal_localizer import TemporalMobileSamSubjectMaskProvider
    from subject_consistency.metric import build_dino_config, evaluate_masked_batch, upstream_path
    from subject_consistency.models import OfficialDinoPatchExtractor
    from subject_consistency.backends.vbench import official_diagnostics

    deterministic_setup(); torch.set_num_threads(2)
    output = new_output(args.output); started = time.time()
    write_json(output/'protocol.json', config)
    detector = CocoSubjectBoxDetector(args.detector_checkpoint, expected_sha256=config['detector']['weight_sha256'],
        device=args.device, threshold=config['detector']['threshold'], size=config['detector']['size'])
    provider = TemporalMobileSamSubjectMaskProvider(detector,
        load_mobile_sam_predictor(args.mobilesam_checkpoint, args.device),
        weights_sha256=assets['mobilesam']['checkpoint_sha256'],
        prompt_policy=config.get('prompt_policy', 'bounded_tracks'),
        **{k: config['tracking'][k] for k in ('minimum_iou','max_gap','smooth_radius')})
    extractor = OfficialDinoPatchExtractor(args.device,
        build_dino_config(args.dino_repo, str(args.dino_weight)), upstream_path())
    run = {'started_unix': started, 'protocol_sha256': sha256_file(args.protocol),
           'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
           'upstream': vars(extractor.upstream_state), 'localizer': provider.provenance,
           'integrity': integrity,
           'origin_reference': 'historical_exact_pixels' if args.source_run else 'pinned_upstream_function_actual_lossless_pixels',
           'num_shards': args.num_shards, 'shard_index': args.shard_index,
           'base_count': len(index[args.shard_index::args.num_shards]),
           'dataset_index_sha256': sha256_file(args.dataset/'index.jsonl'),
           'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                [Path(__file__), ROOT/'scripts/counterfactual/score_region_discrimination.py',
                 ROOT/'scripts/counterfactual/subject_artifacts.py',
                 ROOT/'packages/audit-models/src/vbench_audit_models/foreground.py',
                 *(ROOT/'metrics/subject-consistency/src').rglob('*.py')]}}
    write_json(output/'run.json', run)
    failures = completed = 0
    maximum_origin_error = 0.
    entries = index[args.shard_index::args.num_shards]
    for ordinal, entry in enumerate(entries, 1):
        manifest_path = artifact_path(args.dataset, entry['manifest'])
        if sha256_file(manifest_path) != entry['manifest_sha256']:
            raise ValueError('construction manifest changed')
        manifest = json.loads(manifest_path.read_text()); base = manifest['base']; uid = base['video_uid']
        if (base != expected[uid] or manifest['status'] != entry['status']
                or (args.source_run and manifest['status'] != sources[uid]['construction_status'])):
            raise ValueError('construction identity changed')
        row = {'base': base, 'construction_status': manifest['status'],
               'rejection_reasons': manifest['rejection_reasons'], 'status': 'construction_rejected', 'variants': {}}
        if manifest['status'] == 'accepted':
            before = time.time()
            try:
                for name, files in manifest['variants'].items():
                    frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0,3,1,2)
                    value = {'scores': {}, 'decoded_rgb_sha256': hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
                             'num_frames': len(frames)}
                    official = official_diagnostics(extractor.features_from_frames(frames)).final_score
                    reference = (sources[uid]['variants'][name]['scores']['official']['score'] if args.source_run
                                 else upstream_score_from_frames(extractor.module, extractor.model, frames, args.device))
                    error = abs(official - reference); maximum_origin_error = max(maximum_origin_error, error)
                    if error > 1e-6:
                        raise ValueError(f'Origin parity failed: {error}')
                    value['scores']['official'] = {'status': 'succeeded', 'score': official}
                    value['origin_reference_error'] = error
                    masks = provider.masks_for(Path(uid), frames, base['subject_en'])
                    value['localizer_diagnostics'] = provider.last_diagnostics
                    value['scoring_masks'] = {}
                    for method, current in [('direct', provider.last_direct_masks), (candidate_name, masks)]:
                        path = output/'scoring_masks'/f"{uid}__{name.replace('/', '__')}__{method}.npz"
                        write_npz(path, masks=current.instance_masks.numpy().astype(np.uint8),
                                  present=current.instance_present.numpy(),
                                  metadata_json=json.dumps({'source': 'actual_variant_only', 'method': method,
                                      'decoded_rgb_sha256': value['decoded_rgb_sha256']}, sort_keys=True))
                        value['scoring_masks'][method] = {'path': str(path.relative_to(output)), 'sha256': sha256_file(path)}
                        class FixedMasks:
                            def masks_for(self, *unused): return current
                        adapter = ClipExtractor(extractor, frames)
                        for policy in config['reported_missing_policies']:
                            result = evaluate_masked_batch([Path(uid)], {uid: {'subject_en': base['subject_en']}},
                                args.device, {}, FixedMasks(), extractor=adapter,
                                encoding_mode='preencode_crop', missing_policy=policy)[0]
                            value['scores'][method+'_'+policy] = {
                                k: result[k] for k in ('status', 'score', 'failure_reason', 'diagnostics')}
                    row['variants'][name] = value
                row.update(status='completed', scoring_seconds=time.time()-before); completed += 1
            except Exception as exc:
                failures += 1; row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
            print(json.dumps({'constructed_completed': completed, 'uid': uid, 'status': row['status'],
                              'runtime_failures': failures, 'elapsed_s': time.time()-started}), flush=True)
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        write_json(output/'progress.json', {'processed': ordinal, 'total': len(entries),
                   'constructed_completed': completed, 'runtime_failures': failures, 'elapsed_s': time.time()-started})
    run.update(completed=True, completed_bases=completed, runtime_failures=failures,
               maximum_origin_reference_error=maximum_origin_error, finished_unix=time.time(),
               wall_seconds=time.time()-started, scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
