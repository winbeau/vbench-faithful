"""Apply frozen background formulas to reviewed dev interventions.

This runner reads image lists and review decisions, never construction masks.
The default is the frozen 19-class method. An explicit development vocabulary
option expands scoring object classes and may enable an explicit automatic
MobileSAM fallback, never using construction masks or boxes.
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
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def development_provider(method, vocabulary, *, device):
    from vbench_audit_models.coco_vocabulary import build_coco80_provider
    if vocabulary.get('automatic_fallback') and vocabulary.get('caption_fallback'):
        raise ValueError('choose one explicit fallback, not an implicit mixture')
    for path, expected_hash in vocabulary['source_sha256'].items():
        if sha256_file(ROOT/path) != expected_hash:
            raise ValueError('vocabulary adapter changed after development freeze')
    for path, expected_hash in vocabulary.get('external_source_sha256', {}).items():
        if sha256_file(Path(path)) != expected_hash:
            raise ValueError('external scoring source changed after development freeze')
    provider = build_coco80_provider(method['localizer'], vocabulary, device=device)
    if vocabulary.get('automatic_fallback'):
        from vbench_audit_models.automatic_foreground import SamFallbackForegroundProvider
        provider = SamFallbackForegroundProvider(provider, vocabulary['automatic_fallback'])
        provider.provenance['automatic_source_sha256'] = vocabulary['external_source_sha256']
    if vocabulary.get('caption_fallback'):
        if vocabulary.get('caption_policy_revision', 1) == 2:
            from vbench_audit_models.caption_foreground_v2 import GritSamFallbackForegroundProviderV2
            provider = GritSamFallbackForegroundProviderV2(provider, vocabulary['caption_fallback'], device=device)
        elif vocabulary.get('caption_policy_revision', 1) == 1:
            from vbench_audit_models.caption_foreground import GritSamFallbackForegroundProvider
            provider = GritSamFallbackForegroundProvider(provider, vocabulary['caption_fallback'], device=device)
        else:
            raise ValueError('unknown caption policy revision')
    return provider


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method-protocol', type=Path, default=ROOT/'configs/background-repair/holdout_protocol_v1.json')
    p.add_argument('--video-root', type=Path, default=Path('/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--scoring-vocabulary', type=Path, help='Explicit development localizer extension; weights/formulas unchanged.')
    args = p.parse_args()
    method = frozen_protocol(args.method_protocol)
    construction = json.loads((args.dataset/'run.json').read_text())
    protocol = json.loads((args.dataset/'protocol.json').read_text())
    verification = json.loads(args.verification.read_text())
    review = json.loads(args.review.read_text())
    index_sha = sha256_file(args.dataset/'index.jsonl')
    if (not construction.get('completed') or construction['index_sha256'] != index_sha
            or verification['dataset_index_sha256'] != index_sha or review['dataset_index_sha256'] != index_sha
            or protocol['stage'] not in ('official_background_dev_segformer_grabcut_v2', 'official_background_dev_segformer_grabcut_v3')
            or protocol['split'] != 'dev' or review['metric_scores_used'] is not False):
        raise ValueError('incomplete, unreviewed, or non-development dataset')
    entries = read_jsonl(args.dataset/'index.jsonl')
    expected = {r['video_uid']: r for r in read_jsonl(ROOT/'configs/background-repair/natural1720_manifest_v2.jsonl') if r['split'] == 'dev'}
    if len(entries) != 680 or {r['video_uid'] for r in entries} != set(expected):
        raise ValueError('native background development population mismatch')
    verified = {r['video_uid']: r['status'] for r in verification['results']}
    decisions = {r['video_uid']: r for r in review['decisions']}
    accepted = {r['video_uid'] for r in entries if r['status'] == 'accepted'}
    if (set(decisions) != accepted or len(decisions) != len(review['decisions'])
            or len(verified) != len(entries)):
        raise ValueError('review or verification coverage incomplete')
    for uid in accepted:
        if verified[uid] != 'accepted_replay_verified' or decisions[uid]['decision'] not in ('usable', 'reject', 'pending'):
            raise ValueError('candidate lacks pixel verification or explicit semantic decision')

    import torch
    from background_consistency.calibration import patch_score_grid
    from background_consistency.inference import configure_inference
    from background_consistency.models import build_model
    from background_consistency.patches import patch_views
    from background_consistency.runtime import build_foreground_provider

    configure_inference(0); torch.set_num_threads(2)
    output = new_output(args.output)
    encoder = build_model(method['clip'], device=args.device)
    vocabulary = None
    if args.scoring_vocabulary:
        vocabulary = json.loads(args.scoring_vocabulary.read_text())
        if vocabulary['base_method_protocol_sha256'] != sha256_file(args.method_protocol):
            raise ValueError('vocabulary extension belongs to a different base method')
        provider = development_provider(method, vocabulary, device=args.device)
    else:
        provider = build_foreground_provider(method['localizer'], device=args.device)
    started = time.time()
    run = {'started_unix': started, 'dataset_index_sha256': index_sha,
        'method_protocol_sha256': sha256_file(args.method_protocol), 'review_sha256': sha256_file(args.review),
        'verification_sha256': sha256_file(args.verification), 'construction_protocol_sha256': construction['protocol_sha256'],
        'source_sha256': method['source_sha256'], 'runner_sha256': sha256_file(Path(__file__)),
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device,
        'encoder': encoder.provenance, 'localizer': provider.provenance,
        'primary_method': 'patch_frame_all_pairs_calibrated', 'video_count': len(entries),
        'cohort': 'official background development only; quality decisions precede scoring; pending visual reviews are explicitly unscored',
        'semantic_review_counts': dict(Counter(row['decision'] for row in decisions.values()))}
    run['scoring_candidate'] = vocabulary['name'] if vocabulary else 'frozen_coco19'
    if vocabulary:
        run['vocabulary_protocol_sha256'] = sha256_file(args.scoring_vocabulary)
        run['additional_source_sha256'] = vocabulary['source_sha256']
    write_json(output/'run.json', run)
    counts = Counter()
    for i, entry in enumerate(entries):
        uid = entry['video_uid']
        row = {**entry, 'construction_status': entry['status'], 'base': expected[uid], 'variants': {}}
        try:
            path = artifact_path(args.dataset, entry['manifest'])
            if sha256_file(path) != entry['manifest_sha256']:
                raise ValueError('construction manifest changed')
            source = json.loads(path.read_text())
            if source['base'] != expected[uid]:
                raise ValueError('construction source is not this background video')
            if entry['status'] != 'accepted':
                row['status'] = entry['status']
            elif decisions[uid]['decision'] == 'reject':
                row.update(status='semantic_rejected', semantic_review=decisions[uid])
            elif decisions[uid]['decision'] == 'pending':
                row.update(status='semantic_review_pending', semantic_review=decisions[uid])
            else:
                row['semantic_review'] = decisions[uid]
                for variant, files in source['variants'].items():
                    frames = torch.from_numpy(read_png_sequence(args.dataset, files)).permute(0, 3, 1, 2)
                    masks = provider.masks_for(frames)
                    cls, patches, views, fractions, valid = patch_views(encoder, frames, masks)
                    scores = patch_score_grid(cls, views, valid)
                    if set(scores) != set(method['methods']):
                        raise ValueError('frozen score grid changed')
                    identity = uid+'__'+variant.replace('/', '__')
                    frame_sha = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
                    mask_path = output/'scoring_masks'/f'{identity}.npz'
                    write_npz(mask_path, masks=masks, metadata_json=json.dumps({**provider.provenance,
                        'frame_array_sha256': frame_sha, 'input_manifest_sha256': entry['manifest_sha256']}))
                    feature_path = output/'features'/f'{identity}.npz'
                    write_npz(feature_path, cls=cls.cpu().numpy(), patches=patches.cpu().numpy(),
                        **{k: v.cpu().numpy() for k, v in views.items()},
                        **{k+'_background_fraction': v.cpu().numpy() for k, v in fractions.items()})
                    value = {'scores': scores, 'num_frames': len(frames), 'frame_array_sha256': frame_sha,
                        'scoring_mask': {'path': str(mask_path.relative_to(output)), 'sha256': sha256_file(mask_path)},
                        'features': {'path': str(feature_path.relative_to(output)), 'sha256': sha256_file(feature_path)},
                        'localizer_diagnostics': provider.last_diagnostics}
                    if variant == 'clean':
                        original = args.video_root/source['base']['relative_video_path']
                        if sha256_file(original) != source['video_sha256']:
                            raise ValueError('official source video changed')
                        value['upstream_score'] = encoder.upstream_score(original)
                        value['parity_absolute_error'] = abs(value['upstream_score']-scores['official'])
                        if value['parity_absolute_error'] > 1e-6:
                            raise ValueError('official source/PNG score parity failed')
                    row['variants'][variant] = value
                row['status'] = 'completed'
        except Exception as exc:
            row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
        counts[row['status']] += 1
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        progress = {'processed': i+1, 'total': len(entries), 'counts': dict(counts), 'elapsed_seconds': time.time()-started}
        write_json(output/'progress.json', progress)
        if row['status'] in ('completed', 'failed') or (i+1) % 50 == 0:
            print(json.dumps(progress), flush=True)
    run.update(completed=True, finished_unix=time.time(), counts=dict(counts), scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json', run)
    return int(counts['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
