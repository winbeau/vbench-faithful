"""Reuse subject SegFormer/GrabCut on the native background development set."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from .build_background_interventions import donor_map, foreground_and_background_edits
from .common import ROOT, sha256_file
from .generate_subject_masks import SegFormerConstructionLocalizer, deterministic_setup
from .region_discrimination import RejectedBase, refine_grabcut, window_indices
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json, write_npz, write_png_sequence


def choose_target(labels, id2label, names, *, rule='observed_median_mean_v3'):
    """Choose once from semantic evidence, not desired area or metric effects."""
    ids = {name: index for index, name in id2label.items()}
    if set(names) - set(ids):
        raise ValueError('foreground vocabulary not supported by this model')
    # Count all classes in one pass per frame; expanded vocabularies otherwise
    # repeat a full video-sized comparison for every eligible class.
    counts = np.stack([np.bincount(frame.reshape(-1), minlength=max(id2label)+1) for frame in labels])
    fractions_by_id = counts / (labels.shape[1]*labels.shape[2])
    areas = {name: fractions_by_id[:, ids[name]] for name in names}
    fractions = {name: values.tolist() for name, values in areas.items()}
    if rule == 'legacy_median_lexical_v2':
        # Exact replay of the archived v2 bug; never the default for new runs.
        target = min(names, key=lambda name: (-float(np.median(areas[name])), name))
    elif rule == 'observed_median_mean_v3':
        observed = [name for name in names if areas[name].any()]
        if not observed:
            return None, None, fractions
        target = min(observed, key=lambda name: (-float(np.median(areas[name])),
                                               -float(np.mean(areas[name])), name))
    else:
        raise ValueError('unknown target selection rule')
    return target, ids[target], fractions


def refine_selected(frames, labels, target, target_id):
    if target is None:
        return np.zeros(frames.shape[:3], np.uint8), []
    masks, failures = [], []
    for index, (frame, segmentation) in enumerate(zip(frames, labels)):
        try:
            mask = refine_grabcut(frame, (segmentation == target_id).astype(np.uint8), person=target == 'person')
        except RejectedBase as exc:
            mask = np.zeros(frame.shape[:2], np.uint8)
            failures.append({'frame': index, 'reason': exc.reason})
        masks.append(mask)
    return np.stack(masks), failures


def eligibility(semantic_areas, refined_areas, failures, protocol):
    reasons = []
    if (semantic_areas >= protocol['semantic_presence_min_area']).mean() < protocol['semantic_presence_fraction_min']:
        reasons.append('no_persistent_semantic_foreground')
    if failures:
        reasons.append('grabcut_frame_failure')
    if not protocol['mean_foreground_min'] <= refined_areas.mean() <= protocol['mean_foreground_max']:
        reasons.append('mean_foreground_outside_target_range')
    if refined_areas.min() < protocol['frame_foreground_min'] or refined_areas.max() > protocol['frame_foreground_max']:
        reasons.append('frame_foreground_outside_subject_protocol_range')
    return reasons


def partition_entries(entries, shard_index, shard_count):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('invalid construction shard')
    return entries[shard_index::shard_count]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/background-repair/construction_refined_dev_v3.json')
    p.add_argument('--semantic-cache', type=Path, help='Reuse verified all-150-class labels, then repeat target selection, GrabCut and all edits.')
    p.add_argument('--video-root', type=Path, default=Path('/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos'))
    p.add_argument('--segformer-dir', type=Path, default=Path('/root/wenbiao_zhao/models/subject-repair/segformer-b0-ade20k'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    manifest_path = ROOT/'configs/background-repair/natural1720_manifest_v2.jsonl'
    if sha256_file(manifest_path) != protocol['manifest_sha256'] or protocol['split'] != 'dev':
        raise ValueError('this development run requires the frozen native background input')
    entries = [r for r in read_jsonl(manifest_path) if r['split'] == 'dev']
    if len(entries) != protocol['candidate_count'] or any(r['dimension'] != 'background_consistency' for r in entries):
        raise ValueError('background development population mismatch')
    donors = donor_map(entries, strategy=protocol['donor_strategy'])
    deterministic_setup()
    import torch
    from vbench_audit_core.upstream import import_official_module
    torch.set_num_threads(2)
    module, state = import_official_module('background_consistency')
    assets = json.loads((ROOT/'configs/subject-repair/assets.lock.json').read_text())['segformer']['files']
    cache_entries, cache_run = {}, None
    if args.semantic_cache:
        cache_run = json.loads((args.semantic_cache/'run.json').read_text())
        if (not cache_run.get('completed') or cache_run['index_sha256'] != sha256_file(args.semantic_cache/'index.jsonl')
                or cache_run['input_manifest_sha256'] != protocol['manifest_sha256']
                or cache_run['source_sha256']['scripts/counterfactual/generate_subject_masks.py'] != sha256_file(ROOT/'scripts/counterfactual/generate_subject_masks.py')):
            raise ValueError('semantic cache input or inference provenance mismatch')
        cache_rows = read_jsonl(args.semantic_cache/'index.jsonl')
        cache_entries = {r['video_uid']: r for r in cache_rows}
        if len(cache_entries) != len(cache_rows) or set(cache_entries) != {r['video_uid'] for r in entries}:
            raise ValueError('semantic cache population mismatch')
        localizer_provenance = cache_run['localizer']
        id2label = {int(k): v for k, v in json.loads((ROOT/'configs/subject-repair/ade20k_labels.json').read_text())['id2label'].items()}
    else:
        localizer = SegFormerConstructionLocalizer(args.segformer_dir, device=args.device)
        localizer_provenance, id2label = localizer.provenance, localizer.id2label
    if localizer_provenance['files_sha256'] != assets:
        raise ValueError('construction weights differ from subject experiment')
    population_count = len(entries)
    entries = partition_entries(entries, args.shard_index, args.shard_count)
    output = new_output(args.output)
    started = time.time()
    sources = [Path(__file__), ROOT/'scripts/counterfactual/generate_subject_masks.py',
        ROOT/'scripts/counterfactual/region_discrimination.py', ROOT/'scripts/counterfactual/build_background_interventions.py',
        ROOT/'scripts/counterfactual/subject_artifacts.py']
    run = {'started_unix': started, 'total': len(entries), 'protocol_sha256': sha256_file(args.protocol),
        'input_manifest_sha256': sha256_file(manifest_path), 'localizer': localizer_provenance,
        'upstream': vars(state), 'device': args.device, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        'metric_scores_used': False, 'preference_labels_used': False, 'torch_version': torch.__version__}
    run['shard'] = {'index': args.shard_index, 'count': args.shard_count, 'population_count': population_count,
                    'assignment': 'manifest_index modulo shard_count; donor pairing uses complete population'}
    if cache_run:
        run['semantic_cache'] = {'path': str(args.semantic_cache), 'run_sha256': sha256_file(args.semantic_cache/'run.json'),
                                 'index_sha256': cache_run['index_sha256'], 'scope': 'all-150-class semantic labels only; target/refinement/eligibility/edits recomputed'}
    write_json(output/'run.json', run); write_json(output/'protocol.json', protocol)
    counts = Counter()

    def decode(path):
        return module.load_video(str(path)).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    for i, base in enumerate(entries):
        uid = base['video_uid']
        row = {'base': base, 'base_id': uid, 'status': 'construction_rejected', 'rejection_reasons': [], 'variants': {}}
        try:
            source = args.video_root/base['relative_video_path']
            row['video_sha256'] = sha256_file(source)
            frames = decode(source)
            if args.semantic_cache:
                cached_entry = cache_entries[uid]
                cached_path = artifact_path(args.semantic_cache, cached_entry['manifest'])
                if sha256_file(cached_path) != cached_entry['manifest_sha256']:
                    raise ValueError('cached semantic manifest changed')
                cached = json.loads(cached_path.read_text())
                if cached['base'] != base or cached['video_sha256'] != row['video_sha256']:
                    raise ValueError('cached semantics belong to a different source')
                cached_mask_path = artifact_path(args.semantic_cache, cached['construction_mask']['path'])
                if sha256_file(cached_mask_path) != cached['construction_mask']['sha256']:
                    raise ValueError('cached semantic labels changed')
                with np.load(cached_mask_path, allow_pickle=False) as saved:
                    labels = saved['semantic_labels']
                if labels.shape != frames.shape[:3] or not np.isin(labels, list(id2label)).all():
                    raise ValueError('cached semantic shape or class vocabulary mismatch')
                row['semantic_cache_mask_sha256'] = cached['construction_mask']['sha256']
            else:
                labels = np.stack([localizer.labels_for(frame) for frame in frames])
            rule = protocol.get('target_selection_rule', 'legacy_median_lexical_v2')
            target, target_id, areas = choose_target(labels, id2label, protocol['foreground_labels'], rule=rule)
            row.update(shape=list(frames.shape), construction_target=target,
                construction_target_source=protocol['target_source'], semantic_class_fractions=areas)
            masks, failures = refine_selected(frames, labels, target, target_id)
            mask_path = output/'construction_masks'/f'{uid}.npz'
            write_npz(mask_path, masks=masks, semantic_labels=labels,
                metadata_json=json.dumps({'role': 'construction', 'target': target,
                    'target_source': protocol['target_source'], 'source_video_sha256': row['video_sha256']}))
            area = masks.mean((1, 2))
            row.update(construction_mask={'path': str(mask_path.relative_to(output)), 'sha256': sha256_file(mask_path)},
                foreground_fraction=area.tolist(), refinement_failures=failures)
            semantic_area = np.array(areas[target]) if target is not None else np.zeros(len(frames))
            row['rejection_reasons'] = eligibility(semantic_area, area, failures, protocol)
            if target is None:
                row['rejection_reasons'].insert(0, 'no_observed_foreground_class')
            row['localization_state'] = ('no_observed_foreground_class' if target is None else
                'empty_after_refinement' if not masks.any() else 'partial_refinement_failure' if failures else 'mask_available')
            row['mask_area_interpretation'] = 'Predicted mask area, not ground-truth subject size; failed refinement frames use zero placeholders.'
            if len(frames) < 4:
                row['rejection_reasons'].append('too_few_frames')
            # All candidates, including rejections, retain semantic-mask evidence.
            if not row['rejection_reasons']:
                donor = donors[uid]
                donor_path = args.video_root/donor['relative_video_path']
                sigma = protocol['sigma_person'] if target == 'person' else protocol['sigma_other']
                row['gaussian_sigma'] = sigma
                fg, bg, donor_indices = foreground_and_background_edits(frames, masks, decode(donor_path), sigma=sigma)
                clean_files = write_png_sequence(output, f'clips/{uid}/clean', frames)
                fg_files = write_png_sequence(output, f'clips/{uid}/subject_blur', fg)
                bg_files = write_png_sequence(output, f'clips/{uid}/background_switch', bg)
                row['variants'] = {'clean': clean_files, 'full/subject_blur': fg_files}
                row['windows'] = {}
                for position in ('start', 'middle', 'end'):
                    indices = window_indices(len(frames), position)
                    row['windows'][position] = list(indices)
                    row['variants'][position+'/subject_blur'] = [fg_files[t] if t in indices else clean_files[t] for t in range(len(frames))]
                    row['variants'][position+'/background_switch'] = [bg_files[t] if t in indices else clean_files[t] for t in range(len(frames))]
                row.update(status='accepted', semantic_review_status='pending',
                    donor={**donor, 'video_sha256': sha256_file(donor_path)}, donor_frame_indices=donor_indices,
                    proofs={'subject_blur_mask_exterior_changed_pixels': 0, 'background_switch_mask_interior_changed_pixels': 0})
        except Exception as exc:
            row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
        counts[row['status']] += 1
        path = output/'manifests'/f'{uid}.json'
        write_json(path, row)
        index = {'video_uid': uid, 'base_id': uid, 'status': row['status'], 'manifest': str(path.relative_to(output)),
            'manifest_sha256': sha256_file(path), 'rejection_reasons': row['rejection_reasons']}
        with (output/'index.jsonl').open('a') as handle:
            handle.write(json.dumps(index, sort_keys=True)+'\n')
        progress = {'processed': i+1, 'total': len(entries), 'counts': dict(counts), 'elapsed_seconds': time.time()-started}
        write_json(output/'progress.json', progress)
        print(json.dumps(progress), flush=True)
    run.update(completed=True, finished_unix=time.time(), counts=dict(counts), index_sha256=sha256_file(output/'index.jsonl'))
    write_json(output/'run.json', run)
    return int(counts['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
