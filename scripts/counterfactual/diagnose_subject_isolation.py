"""Representation diagnostics under a frozen, explicitly scoped protocol.

Independent MobileSAM masks are replayed from each exact variant's frozen
scoring run. Construction masks enter only a separate localization diagnostic
after scoring, never the representation or any sample-selection decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from .common import ROOT, sha256_file
from .build_region_discrimination import verify
from .generate_subject_masks import deterministic_setup
from .score_region_discrimination import ClipExtractor
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_jsonl
from .subject_region_statistics import paired_bootstrap

MODES = ("post_pool", "preencode_full", "preencode_crop")


def validate_protocol(dataset: Path, reference: Path, protocol: dict) -> dict:
    """Reject cohort/prompt drift before loading the representation model."""
    if tuple(protocol['representations']) != MODES or protocol['fixed_parameters'] != {
        'background_rgb': [128, 128, 128], 'mask_threshold': .5,
        'crop_margin': .1, 'crop_side': 224,
    }:
        raise ValueError('protocol does not match the implemented representation settings')
    index_hash = sha256_file(dataset / 'index.jsonl')
    reference_run = json.loads((reference / 'run.json').read_text())
    if reference_run['dataset_index_sha256'] != index_hash:
        raise ValueError('reference masks were scored on a different dataset')
    pinned = protocol.get('pinned_inputs', {})
    if pinned:
        if pinned['dataset_index_sha256'] != index_hash:
            raise ValueError('dataset differs from the preregistered cohort')
        if pinned['localizer_sha256'] != reference_run['localizer_sha256']:
            raise ValueError('reference used different human localizer prompts')
    index = read_jsonl(dataset / 'index.jsonl')
    if 'cohort' in protocol:
        actual = [{'base_id': row['base_id'], 'video_uid': row['video_uid'],
                   'source_prompt_id': row['source_prompt_id']} for row in index]
        if actual != protocol['cohort'] or len(actual) != protocol['sample_size']:
            raise ValueError('cohort differs from the preregistered identities')
        if {row['source_prompt_id'] for row in actual} & set(protocol['development_prompt_ids']):
            raise ValueError('new evaluation overlaps scored development prompts')
    return reference_run


def diagnose(dataset: Path, reference: Path, output: Path, extractor, device: str, *,
             stage: str = 'development_only') -> None:
    import torch
    from subject_consistency.metric import evaluate_masked_batch
    from subject_consistency.models import NpzSubjectMaskProvider

    rows, mask_diagnostics, groups = [], [], {}
    old = read_jsonl(reference / 'scores.jsonl')
    by = {(r['base_id'], r['position'], r['level'], r['backend']): r for r in old}
    provider = NpzSubjectMaskProvider(reference / 'scoring_masks')
    for index in read_jsonl(dataset / 'index.jsonl'):
        if index['status'] != 'accepted':
            continue
        manifest = json.loads(artifact_path(dataset, index['manifest']).read_text())
        base = manifest['base']; uid = base['video_uid']; bid = base['base_id']
        groups[bid] = base['prompt_id']
        for variant, files in manifest['variants'].items():
            position, level = ('shared', 'clean') if variant == 'clean' else variant.split('/')
            previous = by[(bid, 'full' if level == 'clean' else position, level, 'masked_zero')]
            if previous['status'] != 'succeeded':
                raise ValueError(f'pilot does not have complete independent masks: {bid} {variant}')
            name = f"{bid}__{variant.replace('/', '__')}"
            mask_path = artifact_path(reference, previous['scoring_mask_file']['path'])
            if sha256_file(mask_path) != previous['scoring_mask_file']['sha256']:
                raise ValueError('scoring-mask artifact hash mismatch')
            with np.load(mask_path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata_json'].item()))
                if metadata['role'] != 'scoring' or metadata['family'] != 'mobilesam' or metadata['construction_masks_reused']:
                    raise ValueError('only independent MobileSAM scoring masks are permitted')
            frames = torch.from_numpy(read_png_sequence(dataset, files)).permute(0, 3, 1, 2)
            for mode in MODES:
                adapter = ClipExtractor(extractor, frames)
                for policy in ('zero', 'exclude'):
                    result = evaluate_masked_batch([Path(name)], {name: {'subject_en': base['subject_en']}},
                        device, {}, provider, extractor=adapter, encoding_mode=mode, missing_policy=policy)[0]
                    row = {'base_id': bid, 'video_uid': uid, 'position': position, 'level': level,
                           'encoding_mode': mode, 'missing_policy': policy, 'score': result['score'],
                           'status': result['status'], 'failure_reason': result['failure_reason'],
                           'diagnostics': result['diagnostics'], 'scoring_mask_sha256': sha256_file(mask_path),
                           'source_prompt_id': base['prompt_id']}
                    if mode == 'post_pool' and result['status'] == 'succeeded':
                        expected = by[(bid, 'full' if level == 'clean' else position, level, 'masked_'+policy)]['score']
                        if abs(result['score'] - expected) > 1e-6:
                            raise ValueError('post-pool baseline does not replay the frozen pilot')
                    rows.append(row)
            write_jsonl(output / 'scores.jsonl', rows)
        # Diagnostic branch: the construction support cannot reach any score.
        source = artifact_path(dataset, manifest['construction_masks']['path'])
        if sha256_file(source) != manifest['construction_masks']['sha256']:
            raise ValueError('construction diagnostic mask hash mismatch')
        with np.load(source, allow_pickle=False) as data:
            construction = data['masks'].astype(bool)
        masks = {}
        for variant in ('clean', 'full/background_corrupt', 'full/subject_corrupt'):
            path = reference / 'scoring_masks' / f"{bid}__{variant.replace('/', '__')}.npz"
            with np.load(path, allow_pickle=False) as data:
                masks[variant] = data['masks'].astype(bool)
        clean, bg = masks['clean'], masks['full/background_corrupt']
        union = (clean | bg).sum((1, 2)); intersection = (clean & bg).sum((1, 2))
        leakage = (clean & ~construction).sum((1, 2)) / np.maximum(1, clean.sum((1, 2)))
        mask_diagnostics.append({'base_id': bid, 'background_mask_iou_per_frame':
                                np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union>0).tolist(),
                                'clean_scoring_mask_outside_construction_fraction': leakage.tolist(),
                                'interpretation': 'Diagnostic overlap only; neither model output is ground truth.'})
        write_jsonl(output / 'localization_diagnostics.jsonl', mask_diagnostics)
        print(f'{bid}: completed {len(rows)} diagnostic scores', flush=True)
    ids = sorted(groups)
    lookup = {(r['base_id'], r['position'], r['level'], r['encoding_mode'], r['missing_policy']):r for r in rows}
    report = {'stage': stage, 'n_bases': len(ids), 'n_prompt_clusters': len(set(groups.values())), 'positions': {}}
    for position in ('full', 'start', 'middle', 'end'):
        report['positions'][position] = {}
        for policy in ('zero', 'exclude'):
            absolute, subject, deltas = {}, {}, {}
            for mode in MODES:
                absolute[mode], subject[mode], deltas[mode] = {}, {}, {}
                for bid in ids:
                    selected = [lookup[(bid, 'shared' if level == 'clean' else position, level, mode, policy)]
                                for level in ('clean', 'background_corrupt', 'subject_corrupt')]
                    if any(r['status'] != 'succeeded' for r in selected):
                        raise ValueError('incomplete development ablation; inspect explicit failed records')
                    clean, bg, fg = (r['score'] for r in selected)
                    absolute[mode][bid] = abs(clean-bg)
                    subject[mode][bid] = clean-fg
                    deltas[mode][bid] = {'background_signed_drop': clean-bg, 'subject_signed_drop':clean-fg}
            improvement = {mode+'_reduction_vs_post_pool': {bid:absolute['post_pool'][bid]-absolute[mode][bid] for bid in ids}
                           for mode in MODES[1:]}
            report['positions'][position][policy] = {
                'background_absolute_change': paired_bootstrap({**absolute, **improvement}, groups, statistic_name='median'),
                'subject_signed_drop': paired_bootstrap(subject, groups, statistic_name='median'),
                'subject_positive_drop_count':{mode:sum(v>0 for v in column.values()) for mode,column in subject.items()},
                'per_base':deltas}
    write_json(output / 'statistics.json', report)
    print(json.dumps(report['positions']['full']['zero']['background_absolute_change'],indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dino-repo');parser.add_argument('--dino-weight')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--protocol', type=Path,
                        default=ROOT/'configs/subject-repair/isolation_development_protocol.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    reference_run = validate_protocol(args.dataset, args.reference, protocol)
    integrity = verify(args.dataset)
    from subject_consistency.metric import build_dino_config, upstream_path
    from subject_consistency.models import OfficialDinoPatchExtractor
    deterministic_setup()
    output = new_output(args.output)
    config = build_dino_config(args.dino_repo, args.dino_weight)
    if sha256_file(Path(config['path'])) != reference_run['dino_weight_sha256']:
        raise ValueError('DINO weights differ from the reference scorer')
    extractor = OfficialDinoPatchExtractor(args.device, config, upstream_path())
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'run.json', {'stage':protocol['stage'],'device':args.device,
        'python':platform.python_version(), 'dino_weight_sha256':sha256_file(Path(config['path'])),
        'upstream':vars(extractor.upstream_state),'dataset_index_sha256':sha256_file(args.dataset/'index.jsonl'),
        'reference_scores_sha256':sha256_file(args.reference/'scores.jsonl'),
        'protocol_sha256':sha256_file(args.protocol), 'integrity':integrity,
        'reference_run_sha256':sha256_file(args.reference/'run.json'),
        'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for directory in (
            ROOT/'metrics/subject-consistency/src',ROOT/'scripts/counterfactual') for p in sorted(directory.rglob('*.py'))}})
    diagnose(args.dataset, args.reference, output, extractor, args.device, stage=protocol['stage'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
