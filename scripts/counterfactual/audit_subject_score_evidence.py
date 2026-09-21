"""Audit actual scoring masks and whether edited frames contributed evidence.

This reads completed runs only. It neither changes scores nor removes candidates.
An unchanged score after excluding every edited frame is not supported stability.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .common import sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def checked_artifact(root, descriptor):
    path = artifact_path(root, descriptor['path'])
    if sha256_file(path) != descriptor['sha256']:
        raise ValueError('scoring artifact hash changed')
    return path


def encoder_presence(score, raw_present, source_coverage):
    """Validate the exact coverage used by the existing all-pairs reducer."""
    if score.get('status') != 'succeeded':
        return None
    d = score['diagnostics']
    coverage = np.asarray(d['coverage'], dtype=float)
    n = len(raw_present)
    if (coverage.shape != (n,) or not np.isfinite(coverage).all()
            or np.any((coverage < 0) | (coverage > 1)) or d['max_frames'] is not None):
        raise ValueError('invalid or subsampled encoder coverage')
    present = coverage > 0
    count = int(present.sum())
    if np.any(present & ~raw_present):
        raise ValueError('encoder evidence without an actual scoring mask')
    recorded_source = np.asarray(d['isolation']['source_coverage'], dtype=float)
    if (recorded_source.shape != (n,) or not np.isfinite(recorded_source).all()
            or not np.allclose(recorded_source, source_coverage, rtol=0, atol=1e-7)):
        raise ValueError('source coverage differs from actual scoring masks')
    if (d['missing_policy'] != 'exclude' or d['num_frames'] != n
            or d['num_present_frames'] != count or d['num_missing_frames'] != n-count
            or d['pair_count'] != count*(count-1)//2
            or d['pair_denominator'] != n*(n-1)//2 or count < 2):
        raise ValueError('score pair denominator differs from encoder evidence')
    number = score.get('score')
    if (number is None or not math.isfinite(number) or not -1e-6 <= number <= 1.000001
            or abs(number-d['score']) > 1e-7):
        raise ValueError('invalid succeeded score')
    return present


def variant_evidence(root, variant, method):
    descriptor = variant.get('scoring_masks', {}).get(method)
    if descriptor is None:
        return {'status': 'missing_scoring_artifact'}, None
    with np.load(checked_artifact(root, descriptor), allow_pickle=False) as data:
        masks, recorded = data['masks'], data['present']
        metadata = json.loads(data['metadata_json'].item())
    if (masks.ndim != 4 or masks.shape[0] != variant['num_frames']
            or not np.isin(masks, (0, 1)).all()):
        raise ValueError('invalid actual scoring mask array')
    actual = masks.reshape(*masks.shape[:2], -1).any(-1)
    if not np.array_equal(actual, recorded):
        raise ValueError('recorded mask presence differs from pixels')
    if (metadata['source'] != 'actual_variant_only' or metadata['method'] != method
            or metadata['decoded_rgb_sha256'] != variant['decoded_rgb_sha256']):
        raise ValueError('scoring masks do not identify the actual variant')
    union = masks.any(1)
    raw_present = actual.any(1)
    source_coverage = union.mean((1, 2))
    score = variant.get('scores', {}).get(method+'_exclude', {})
    present = encoder_presence(score, raw_present, source_coverage)
    result = {'status': score.get('status', 'missing_score'), 'frames': len(raw_present),
              'nonempty_mask_frames': int(raw_present.sum()),
              'encoder_evidence_frames': None if present is None else int(present.sum()),
              'nonempty_masks_lost_at_encoding': None if present is None else int((raw_present & ~present).sum()),
              'mask_sha256': descriptor['sha256']}
    return result, present


def condition_evidence(clean, background, subject, edited):
    """Do not treat unavailable or completely omitted intervention as observed."""
    indices = np.asarray(edited, dtype=int)
    lengths = {len(p) for p in (clean, background, subject) if p is not None}
    if (indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices)
            or len(lengths) > 1 or (lengths and (indices.min() < 0 or indices.max() >= next(iter(lengths))))):
        raise ValueError('invalid intervention frame indices')
    result = {'edited_frames': len(indices)}
    for name, present in [('clean', clean), ('background', background), ('subject', subject)]:
        result[name+'_edited_evidence_frames'] = None if present is None else int(present[indices].sum())
        result[name+'_all_frames_have_evidence'] = bool(present is not None and present.all())
    result['background_intervention_observed'] = bool(
        clean is not None and background is not None
        and clean[indices].any() and background[indices].any())
    result['subject_intervention_observed'] = bool(
        clean is not None and subject is not None
        and clean[indices].any() and subject[indices].any())
    return result


def audit(run_root, dataset, method='hybrid'):
    run = json.loads((run_root/'run.json').read_text())
    protocol = json.loads((run_root/'protocol.json').read_text())
    if (not run.get('completed') or sha256_file(run_root/'scores.jsonl') != run['scores_sha256']
            or sha256_file(run_root/'protocol.json') != run['protocol_sha256']
            or sha256_file(dataset/'index.jsonl') != run['dataset_index_sha256']
            or protocol['dataset_index_sha256'] != run['dataset_index_sha256']):
        raise ValueError('incomplete or changed run')
    rows = read_jsonl(run_root/'scores.jsonl')
    index_rows = read_jsonl(dataset/'index.jsonl')
    index = {r['video_uid']: r for r in index_rows}
    ids = [r['base']['video_uid'] for r in rows]
    if (len(rows) != protocol['cohort_candidates'] or len(set(ids)) != len(rows)
            or len(index) != len(index_rows) or set(ids) != set(index)):
        raise ValueError('candidate population changed')
    if method not in {'direct', protocol.get('candidate_name', 'tracked')}:
        raise ValueError('method is absent from the registered scoring policy')
    positions = protocol.get('positions', ['start', 'middle', 'end', 'full'])
    accepted = {r['base']['video_uid'] for r in rows if r['construction_status'] == 'accepted'}
    if (len(accepted) != protocol['expected_constructed']
            or not set(protocol.get('input_review_exclusions', {})) <= accepted):
        raise ValueError('constructed or primary population changed')
    cases = []
    source_root = Path(run.get('source_root', run_root))
    for row in rows:
        uid = row['base']['video_uid']
        if row['construction_status'] != index[uid]['status']:
            raise ValueError('construction qualification changed')
        if row['construction_status'] != 'accepted':
            continue
        descriptor = index[uid]
        manifest_path = artifact_path(dataset, descriptor['manifest'])
        if sha256_file(manifest_path) != descriptor['manifest_sha256']:
            raise ValueError('construction manifest changed')
        manifest = json.loads(manifest_path.read_text())
        root = artifact_path(source_root, row['source_shard']) if 'source_shard' in row else run_root
        case = {'video_uid': uid, 'review_eligible': uid not in protocol.get('input_review_exclusions', {}),
                'variants': {}, 'positions': {}}
        active = {}
        for name, variant in row['variants'].items():
            case['variants'][name], active[name] = variant_evidence(root, variant, method)
        for position in positions:
            indices = manifest['positions'][position]['parameters']['window_indices']
            case['positions'][position] = condition_evidence(active.get('clean'),
                active.get(position+'/background_corrupt'), active.get(position+'/subject_corrupt'), indices)
        cases.append(case)
    result = {'candidate_denominator': len(rows), 'constructed_denominator': len(cases),
              'candidate_status_counts': dict(Counter(r['status'] for r in rows)), 'method': method,
              'cases': cases, 'interpretation': 'Predicted evidence coverage is not semantic mask accuracy. '
              'No scores, candidate membership, or historic gates are changed. An intervention is observed '
              'only if at least one edited frame has encoder evidence in both clean and that actual variant.',
              'provenance': {'scores_sha256': run['scores_sha256'],
                  'run_sha256': sha256_file(run_root/'run.json'),
                  'protocol_sha256': run['protocol_sha256'],
                  'dataset_index_sha256': run['dataset_index_sha256'],
                  'audit_source_sha256': sha256_file(Path(__file__))}}
    for population in ['full_numeric', 'primary']:
        selected = [c for c in cases if population == 'full_numeric' or c['review_eligible']]
        result[population] = {'denominator': len(selected), 'positions': {}}
        for position in positions:
            conditions = [c['positions'][position] for c in selected]
            result[population]['positions'][position] = {
                key: sum(bool(c[key]) for c in conditions) for key in [
                    'background_intervention_observed', 'subject_intervention_observed',
                    'clean_all_frames_have_evidence', 'background_all_frames_have_evidence',
                    'subject_all_frames_have_evidence']}
            for name in ['clean', 'background', 'subject']:
                key = name+'_edited_evidence_frames'
                result[population]['positions'][position].update({
                    name+'_score_unavailable': sum(c[key] is None for c in conditions),
                    name+'_all_edited_frames_excluded': sum(c[key] == 0 for c in conditions)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'dataset', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--method', default='hybrid')
    args = parser.parse_args()
    result = audit(args.run, args.dataset, args.method)
    write_json(new_output(args.output)/'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
