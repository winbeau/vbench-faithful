"""Audit complete construction coverage and optional fresh/cache parity.

This reads construction artifacts only. Numerical frame coverage is distinct
from semantic mask quality and is never used to change the scoring cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .common import sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def completed_entries(root):
    run = json.loads((root/'run.json').read_text())
    if not run.get('completed') or run['index_sha256'] != sha256_file(root/'index.jsonl'):
        raise ValueError('construction is incomplete or index changed')
    index = read_jsonl(root/'index.jsonl'); entries = {}
    for row in index:
        uid = row['video_uid']
        if uid in entries:
            raise ValueError('duplicate construction identity')
        entry = json.loads(artifact_path(root, row['manifest']).read_text())
        if entry['base']['video_uid'] != uid or entry['status'] != row['status']:
            raise ValueError('construction manifest identity/status mismatch')
        entries[uid] = entry
    return run, entries


def masks(root, entry):
    descriptor = entry['mask_file']; path = artifact_path(root, descriptor['path'])
    if sha256_file(path) != descriptor['sha256']:
        raise ValueError('construction mask checksum changed')
    with np.load(path, allow_pickle=False) as archive:
        result = archive['masks']
    if result.ndim != 3 or not np.isin(result, (0, 1)).all():
        raise ValueError('construction masks must be binary T,H,W')
    return result


def audit(root, manifest, previous=None):
    run, entries = completed_entries(root)
    bases = read_jsonl(manifest); expected = {r['video_uid']: r for r in bases}
    if len(expected) != len(bases) or expected.keys() != entries.keys():
        raise ValueError('construction does not cover the full input manifest')
    previous_run, old = completed_entries(previous) if previous else (None, {})
    details, pairs = [], []
    categories = Counter(); subjects = defaultdict(Counter); generators = defaultdict(Counter)
    frame_reasons = Counter(); base_reasons = Counter(); fallbacks = Counter()
    for uid, entry in sorted(entries.items()):
        if entry['base'] != expected[uid]:
            raise ValueError('frozen base metadata changed')
        base_reasons.update(entry['rejection_reasons'])
        frame_reasons.update(reason for row in entry.get('frame_rejection_reasons', []) for reason in row)
        fallback = [x for x in entry.get('frame_recovery_fallbacks', []) if x]
        fallbacks.update(fallback)
        item = {'video_uid': uid, 'subject_en': entry['base']['subject_en'],
                'generator': entry['base']['generator'], 'status': entry['status'],
                'rejection_reasons': entry['rejection_reasons'], 'fallback_frames': len(fallback)}
        actual = None
        if 'mask_file' not in entry:
            kind = 'no_mask_artifact'; item['frames'] = None
        else:
            actual = masks(root, entry); areas = actual.mean((1, 2)); zero = areas == 0
            if (not np.array_equal(actual.sum((1, 2)), entry['area_pixels'])
                    or not np.array_equal(areas, entry['area_ratio'])):
                raise ValueError('reported construction area does not match actual masks')
            valid = (areas >= .01) & (areas <= .50)
            if (entry['status'] == 'accepted') != bool(valid.all()):
                raise ValueError('construction acceptance disagrees with actual area gates')
            kind = ('accepted' if entry['status'] == 'accepted' else 'all_frames_empty' if zero.all()
                    else 'some_frames_empty' if zero.any() else 'nonempty_area_gate_failure')
            item.update(frames=len(areas), empty_frames=int(zero.sum()),
                        area_valid_frames=int(valid.sum()), area_valid_frame_fraction=float(valid.mean()),
                        min_area_fraction=float(areas.min()), median_area_fraction=float(np.median(areas)),
                        max_area_fraction=float(areas.max()))
        item['disjoint_category'] = kind; details.append(item)
        categories[kind] += 1; subjects[item['subject_en']][kind] += 1; generators[item['generator']][kind] += 1
        if uid in old:
            before = old[uid]; pair = {'video_uid': uid, 'fresh_status': entry['status'],
                'cached_status': before['status'], 'status_matches': entry['status'] == before['status']}
            if actual is not None and 'mask_file' in before:
                cached = masks(previous, before)
                pair.update(subject_masks_identical=bool(np.array_equal(actual, cached)),
                    source_video_sha256_matches=entry['source_video_sha256'] == before['source_video_sha256'])
            if entry['status'] == before['status'] == 'accepted':
                pair['clean_frame_hashes_identical'] = ([r['sha256'] for r in entry['frames']] ==
                                                       [r['sha256'] for r in before['frames']])
            pairs.append(pair)
    numerical_rejected = [d for d in details if d['status'] != 'accepted' and d['frames'] is not None]
    summary = {'total_candidates': len(entries), 'accepted': categories['accepted'],
        'rejected': len(entries)-categories['accepted'], 'disjoint_categories': dict(categories),
        'by_subject': {k: dict(v) for k, v in sorted(subjects.items())},
        'by_generator': {k: dict(v) for k, v in sorted(generators.items())},
        'base_rejection_reasons_nonexclusive': dict(base_reasons),
        'frame_rejection_reasons_nonexclusive': dict(frame_reasons), 'frame_recovery_fallbacks': dict(fallbacks),
        'rejected_with_at_least_80_percent_frames_in_area_gates': sum(d['area_valid_frame_fraction'] >= .8 for d in numerical_rejected),
        'rejected_with_exactly_one_empty_frame': sum(d['empty_frames'] == 1 for d in numerical_rejected),
        'interpretation': 'Frame-area coverage is not semantic completeness or IoU. Empty construction masks do not establish subject absence. No candidate is added, removed, or rescored by this audit.',
        'provenance': {'construction_run_sha256': sha256_file(root/'run.json'),
            'construction_index_sha256': sha256_file(root/'index.jsonl'), 'manifest_sha256': sha256_file(manifest),
            'audit_source_sha256': sha256_file(Path(__file__))}}
    if previous:
        summary['fresh_cached_parity'] = {'expected_overlap': len(expected.keys() & old.keys()),
            'checked_overlap': len(pairs), 'mask_pairs': sum('subject_masks_identical' in p for p in pairs),
            'all_statuses_match': all(p['status_matches'] for p in pairs),
            'all_subject_masks_match': all(p.get('subject_masks_identical', True) for p in pairs),
            'all_source_videos_match': all(p.get('source_video_sha256_matches', True) for p in pairs),
            'all_accepted_clean_frame_hashes_match': all(p.get('clean_frame_hashes_identical', True) for p in pairs),
            'previous_index_sha256': previous_run['index_sha256']}
    return summary, details, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('construction', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    summary, details, pairs = audit(args.construction, args.manifest, args.previous)
    output = new_output(args.output)
    write_json(output/'summary.json', summary); write_json(output/'per_base.json', details)
    write_json(output/'fresh_cached_pairs.json', pairs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
