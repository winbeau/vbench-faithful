"""Compare matching construction inputs, never equating mask area with truth."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .common import sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def load_completed(root):
    run = json.loads((root/'run.json').read_text())
    if not run.get('completed') or sha256_file(root/'index.jsonl') != run['index_sha256']:
        raise ValueError('complete unchanged construction required')
    result = {}
    for entry in read_jsonl(root/'index.jsonl'):
        path = artifact_path(root, entry['manifest'])
        if sha256_file(path) != entry['manifest_sha256']:
            raise ValueError('construction manifest changed')
        uid = entry['video_uid']
        if uid in result:
            raise ValueError('duplicate construction input')
        result[uid] = json.loads(path.read_text())
    return run, result


def population_summary(rows):
    counts, reasons, targets, empty_causes, bands = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in rows:
        counts[row['status']] += 1
        reasons.update(row['rejection_reasons'])
        targets[row.get('construction_target') or 'NO_TARGET'] += 1
        area = np.asarray(row.get('foreground_fraction', []))
        if not len(area):
            continue
        bands['below_15_percent' if area.mean() < .15 else 'above_25_percent' if area.mean() > .25 else '15_to_25_percent'] += 1
        if not area.any():
            target = row['construction_target']
            fractions = row['semantic_class_fractions']
            if target is not None and np.any(fractions[target]):
                empty_causes['semantic_present_refinement_empty'] += 1
            elif any(np.any(x) for x in fractions.values()):
                empty_causes['selected_absent_other_class_present'] += 1
            else:
                empty_causes['no_detection_in_configured_vocabulary'] += 1
    return {'count': len(rows), 'statuses': dict(counts), 'area_bands': dict(bands),
            'all_empty': sum(empty_causes.values()), 'empty_causes': dict(empty_causes),
            'overlapping_rejection_reasons': dict(reasons), 'selected_targets': dict(targets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old_run, old = load_completed(args.old); new_run, new = load_completed(args.new)
    if set(old) != set(new) or old_run['input_manifest_sha256'] != new_run['input_manifest_sha256']:
        raise ValueError('construction populations are not the same')
    rows = []
    for uid, before in old.items():
        after = new[uid]
        if before['base'] != after['base'] or before['video_sha256'] != after['video_sha256']:
            raise ValueError('construction comparison changed source video identity')
        old_area = np.asarray(before.get('foreground_fraction', []))
        new_area = np.asarray(after.get('foreground_fraction', []))
        rows.append({'video_uid': uid, 'prompt_id': before['base']['prompt_id'], 'generator': before['base']['generator'],
                     'old_target': before.get('construction_target'), 'new_target': after.get('construction_target'),
                     'old_status': before['status'], 'new_status': after['status'],
                     'old_mask_available': bool(len(old_area)), 'new_mask_available': bool(len(new_area)),
                     'old_empty': bool(len(old_area) and not old_area.any()), 'new_empty': bool(len(new_area) and not new_area.any()),
                     'old_mean_mask_area': float(old_area.mean()) if len(old_area) else None,
                     'new_mean_mask_area': float(new_area.mean()) if len(new_area) else None,
                     'new_localization_state': after.get('localization_state'),
                     'new_rejection_reasons': after['rejection_reasons']})
    output = new_output(args.output)
    report = {'old_index_sha256': old_run['index_sha256'], 'new_index_sha256': new_run['index_sha256'],
              'input_manifest_sha256': new_run['input_manifest_sha256'], 'metric_scores_used': False,
              'scope': 'same 680 official background dev videos; prediction area is not true subject area or coverage',
              'old': population_summary(list(old.values())), 'new': population_summary(list(new.values())),
              'first182_old': population_summary(list(old.values())[:182]),
              'first182_new': population_summary([new[uid] for uid in list(old)[:182]]),
              'old_empty_now_nonempty': sum(r['old_empty'] and r['new_mask_available'] and not r['new_empty'] for r in rows),
              'old_nonempty_now_empty': sum(r['old_mask_available'] and not r['old_empty'] and r['new_empty'] for r in rows),
              'source_sha256': sha256_file(Path(__file__))}
    write_json(output/'comparison.json', report)
    with (output/'cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output/'cases.json', rows)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
