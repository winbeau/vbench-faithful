"""Check same-input localizer comparisons and independent natural clean parity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.counterfactual.common import sha256_file
from scripts.counterfactual.subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def verified(path):
    run = json.loads((path/'run.json').read_text())
    rows = read_jsonl(path/'scores.jsonl')
    if (not run.get('completed') or sha256_file(path/'scores.jsonl') != run['scores_sha256']
            or len(rows) != run['video_count'] or len({row['video_uid'] for row in rows}) != len(rows)):
        raise ValueError('complete hash-verified population required')
    return run, {row['video_uid']: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', required=True, help='Run directory pattern containing {method}.')
    parser.add_argument('--natural', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if '{method}' not in args.pattern:
        raise ValueError('explicit localizer directory placeholder required')
    natural_run, natural = verified(args.natural)
    natural_protocol = json.loads((args.natural/'protocol.json').read_text())
    review_sha = sha256_file(args.review)
    cases, hashes, checks = {}, {}, []
    for name, natural_method, mask_key in [('frozen19', 'frozen19', None),
                                          ('coco80', 'coco80', 'coco80_masks'),
                                          ('caption-v2', 'caption', 'masks')]:
        directory = Path(args.pattern.format(method=name))
        run, rows = verified(directory)
        if run['review_sha256'] != review_sha or set(rows) != set(natural) or len(rows) != 680:
            raise ValueError('review or natural input population mismatch')
        if (run['method_protocol_sha256'] != natural_protocol['method_protocol_sha256']
                or run['source_sha256'] != natural_run['source_sha256']
                or run['primary_method'] != natural_run['primary_method']):
            raise ValueError('frozen scoring formula or source changed')
        if name == 'caption-v2' and run['vocabulary_protocol_sha256'] != natural_run['candidate_protocol_sha256']:
            raise ValueError('caption candidate differs from natural reference')
        cases[name], hashes[name] = rows, sha256_file(directory/'run.json')
        for uid, row in rows.items():
            if row['status'] != 'completed':
                continue
            clean, original = row['variants']['clean'], natural[uid]
            error = abs(clean['scores'][run['primary_method']]-original['scores'][natural_method]['score'])
            if (clean['frame_array_sha256'] != original['frame_array_sha256']
                    or not math.isfinite(error) or error > 1e-6):
                raise ValueError('natural/PNG clean parity failed')
            same_mask = None
            if mask_key is not None:
                paths = [(directory, clean['scoring_mask']), (args.natural, original['scoring_mask'])]
                verified_paths = []
                for root, record in paths:
                    path = artifact_path(root, record['path'])
                    if sha256_file(path) != record['sha256']:
                        raise ValueError('scoring mask artifact changed')
                    verified_paths.append(path)
                with np.load(verified_paths[0], allow_pickle=False) as a, np.load(verified_paths[1], allow_pickle=False) as b:
                    same_mask = bool(np.array_equal(a['masks'], b[mask_key]))
                if not same_mask:
                    raise ValueError('natural and PNG scoring masks differ')
            checks.append({'video_uid': uid, 'method': natural_method, 'natural_input_equal': True,
                           'score_error': error, 'natural_mask_equal': same_mask})
    baseline, compared = cases['frozen19'], 0
    for rows in cases.values():
        for uid, row in rows.items():
            if row['status'] != baseline[uid]['status']:
                raise ValueError('localizer comparison silently changed the evaluated population')
            if row['status'] != 'completed':
                continue
            if set(row['variants']) != set(baseline[uid]['variants']) or len(row['variants']) != 8:
                raise ValueError('intervention population mismatch')
            for variant, value in row['variants'].items():
                reference = baseline[uid]['variants'][variant]
                if (value['frame_array_sha256'] != reference['frame_array_sha256']
                        or value['scores']['official'] != reference['scores']['official']):
                    raise ValueError('localizer comparison changed pixels or origin scores')
                compared += 1
    output = new_output(args.output)
    result = {'full_population_count': len(baseline), 'compared_video_variants_including_reference': compared,
        'all_actual_input_hashes_equal': True, 'max_origin_difference_between_localizers': 0,
        'natural_clean_parity': checks, 'max_natural_clean_score_difference': max(row['score_error'] for row in checks),
        'natural_masks_compared': sum(row['natural_mask_equal'] is not None for row in checks),
        'score_runs_sha256': hashes, 'natural_scores_sha256': natural_run['scores_sha256'],
        'review_sha256': review_sha, 'source_sha256': sha256_file(Path(__file__))}
    write_json(output/'validation.json', result)
    print(json.dumps({key: result[key] for key in ('full_population_count', 'compared_video_variants_including_reference',
          'max_natural_clean_score_difference', 'natural_masks_compared')}))


if __name__ == '__main__':
    main()
