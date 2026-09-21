"""Audit recorded natural localization failures without reading score values."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .common import sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def diagnose(row):
    diagnostics = row['localizer_diagnostics']
    fractions = diagnostics['foreground_fraction']
    traces = {item['frame']: item for item in diagnostics['caption_fallback_frames']}
    frames = row['num_frames']
    if (len(fractions) != frames or len(traces) != len(diagnostics['caption_fallback_frames'])
            or any(index < 0 or index >= frames for index in traces)
            or len(traces) != diagnostics['num_empty_before_fallback']
            or any(not 0 <= value <= 1 for value in fractions)):
        raise ValueError('inconsistent recorded localization coverage')
    counts, selected = Counter(), Counter()
    for index, fraction in enumerate(fractions):
        if index not in traces:
            if fraction <= 0:
                raise ValueError('base empty frame bypassed caption fallback')
            counts['base_coco80_nonempty'] += 1
            continue
        trace = traces[index]
        captions = trace['captions']
        choice = trace['selected_index']
        eligible = {item['index']: item for item in captions if item['eligible']}
        if choice is not None:
            if choice not in eligible:
                raise ValueError('selected caption violates frozen eligibility')
            selected[eligible[choice]['head']['label']] += 1
            counts['caption_sam_nonempty' if fraction > 0 else 'selected_box_sam_empty'] += 1
        else:
            if eligible or fraction > 0:
                raise ValueError('caption selection does not match recorded masks')
            if not captions:
                counts['no_region_captions'] += 1
            elif not any(item['head']['kind'] == 'object' for item in captions):
                counts['only_scene_or_unknown_heads'] += 1
            else:
                counts['object_boxes_failed_frozen_eligibility'] += 1
    empty = sum(count for reason, count in counts.items()
                if reason not in ('base_coco80_nonempty', 'caption_sam_nonempty'))
    if empty != diagnostics['num_empty_foreground_frames'] or sum(counts.values()) != frames:
        raise ValueError('failure partition does not reproduce observed coverage')
    return {
        'video_uid': row['video_uid'], 'generator': row['generator'], 'prompt_id': row['prompt_id'],
        'num_frames': frames, 'num_empty': empty, 'frame_partition': dict(counts),
        'selected_caption_head_labels': dict(selected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run/'run.json').read_text())
    if not run.get('completed') or sha256_file(args.run/'scores.jsonl') != run['scores_sha256']:
        raise ValueError('complete hash-verified natural run required')
    rows = read_jsonl(args.run/'scores.jsonl')
    if len(rows) != run['video_count'] or len({row['video_uid'] for row in rows}) != len(rows):
        raise ValueError('missing or duplicated natural videos')
    cases = [diagnose(row) for row in rows if row['status'] == 'completed']
    totals, selected = Counter(), Counter()
    for case in cases:
        totals.update(case['frame_partition'])
        selected.update(case['selected_caption_head_labels'])
    result = {
        'scores_sha256': run['scores_sha256'], 'run_sha256': sha256_file(args.run/'run.json'),
        'source_sha256': sha256_file(Path(__file__)),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'frames': sum(case['num_frames'] for case in cases), 'frame_partition': dict(totals),
        'selected_caption_head_labels': dict(selected),
        'all_empty_video_count': sum(case['num_empty'] == case['num_frames'] for case in cases),
        'per_video': cases,
        'scope': 'Recorded model-output coverage only; no score values or preference labels used. '
                 'An empty proposal list is not evidence that the actual image has no foreground. '
                 'A nonempty mask is not a semantic accuracy annotation.',
    }
    output = new_output(args.output)
    write_json(output/'diagnosis.json', result)
    print(json.dumps({key: result[key] for key in ('status_counts', 'frames', 'frame_partition',
                                                  'all_empty_video_count')}))


if __name__ == '__main__':
    main()
