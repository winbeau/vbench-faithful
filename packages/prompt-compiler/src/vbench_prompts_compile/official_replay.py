"""Replay VBench fd18b3d decision rules over frozen evidence, without GPU imports.

Preserve the upstream spatial sign blindness and same-label box pairing. These
are research findings, not implementation bugs to silently fix in an Origin run.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path

UPSTREAM_SHA = 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'


def position_score(locality, a, b, iou_threshold=0.1):
    dx = (b[0] + b[2] - a[0] - a[2]) / 2
    dy = (b[1] + b[3] - a[1] - a[3]) / 2
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    iou = intersection / union  # upstream also raises on degenerate zero-area boxes
    if locality in 'on the right of' or locality in 'on the left of':
        dominant = abs(dx) > abs(dy)
    elif locality in 'on the bottom of' or locality in 'on the top of':
        dominant = abs(dy) > abs(dx)
    else:
        dominant = False
    if not dominant:
        return 0.0
    return 1.0 if iou < iou_threshold else iou_threshold / iou


def spatial_scores(target, frames):
    scores = []
    for frame in frames:
        boxes = [item['box'] for item in frame if item['label'] in (target['object_a'], target['object_b'])]
        scores.append(max([0.0] + [position_score(target['relationship'], a, b) for a, b in combinations(boxes, 2)]))
    return scores


def scene_scores(scene_key, captions):
    # Exact case-sensitive substring membership including literal space splitting.
    return [int(all(word in caption for word in scene_key.split(' '))) for caption in captions]


def object_scores(object_key, frame_labels):
    a, b = object_key.split(' and ')
    return [int(a.strip() in labels and b.strip() in labels) for labels in frame_labels]


def action_filename_label(video_path):
    return Path(video_path).name.lower().split('-')[0].split('person is ')[-1].split('_')[0]


def action_score(label, top5):
    return int(any(entry['label'] == label and round(float(entry['score']), 4) >= .85 for entry in top5[:5]))
