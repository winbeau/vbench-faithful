"""Conservative temporal confirmation of existing object detections.

Only entities and this video's ordered detection frames enter the scorer. A
positive detection needs a same-label box in an immediately adjacent frame.
This is not a tracker, an independent detector, or evidence of semantic absence:
unconfirmed positives are explicit abstentions and never fill negative frames.
"""
from __future__ import annotations

import math

BACKEND_VERSION = 'repair-v2'
MIN_ADJACENT_IOU = 0.5


def box_iou(a, b):
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    return intersection / ((a[2]-a[0]) * (a[3]-a[1]) +
                           (b[2]-b[0]) * (b[3]-b[1]) - intersection)


def index_frame(frame):
    """Reject corrupt evidence, including corrupt unrelated boxes, as missing."""
    if not isinstance(frame, list):
        raise ValueError('missing_or_invalid_detection_frame')
    indexed = {}
    for item in frame:
        if not isinstance(item, dict) or not isinstance(item.get('label'), str) or not item['label'].strip():
            raise ValueError('invalid_detection_label')
        box = item.get('box')
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
                box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('invalid_detection_box')
        indexed.setdefault(item['label'], []).append(box)
    return indexed


def video_scores(entities, frame_detections, *, confirmation='boxes', num_frames=16):
    """Require each current entity to have immediate temporal corroboration.

    ``labels`` is an explicit ablation without localization; production uses
    ``boxes`` and a fixed IoU >= 0.5. A missing neighbor cannot corroborate, and
    indices 0 and N-1 are not adjacent. All planned frame slots remain present.
    Entity requirements are sets, as in the prompt entities contract (no counts).
    """
    if confirmation not in {'boxes', 'labels'}:
        raise ValueError('unknown Objects confirmation method')
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError('positive frame count required')
    if (not isinstance(entities, list) or not entities or
            any(not isinstance(v, str) or not v.strip() for v in entities)):
        raise ValueError('nonempty entity list required')
    entities = list(dict.fromkeys(entities))
    if not isinstance(frame_detections, list) or len(frame_detections) > num_frames:
        raise ValueError('invalid frame sequence')
    frames, errors = [], []
    for i in range(num_frames):
        try:
            frames.append(index_frame(frame_detections[i] if i < len(frame_detections) else None))
        except ValueError as error:
            frames.append(None)
            errors.append({'frame': i, 'reason': str(error)})
    states, entity_states, raw_scores = [], [], []
    for i, frame in enumerate(frames):
        if frame is None:
            states.append('missing')
            entity_states.append({name: 'missing' for name in entities})
            raw_scores.append(0.)
            continue
        raw_scores.append(float(all(name in frame for name in entities)))
        neighbors = [frames[j] for j in (i-1, i+1) if 0 <= j < num_frames and frames[j] is not None]
        current = {}
        for name in entities:
            candidates = frame.get(name, [])
            if not candidates:
                current[name] = 'absent'
                continue
            matches = [box for neighbor in neighbors for box in neighbor.get(name, [])]
            confirmed = bool(matches) if confirmation == 'labels' else any(
                box_iou(a, b) >= MIN_ADJACENT_IOU for a in candidates for b in matches)
            current[name] = 'supported' if confirmed else 'unconfirmed'
        entity_states.append(current)
        states.append('absent' if 'absent' in current.values() else
                      'supported' if all(v == 'supported' for v in current.values()) else 'unconfirmed')
    scores = [float(state == 'supported') for state in states]
    covered = sum(state != 'missing' for state in states)
    return {'score': sum(scores) / num_frames, 'coverage': covered / num_frames,
            'abstention': states.count('unconfirmed') / num_frames,
            'missing': 1 - covered / num_frames, 'status': 'ok' if covered == num_frames else 'partial',
            'frame_scores': scores, 'errors': errors,
            'objects_resolution': {'version': BACKEND_VERSION, 'confirmation': confirmation,
                'min_adjacent_iou': MIN_ADJACENT_IOU if confirmation == 'boxes' else None,
                'frame_states': states, 'frame_entity_states': entity_states,
                'raw_frame_scores': raw_scores}}
