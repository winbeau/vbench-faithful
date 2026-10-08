"""Direction-aware Repair geometry over unchanged GRiT detections.

Image coordinates increase rightwards/downwards. Relations describe the subject
relative to the object. Origin lives separately in official_replay; these rules
must never be substituted into an official baseline.
"""
from __future__ import annotations

import math
import re

from .experiments import RELATIONS
from .official_replay import position_score as unsigned_position_score

OPPOSITE = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}
BACKEND_VERSION = 'repair-v2'


def entity_name(name, aliases=None):
    """Remove a leading article, then serialize using a public text inventory.

    No detector labels, expected test target, fuzzy matching, or head-noun
    guessing is involved. Unknown/qualified names stay unknown/qualified.
    """
    value = ' '.join(name.lower().split())
    value = re.sub(r'^(?:a|an|the)\s+', '', value)
    return (aliases or {}).get(value, value)


def validate_box(box):
    if (len(box) != 4 or any(not math.isfinite(v) for v in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('spatial boxes must have finite coordinates and positive area')


def position_score(relation, subject, object_, iou_threshold=0.1):
    """Official axis/IoU weight gated by the requested, ordered direction."""
    if relation not in RELATIONS:
        raise ValueError('unsupported spatial direction: ' + str(relation))
    if not 0 < iou_threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    validate_box(subject)
    validate_box(object_)
    dx = (object_[0] + object_[2] - subject[0] - subject[2]) / 2
    dy = (object_[1] + object_[3] - subject[1] - subject[3]) / 2
    direction_ok = {'left': dx > 0, 'right': dx < 0,
                    'above': dy > 0, 'below': dy < 0}[relation]
    if not direction_ok:
        return 0.0
    return unsigned_position_score(RELATIONS[relation], subject, object_, iou_threshold)


def frame_score(triple, frame, *, signed=True):
    """Existential score over distinct subject/object instances, never A-or-B.

    With multiple instances both opposite relations can legitimately have a
    witness. Such frames are not an unambiguous direction-flip test. ``signed``
    is exposed only for the ordered-pair/unsigned-geometry ablation.
    """
    relation = triple['relation']
    if relation not in RELATIONS:
        raise ValueError('unsupported spatial direction: ' + str(relation))
    subjects = [(i, item['box']) for i, item in enumerate(frame) if item['label'] == triple['subject']]
    objects = [(i, item['box']) for i, item in enumerate(frame) if item['label'] == triple['object']]
    # A corrupt relevant box is a failed frame, not a successful negative.
    for _, box in subjects + objects:
        validate_box(box)
    values = [0.0]
    for i, subject in subjects:
        for j, object_ in objects:
            if i == j:
                continue
            values.append(position_score(relation, subject, object_) if signed else
                          unsigned_position_score(RELATIONS[relation], subject, object_))
    return max(values)
