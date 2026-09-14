from __future__ import annotations

from .schemas import Box, GeometryEvidence

IOU_THRESHOLD = 0.1

_RELATION_ALIASES = {
    "left": "on the left of",
    "right": "on the right of",
    "top": "on the top of",
    "bottom": "on the bottom of",
    "on the left of": "on the left of",
    "on the right of": "on the right of",
    "on the top of": "on the top of",
    "on the bottom of": "on the bottom of",
}


def normalize_relation(relation: str) -> str:
    try:
        return _RELATION_ALIASES[relation.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported spatial relationship: {relation!r}") from exc


def center_distances(subject_box: Box, object_box: Box) -> tuple[float, float]:
    subject_center = ((subject_box[0] + subject_box[2]) / 2, (subject_box[1] + subject_box[3]) / 2)
    object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
    return object_center[0] - subject_center[0], object_center[1] - subject_center[1]


def intersection_over_union(box_a: Box, box_b: Box) -> float:
    x_overlap = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        raise ValueError("bounding boxes must have positive union area")
    return intersection / union


def official_position_score(relation: str, box_a: Box, box_b: Box, iou_threshold: float = IOU_THRESHOLD) -> float:
    """Exact official axis-dominance and IoU behavior for the four supported relations."""

    relation = normalize_relation(relation)
    dx, dy = center_distances(box_a, box_b)
    iou = intersection_over_union(box_a, box_b)
    horizontal = relation in {"on the left of", "on the right of"}
    dominant = abs(dx) > abs(dy) if horizontal else abs(dy) > abs(dx)
    if not dominant:
        return 0.0
    return 1.0 if iou < iou_threshold else iou_threshold / iou


def ordered_position_score(relation: str, subject_box: Box, object_box: Box, iou_threshold: float = IOU_THRESHOLD) -> GeometryEvidence:
    relation = normalize_relation(relation)
    dx, dy = center_distances(subject_box, object_box)
    iou = intersection_over_union(subject_box, object_box)
    horizontal = relation in {"on the left of", "on the right of"}
    axis_dominance = abs(dx) > abs(dy) if horizontal else abs(dy) > abs(dx)
    direction_match = {
        "on the left of": dx > 0,
        "on the right of": dx < 0,
        "on the top of": dy > 0,
        "on the bottom of": dy < 0,
    }[relation]
    if not axis_dominance or not direction_match:
        score = 0.0
    else:
        score = 1.0 if iou < iou_threshold else iou_threshold / iou
    return GeometryEvidence(dx, dy, iou, axis_dominance, direction_match, score)
