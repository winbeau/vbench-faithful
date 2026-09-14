from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from .relation import official_position_score, ordered_position_score
from .schemas import Detection, FrameAuditResult, OrderedRelationQuery, RoleBoundCandidates


class AblationMode(str, Enum):
    OFFICIAL = "official"
    SIGNED_ONLY = "signed_only"
    ORDERED_ROLE = "ordered_role"
    ORDERED_ROLE_IDENTITY_ASSIGNMENT = "ordered_role_identity_assignment"


def detection_from_raw(raw: Any, detection_id: int) -> Detection:
    if isinstance(raw, Detection):
        return raw
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"unsupported detection representation: {raw!r}")
    label, raw_box = raw[0], raw[1]
    confidence = raw[2] if len(raw) >= 3 and isinstance(raw[2], (int, float)) else None
    box = tuple(float(value) for value in raw_box)
    return Detection(detection_id=detection_id, label=str(label), box=box, confidence=confidence)


def bind_role_candidates(query: OrderedRelationQuery, detections: Iterable[Detection], frame_index: int) -> RoleBoundCandidates:
    detection_list = tuple(detections)
    return RoleBoundCandidates(
        query=query,
        subject_candidates=tuple(item for item in detection_list if item.label == query.subject),
        object_candidates=tuple(item for item in detection_list if item.label == query.object),
        frame_index=frame_index,
    )


def assign_role_instance(candidates: Iterable[Detection], strategy: str = "identity_first") -> Detection | None:
    """Assign from identity-filtered candidates without receiving relation geometry."""

    if strategy != "identity_first":
        raise ValueError(f"unsupported assignment strategy: {strategy}")
    candidates = tuple(candidates)
    if not candidates:
        return None
    # Confidence is used only when the detector exposes it. Stable detection id
    # is the deterministic fallback and tie-breaker.
    return min(
        candidates,
        key=lambda item: (
            item.confidence is None,
            -(item.confidence if item.confidence is not None else 0.0),
        ),
    )


def _empty_frame(bound: RoleBoundCandidates, detections: tuple[Detection, ...], reason: str, ambiguous: bool = False) -> FrameAuditResult:
    return FrameAuditResult(
        frame_index=bound.frame_index,
        query=bound.query,
        detections=detections,
        subject_candidate_ids=tuple(item.detection_id for item in bound.subject_candidates),
        object_candidate_ids=tuple(item.detection_id for item in bound.object_candidates),
        assigned_subject_id=None,
        assigned_object_id=None,
        selected_subject_box=None,
        selected_object_box=None,
        geometry=None,
        frame_score=0.0,
        frame_reason=reason,
        role_identity_ambiguous=ambiguous,
    )


def evaluate_frame(
    query: OrderedRelationQuery,
    detections: Iterable[Detection],
    frame_index: int,
    *,
    mode: AblationMode = AblationMode.ORDERED_ROLE_IDENTITY_ASSIGNMENT,
    identity_descriptors_present: bool = False,
) -> FrameAuditResult:
    detection_list = tuple(detections)
    bound = bind_role_candidates(query, detection_list, frame_index)
    if not bound.subject_candidates:
        return _empty_frame(bound, detection_list, "missing_subject")
    if not bound.object_candidates:
        return _empty_frame(bound, detection_list, "missing_object")
    if query.subject == query.object and not identity_descriptors_present:
        return _empty_frame(bound, detection_list, "same_class_role_ambiguity", ambiguous=True)

    if mode == AblationMode.OFFICIAL:
        pooled = tuple(item for item in detection_list if item.label in {query.subject, query.object})
        scores = [official_position_score(query.relation, left.box, right.box) for index, left in enumerate(pooled) for right in pooled[index + 1 :]]
        score = max(scores, default=0.0)
        return FrameAuditResult(
            frame_index, query, detection_list,
            tuple(item.detection_id for item in bound.subject_candidates),
            tuple(item.detection_id for item in bound.object_candidates),
            None, None, None, None, None, score,
            "official_relation_pair_found" if score > 0 else "official_no_relation_pair",
        )

    if mode == AblationMode.SIGNED_ONLY:
        pooled = tuple(item for item in detection_list if item.label in {query.subject, query.object})
        evidence = [ordered_position_score(query.relation, left.box, right.box) for index, left in enumerate(pooled) for right in pooled[index + 1 :]]
        best = max(evidence, key=lambda item: item.score, default=None)
        score = best.score if best else 0.0
        return FrameAuditResult(
            frame_index, query, detection_list,
            tuple(item.detection_id for item in bound.subject_candidates),
            tuple(item.detection_id for item in bound.object_candidates),
            None, None, None, None, best, score,
            "signed_relation_pair_found" if score > 0 else "signed_no_relation_pair",
        )

    if mode == AblationMode.ORDERED_ROLE:
        candidates = [
            (subject, object_, ordered_position_score(query.relation, subject.box, object_.box))
            for subject in bound.subject_candidates
            for object_ in bound.object_candidates
            if subject.detection_id != object_.detection_id
        ]
        if not candidates:
            return _empty_frame(bound, detection_list, "same_detection_for_both_roles")
        subject, object_, geometry = max(candidates, key=lambda item: item[2].score)
    else:
        subject = assign_role_instance(bound.subject_candidates)
        object_ = assign_role_instance(bound.object_candidates)
        if subject is None:
            return _empty_frame(bound, detection_list, "missing_subject")
        if object_ is None:
            return _empty_frame(bound, detection_list, "missing_object")
        if subject.detection_id == object_.detection_id:
            return _empty_frame(bound, detection_list, "same_detection_for_both_roles", ambiguous=True)
        geometry = ordered_position_score(query.relation, subject.box, object_.box)

    if geometry.score > 0 and geometry.iou >= 0.1:
        reason = "relation_satisfied_with_iou_penalty"
    elif geometry.score > 0:
        reason = "relation_satisfied"
    elif not geometry.direction_match:
        reason = "direction_mismatch"
    else:
        reason = "axis_mismatch"
    return FrameAuditResult(
        frame_index=frame_index,
        query=query,
        detections=detection_list,
        subject_candidate_ids=tuple(item.detection_id for item in bound.subject_candidates),
        object_candidate_ids=tuple(item.detection_id for item in bound.object_candidates),
        assigned_subject_id=subject.detection_id,
        assigned_object_id=object_.detection_id,
        selected_subject_box=subject.box,
        selected_object_box=object_.box,
        geometry=geometry,
        frame_score=geometry.score,
        frame_reason=reason,
    )
