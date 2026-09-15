from __future__ import annotations

from typing import Any, Iterable

from ..diagnostics import DiagnosticsLevel, aggregate_video, serialize_video_diagnostics
from ..models import AblationMode, detection_from_raw, evaluate_frame
from ..schemas import OrderedRelationQuery, VideoAuditResult


def score_predictions(
    video: str,
    prompt: str,
    query: OrderedRelationQuery,
    predictions: Iterable[Iterable[Any]],
    *,
    sampled_frame_indices: Iterable[int] | None = None,
    mode: AblationMode = AblationMode.ORDERED_ROLE_IDENTITY_ASSIGNMENT,
    condition_on_detection: bool = False,
) -> VideoAuditResult:
    prediction_list = list(predictions)
    indices = list(sampled_frame_indices) if sampled_frame_indices is not None else list(range(len(prediction_list)))
    if len(indices) != len(prediction_list):
        raise ValueError("sampled_frame_indices must align with predictions")
    frame_results = []
    for frame_index, raw_detections in zip(indices, prediction_list):
        detections = tuple(detection_from_raw(raw, detection_id) for detection_id, raw in enumerate(raw_detections))
        frame_results.append(evaluate_frame(query, detections, frame_index, mode=mode))
    return aggregate_video(
        video, prompt, query, frame_results, indices, condition_on_detection=condition_on_detection
    )


def result_payload(result: VideoAuditResult, level: DiagnosticsLevel) -> dict[str, object]:
    return {
        "prompt": result.prompt,
        "subject": result.query.subject,
        "relation": result.query.relation,
        "object": result.query.object,
        "backend": "audit",
        "failure_reason": None if result.video_score > 0 else _video_failure_reason(result),
        "diagnostics": serialize_video_diagnostics(result, level),
    }


def _video_failure_reason(result: VideoAuditResult) -> str | None:
    if result.ambiguous_role_count:
        return "same_class_role_ambiguity"
    if result.missing_subject_count:
        return "missing_subject"
    if result.missing_object_count:
        return "missing_object"
    if result.frame_results and all(frame.frame_reason == "direction_mismatch" for frame in result.frame_results):
        return "direction_mismatch"
    return "relation_not_satisfied" if result.frame_results else "no_sampled_frames"
