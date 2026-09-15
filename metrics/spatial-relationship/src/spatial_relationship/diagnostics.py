from __future__ import annotations

import os
from enum import Enum
from typing import Iterable

from .schemas import FrameAuditResult, OrderedRelationQuery, VideoAuditResult
from .serialization import to_jsonable


class DiagnosticsLevel(str, Enum):
    FULL = "full"
    COMPACT = "compact"
    OFF = "off"


def diagnostics_level(video_count: int) -> DiagnosticsLevel:
    configured = os.environ.get("VBENCH_AUDIT_DIAGNOSTICS")
    if configured:
        try:
            return DiagnosticsLevel(configured.lower())
        except ValueError as exc:
            raise ValueError("VBENCH_AUDIT_DIAGNOSTICS must be full, compact, or off") from exc
    return DiagnosticsLevel.FULL if video_count == 1 else DiagnosticsLevel.COMPACT


def aggregate_video(
    video: str,
    prompt: str,
    query: OrderedRelationQuery,
    frame_results: Iterable[FrameAuditResult],
    sampled_frame_indices: Iterable[int],
    *,
    condition_on_detection: bool = False,
) -> VideoAuditResult:
    """Average the frame scores over the officially sampled frames.

    ``condition_on_detection`` restricts the denominator to frames where the
    detector returned a candidate for both roles (plan section 9.6, setting 1:
    "both A and B are correctly detected").  Without it a frame whose object was
    never detected contributes a zero, which the CPA instrument then reads as a
    wrong direction rather than as an abstention.
    """

    frames = tuple(frame_results)
    if condition_on_detection:
        scored = tuple(
            frame for frame in frames
            if frame.subject_candidate_ids and frame.object_candidate_ids
        )
        method = "mean_over_frames_with_both_roles_detected"
    else:
        scored = frames
        method = "mean_over_16_officially_sampled_frames"
    score = sum(frame.frame_score for frame in scored) / len(scored) if scored else 0.0
    return VideoAuditResult(
        video, prompt, query, tuple(sampled_frame_indices), frames, score, method, len(scored)
    )


def serialize_video_diagnostics(result: VideoAuditResult, level: DiagnosticsLevel) -> dict[str, object] | None:
    if level == DiagnosticsLevel.OFF:
        return None
    return to_jsonable(result.to_dict(include_frames=level == DiagnosticsLevel.FULL))
