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
) -> VideoAuditResult:
    frames = tuple(frame_results)
    score = sum(frame.frame_score for frame in frames) / len(frames) if frames else 0.0
    return VideoAuditResult(video, prompt, query, tuple(sampled_frame_indices), frames, score)


def serialize_video_diagnostics(result: VideoAuditResult, level: DiagnosticsLevel) -> dict[str, object] | None:
    if level == DiagnosticsLevel.OFF:
        return None
    return to_jsonable(result.to_dict(include_frames=level == DiagnosticsLevel.FULL))
