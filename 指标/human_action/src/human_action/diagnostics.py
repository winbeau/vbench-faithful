from __future__ import annotations

import os
from enum import Enum
from typing import Any

from .schemas import AuditActionResult, OfficialActionResult


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


def official_diagnostics(result: OfficialActionResult) -> dict[str, Any]:
    return {
        "target_action": result.target_action,
        "target_source": result.target_source,
        "sampled_frame_count": result.sampled_frame_count,
        "rounded_top5_actions": list(result.rounded_top5_actions),
        "rounded_top5_probabilities": list(result.rounded_top5_probabilities),
        "accepted_actions": list(result.accepted_actions),
        "threshold": result.threshold,
        "matched": result.matched,
        "aggregation": "exact_label_in_rounded_top5_above_threshold",
    }


def audit_diagnostics(
    result: AuditActionResult,
    level: DiagnosticsLevel,
) -> dict[str, Any] | None:
    if level == DiagnosticsLevel.OFF:
        return None
    payload = {
        "query": result.query.to_dict(),
        "source_frame_count": result.source_frame_count,
        "source_fps": result.source_fps,
        "full_clip": result.full_clip.to_dict(),
        "temporal_mean_target_probability": result.temporal_mean_probability,
        "temporal_coverage": result.temporal_coverage,
        "threshold": result.threshold,
        "threshold_source": result.threshold_source,
        "window_count": len(result.windows),
        "aggregation": "source-frame-count-weighted-window-mean-and-coverage",
        "scalarization": result.scalarization,
    }
    if level == DiagnosticsLevel.FULL:
        payload["windows"] = [window.to_dict() for window in result.windows]
    else:
        payload["window_target_probabilities"] = [
            window.classification.target_probability for window in result.windows
        ]
        payload["window_weights"] = [window.weight for window in result.windows]
    return payload


def audit_result_payload(
    result: AuditActionResult,
    level: DiagnosticsLevel,
) -> dict[str, Any]:
    task_evidence = result.task_evidence()
    return {
        "video": result.video,
        "prompt": result.query.prompt,
        "target_action": result.query.target_action,
        "target_source": result.query.target_source,
        "backend": "audit",
        "score": result.score,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "error": None,
        "full_clip_target_probability": result.full_clip.target_probability,
        "full_clip_target_rank": result.full_clip.target_rank,
        "temporal_mean_target_probability": result.temporal_mean_probability,
        "temporal_coverage": result.temporal_coverage,
        "task_relevant_action_evidence": task_evidence,
        "score_metadata": {
            "status": "succeeded_scalarized" if result.score is not None else "not_available",
            "source": "temporal_mean_target_probability",
            "formula": "sum(window_weight * target_probability) / sum(window_weight)",
            "scalarization": result.scalarization,
        },
        "diagnostics": audit_diagnostics(result, level),
    }
