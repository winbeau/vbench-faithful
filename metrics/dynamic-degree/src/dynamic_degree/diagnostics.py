from __future__ import annotations

import os
from enum import Enum
from typing import Any, Iterable

from .schemas import AuditVideoResult, ChannelEvidence, TransitionEvidence


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


def aggregate_channel(
    transitions: Iterable[TransitionEvidence],
    channel: str,
    significant_threshold: float | None,
    *,
    duration_weighted: bool = True,
    intensity_field: str | None = None,
) -> ChannelEvidence:
    """Duration-weighted mean channel evidence.

    ``intensity_field`` selects which per-transition magnitude is reported as
    ``motion_intensity``.  The audit path passes the lag-normalised intensity so
    the reported magnitude does not carry the sampling interval; the default
    keeps the ballistic ``*_speed`` fields, which is also what the threshold
    comparison and the temporal-coverage fraction use.
    """
    speed_field = {"apparent": "apparent_speed", "camera": "global_speed", "residual": "residual_speed"}[channel]
    magnitude_field = intensity_field or speed_field
    valid = [
        transition
        for transition in transitions
        if transition.valid
        and getattr(transition, magnitude_field) is not None
        and transition.dt_seconds > 0
    ]
    duration = sum(transition.dt_seconds for transition in valid)
    if not valid or duration <= 0:
        return ChannelEvidence(channel, None, None, 0, 0.0)
    if duration_weighted:
        intensity = sum(float(getattr(transition, magnitude_field)) * transition.dt_seconds for transition in valid) / duration
    else:
        intensity = sum(float(getattr(transition, magnitude_field)) for transition in valid) / len(valid)
    coverage = None
    if significant_threshold is not None:
        # Compared in the same domain as the reported intensity, so the
        # static/moving fraction does not inherit the sampling interval.
        coverage = sum(
            transition.dt_seconds
            for transition in valid
            if getattr(transition, magnitude_field) is not None
            and float(getattr(transition, magnitude_field)) > significant_threshold
        ) / duration
    return ChannelEvidence(channel, float(intensity), None if coverage is None else float(coverage), len(valid), float(duration))


def serialize_audit_result(result: AuditVideoResult, level: DiagnosticsLevel) -> dict[str, Any] | None:
    if level == DiagnosticsLevel.OFF:
        return None
    transitions = [transition.to_dict(include_affine=level == DiagnosticsLevel.FULL) for transition in result.transitions]
    return {
        "sampling": {
            "sampled_source_frame_indices": list(result.sampled_source_frame_indices),
            "timestamps": list(result.timestamps),
            "timestamp_source": result.timestamp_source,
            "source_fps": result.source_fps,
            "sampling_interval": result.sampling_interval,
            "delta_t": [transition.dt_seconds for transition in result.transitions],
            "frame_shape": list(result.frame_shape),
        },
        "transitions": transitions,
        "audit_video": {
            "apparent": result.apparent.to_dict(),
            "camera": result.camera.to_dict(),
            "residual": result.residual.to_dict(),
            "valid_transition_count": result.apparent.valid_transition_count,
            "valid_transition_duration": result.apparent.valid_duration,
            "prompt_motion_target": result.target_decision.to_dict(),
            "selected_evidence_channel": result.selected_evidence_channel,
            "task_relevant_motion_evidence": result.task_relevant_motion_evidence,
            "scalar_score": result.score,
            "scalar_score_source": result.scalar_score_source,
            "scalar_score_independently_calibrated": result.scalar_score_independently_calibrated,
            "aggregation_method": (
                "duration_weighted_mean_speed_and_duration_fraction"
                if result.component_provenance["duration_persistence"]
                else "unweighted_mean_transition_motion_intensity_no_temporal_coverage"
            ),
            "component_provenance": result.component_provenance,
            "time_normalization": {
                "exponent": result.time_normalization_exponent,
                "exponent_source": result.time_normalization_exponent_source,
                "exponent_independently_calibrated": result.time_normalization_exponent_independently_calibrated,
                "intensity_units": (
                    "image_diagonals_per_second" if result.time_normalization_exponent == 1.0
                    else f"image_diagonals_per_second_pow_{result.time_normalization_exponent:g}"
                ),
                "ballistic_default_exponent": 1.0,
                "role": (
                    "applied_to_motion_intensity_and_scalar; coverage_fraction_still_uses_"
                    "the_ballistic_speed_against_the_official_derived_threshold"
                ),
            },
            "lag_scaling": None if result.lag_scaling is None else result.lag_scaling.to_dict(),
        },
        "threshold": {
            "value": result.threshold.value,
            "units": result.threshold.units,
            "exponent": result.threshold.exponent,
            "source": result.threshold.source,
            "independently_calibrated": result.threshold.independently_calibrated,
            "role": "compared against the lag-normalised intensity in the same exponent domain",
        },
        "boundary": {
            "official_count_num": result.official_count_num,
            "audit_effective_count_num": result.audit_effective_count_num,
            "boundary_fix_applied": result.boundary_fix_applied,
            "role": "compatibility_diagnostics_only_not_used_by_audit_score",
        },
    }
