from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

import numpy as np

from ..diagnostics import DiagnosticsLevel, aggregate_channel, serialize_audit_result
from ..motion import (
    AffineEstimatorConfig,
    decompose_motion,
    fit_power_law_exponent,
    flow_magnitude,
    motion_statistics,
    resolve_lags,
    scale_intensity,
    top_fraction_mean,
)
from ..prompt_target import MotionTarget, parse_motion_target
from ..schemas import (
    AuditVideoResult,
    LagScalingEvidence,
    MotionThreshold,
    PromptTargetDecision,
    TimedFrameSequence,
    TransitionEvidence,
)


class AuditAblation(str, Enum):
    FULL = "full"
    WITHOUT_GLOBAL_COMPENSATION = "without_global_compensation"
    WITHOUT_TIME_NORMALIZATION = "without_time_normalization"
    WITHOUT_LAG_SCALING = "without_lag_scaling"
    WITHOUT_PROMPT_ROUTING = "without_prompt_routing"
    WITHOUT_CONTINUOUS_PERSISTENCE_AGGREGATION = "without_continuous_persistence_aggregation"


class AuditVariant(str, Enum):
    FULL = "full"
    TIME_ONLY = "time_only"
    SOURCE_ONLY = "source_only"
    DURATION_ONLY = "duration_only"
    SOURCE_TIME = "source_time"


class LagExponentMode(str, Enum):
    """How the time normalisation exponent is chosen.

    FIXED applies the benchmark's pre-registered diffusive exponent to every
    clip, which is what makes the reported dynamic-degree aggregate comparable
    across frame rates.  MEASURED fits each clip separately (a diagnostic: the
    fitted value is itself lag-window dependent, so it under-corrects).  BALLISTIC
    is the previous ``d / dt`` behaviour.
    """

    FIXED = "fixed"
    MEASURED = "measured"
    BALLISTIC = "ballistic"


@dataclass(frozen=True)
class AuditConfig:
    variant: AuditVariant = AuditVariant.FULL
    ablation: AuditAblation = AuditAblation.FULL
    significant_motion_threshold: float | None = None
    threshold_source: str | None = None
    threshold_independently_calibrated: bool = False
    affine: AffineEstimatorConfig = AffineEstimatorConfig()
    boundary_fix_enabled: bool = True
    # Sampling-lag calibration of the time normalisation.  The counterfactual FPS
    # ladder changes only the sampling interval, so dividing by dt**1 (ballistic)
    # is right only for constant-velocity motion.  Measured on the real
    # counterfactual clips, the reported displacement grows like dt**0.5, so the
    # default applies that fixed, pre-registered exponent and the per-clip fit is
    # kept as a diagnostic.  An explicit lag_exponent overrides both and must
    # declare a source, the same way an explicit threshold does.
    lag_scaling_enabled: bool = True
    lag_exponent_mode: LagExponentMode = LagExponentMode.FIXED
    default_lag_exponent: float = 0.5
    default_lag_exponent_source: str = (
        "diffusive_sqrt_lag_default_bootstrap_ci_0p39_0p65_not_independently_calibrated"
    )
    lag_set: tuple[int, ...] = (1, 2, 4)
    max_lag_frames: int = 4
    # The sampling interval at which the Official pixel threshold is defined.
    reference_lag_seconds: float = 0.125
    lag_exponent: float | None = None
    lag_exponent_source: str | None = None
    lag_exponent_independently_calibrated: bool = False
    scalar_score_hook: Callable[[dict[str, Any]], float] | None = None
    scalar_score_source: str | None = None
    scalar_score_independently_calibrated: bool = False

    def components(self) -> dict[str, bool]:
        if self.variant == AuditVariant.TIME_ONLY:
            return {"time_normalization": True, "source_decomposition": False, "duration_persistence": False, "task_routing": False}
        if self.variant == AuditVariant.SOURCE_ONLY:
            return {"time_normalization": False, "source_decomposition": True, "duration_persistence": False, "task_routing": False}
        if self.variant == AuditVariant.DURATION_ONLY:
            return {"time_normalization": False, "source_decomposition": False, "duration_persistence": True, "task_routing": False}
        if self.variant == AuditVariant.SOURCE_TIME:
            return {"time_normalization": True, "source_decomposition": True, "duration_persistence": False, "task_routing": False}
        return {"time_normalization": True, "source_decomposition": True, "duration_persistence": True, "task_routing": True}


def official_count_num(sampled_frame_count: int) -> int:
    return round(4 * (sampled_frame_count / 16.0))


def audit_effective_count_num(sampled_frame_count: int) -> int | None:
    if sampled_frame_count <= 1:
        return None
    return max(1, official_count_num(sampled_frame_count))


def derive_motion_threshold(
    frame_shape: tuple[int, int],
    config: AuditConfig,
    exponent: float,
) -> MotionThreshold:
    """The Official pixel threshold re-expressed in the applied normalisation.

    The Official threshold is a pixel count between adjacent frames at the
    nominal 8 fps sampling, so in the ``d / dt**exponent`` domain it is
    ``official_pixels / diagonal / reference_lag**exponent``.  Using the same
    exponent for the threshold and for the intensity keeps the static/moving
    decision frame-rate invariant: with ``d = A * dt**exponent`` the comparison
    reduces to ``A > official_pixels / diagonal / reference_lag**exponent``,
    independent of the sampling interval.
    """
    if not math.isfinite(exponent):
        raise ValueError("time-normalization exponent must be finite")
    if config.significant_motion_threshold is not None:
        value = float(config.significant_motion_threshold)
        if not math.isfinite(value) or value < 0:
            raise ValueError("significant motion threshold must be finite and non-negative")
        if not config.threshold_source:
            raise ValueError("an explicit significant motion threshold requires threshold_source")
        return MotionThreshold(
            value=value,
            units=_intensity_units(exponent),
            source=config.threshold_source,
            independently_calibrated=config.threshold_independently_calibrated,
            exponent=float(exponent),
        )
    height, width = frame_shape
    diagonal = math.hypot(height, width)
    official_pixels = 6.0 * min(height, width) / 256.0
    if exponent == 0.0:
        return MotionThreshold(
            value=float(official_pixels / diagonal),
            units=_intensity_units(0.0),
            source="ablation_vbench1_pixel_threshold_without_time_normalization",
            independently_calibrated=False,
            exponent=0.0,
        )
    value = official_pixels / diagonal / (config.reference_lag_seconds**exponent)
    return MotionThreshold(
        value=float(value),
        units=_intensity_units(exponent),
        source=(
            f"derived_from_vbench1_pixel_threshold_at_{config.reference_lag_seconds:g}s_"
            f"normalized_by_dt_pow_{exponent:g}_not_independently_calibrated"
        ),
        independently_calibrated=False,
        exponent=float(exponent),
    )


def _intensity_units(exponent: float) -> str:
    if exponent == 0.0:
        return "image_diagonals_per_transition"
    if exponent == 1.0:
        return "image_diagonals_per_second"
    return f"image_diagonals_per_second_pow_{exponent:g}"


def normalized_speed(displacement_pixels: float, frame_shape: tuple[int, int], dt_seconds: float) -> tuple[float, float]:
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    height, width = frame_shape
    diagonal = math.hypot(height, width)
    if diagonal <= 0:
        raise ValueError("frame diagonal must be positive")
    displacement = float(displacement_pixels) / diagonal
    return displacement, displacement / dt_seconds


def _measure_lag_scaling(
    sequence: TimedFrameSequence,
    transitions: list[TransitionEvidence],
    compute_flow: Callable[[Any, Any], Any],
    config: AuditConfig,
) -> LagScalingEvidence:
    """Fit the clip's own displacement-vs-lag power law from its frames.

    Lag 1 is reused from the main pass, so only the longer lags cost extra flow
    evaluations.  `straightness` compares the direct displacement at a lag with
    the sum of the lag-1 displacements over the same span, which separates
    genuine trajectory curvature from an estimator that saturates at large
    displacements.
    """
    frames = sequence.frames
    height, width = sequence.frame_shape
    diagonal = math.hypot(height, width)
    requested = [lag for lag in config.lag_set if lag <= config.max_lag_frames]
    lags = resolve_lags(requested, len(frames))
    if len(lags) < 2:
        return LagScalingEvidence(
            exponent=None,
            exponent_source="ballistic_default_insufficient_lags",
            measured_on_channel="apparent",
            lags=lags,
            lag_seconds=(),
            chord_displacement=(),
            path_displacement=(),
            straightness=(),
            pair_counts=(),
            fit_rmse=None,
            independently_calibrated=False,
        )

    # `transitions` holds one entry per adjacent sampled pair, in order, so it is
    # indexed by *position*: source indices are non-contiguous at every
    # downsampled rung, and keying by them would silently drop every longer lag.
    def adjacent(position: int) -> TransitionEvidence | None:
        if 0 <= position < len(transitions):
            entry = transitions[position]
            if entry.source_frame_index_a == frames[position].source_frame_index:
                return entry
        return None

    lag_seconds: list[float] = []
    chords: list[float] = []
    paths: list[float] = []
    straightness: list[float] = []
    counts: list[int] = []
    for lag in lags:
        pair_chords: list[float] = []
        pair_paths: list[float] = []
        pair_dts: list[float] = []
        for index in range(len(frames) - lag):
            span = [adjacent(position) for position in range(index, index + lag)]
            if any(entry is None or not entry.valid or entry.apparent_displacement is None for entry in span):
                continue
            if lag == 1:
                chord = float(span[0].apparent_displacement)
            else:
                try:
                    flow = np.asarray(compute_flow(frames[index].frame, frames[index + lag].frame), dtype=np.float64)
                except Exception:  # noqa: BLE001 - a diagnostic must not fail the clip
                    continue
                if flow.shape[:2] != sequence.frame_shape or flow.ndim != 3 or flow.shape[2] != 2:
                    continue
                chord = top_fraction_mean(flow_magnitude(flow)) / diagonal
            dt = frames[index + lag].timestamp_seconds - frames[index].timestamp_seconds
            if not math.isfinite(dt) or dt <= 0:
                continue
            pair_chords.append(float(chord))
            pair_paths.append(sum(float(entry.apparent_displacement) for entry in span))
            pair_dts.append(float(dt))
        if not pair_chords:
            continue
        mean_chord = float(np.mean(pair_chords))
        mean_path = float(np.mean(pair_paths))
        lag_seconds.append(float(np.mean(pair_dts)))
        chords.append(mean_chord)
        paths.append(mean_path)
        straightness.append(mean_chord / mean_path if mean_path > 0 else 1.0)
        counts.append(len(pair_chords))

    fit = (
        fit_power_law_exponent(lag_seconds, chords, weights=np.sqrt(counts))
        if len(chords) >= 2
        else None
    )
    return LagScalingEvidence(
        exponent=None if fit is None else fit[0],
        exponent_source="measured_within_clip_power_law" if fit is not None else "ballistic_default_fit_unavailable",
        measured_on_channel="apparent",
        lags=lags,
        lag_seconds=tuple(lag_seconds),
        chord_displacement=tuple(chords),
        path_displacement=tuple(paths),
        straightness=tuple(straightness),
        pair_counts=tuple(counts),
        fit_rmse=None if fit is None else fit[1],
        independently_calibrated=False,
    )


def resolve_time_normalization_exponent(
    components: dict[str, bool],
    config: AuditConfig,
    evidence: LagScalingEvidence | None,
) -> tuple[float, str]:
    """The exponent actually applied, and where it came from."""
    if not components["time_normalization"] or config.ablation == AuditAblation.WITHOUT_TIME_NORMALIZATION:
        return 0.0, "time_normalization_disabled"
    if config.ablation == AuditAblation.WITHOUT_LAG_SCALING:
        return 1.0, "ballistic_default_without_lag_scaling_ablation"
    if config.lag_exponent is not None:
        value = float(config.lag_exponent)
        if not math.isfinite(value):
            raise ValueError("lag_exponent must be finite")
        if not config.lag_exponent_source:
            raise ValueError("an explicit lag_exponent requires lag_exponent_source")
        return value, config.lag_exponent_source
    if not config.lag_scaling_enabled or config.lag_exponent_mode == LagExponentMode.BALLISTIC:
        return 1.0, "lag_scaling_disabled_ballistic_default" if not config.lag_scaling_enabled else "ballistic_mode"
    if config.lag_exponent_mode == LagExponentMode.MEASURED:
        if evidence is not None and evidence.exponent is not None:
            return float(evidence.exponent), "measured_within_clip_power_law"
        return 1.0, "ballistic_default_no_measurement"
    value = float(config.default_lag_exponent)
    if not math.isfinite(value):
        raise ValueError("default_lag_exponent must be finite")
    return value, config.default_lag_exponent_source


def _invalid_transition(frame_a: Any, frame_b: Any, frame_shape: tuple[int, int], reason: str) -> TransitionEvidence:
    return TransitionEvidence(
        source_frame_index_a=frame_a.source_frame_index,
        source_frame_index_b=frame_b.source_frame_index,
        timestamp_a=frame_a.timestamp_seconds,
        timestamp_b=frame_b.timestamp_seconds,
        dt_seconds=frame_b.timestamp_seconds - frame_a.timestamp_seconds,
        frame_shape=frame_shape,
        valid=False,
        invalid_reason=reason,
    )


def _route_evidence(
    target: PromptTargetDecision,
    apparent: Any,
    camera: Any,
    residual: Any,
    *,
    include_persistence: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    aggregation = (
        "structured_intensity_and_temporal_coverage"
        if include_persistence
        else "intensity_only_without_persistence_ablation"
    )

    def single_channel(channel: Any) -> dict[str, Any]:
        evidence = channel.to_dict()
        evidence["evidence_components"] = (
            ["motion_intensity", "temporal_coverage"]
            if include_persistence
            else ["motion_intensity"]
        )
        evidence["aggregation"] = aggregation
        return evidence

    if target.target == MotionTarget.SUBJECT.value:
        return "residual", single_channel(residual)
    if target.target == MotionTarget.CAMERA.value:
        return "camera", single_channel(camera)
    if target.target == MotionTarget.GENERIC.value:
        return "apparent", single_channel(apparent)
    if target.target == MotionTarget.BOTH.value:
        evidence = {
            "camera": single_channel(camera),
            "residual": single_channel(residual),
            "evidence_components": (
                [
                    "camera.motion_intensity",
                    "camera.temporal_coverage",
                    "residual.motion_intensity",
                    "residual.temporal_coverage",
                ]
                if include_persistence
                else ["camera.motion_intensity", "residual.motion_intensity"]
            ),
            "aggregation": aggregation,
            "combination": "structured_not_summed",
        }
        return "camera_and_residual", evidence
    return None, None


def _optional_scalar_score(
    evidence: dict[str, Any] | None,
    config: AuditConfig,
) -> float | None:
    if config.scalar_score_hook is None or evidence is None:
        return None
    if not config.scalar_score_source:
        raise ValueError("scalar_score_hook requires scalar_score_source")
    value = float(config.scalar_score_hook(evidence))
    if not math.isfinite(value):
        raise ValueError("scalar_score_hook must return a finite value")
    return value


def _benchmark_scalar_score(evidence: dict[str, Any] | None) -> float | None:
    """Formal Full-Audit scalar: routed task-relevant motion intensity."""
    if not evidence or "motion_intensity" not in evidence:
        return None
    value = float(evidence["motion_intensity"])
    if not math.isfinite(value):
        raise ValueError("task-relevant motion intensity must be finite")
    return value


def _intensity(
    displacement_diagonals: float | None,
    dt_seconds: float,
    exponent: float,
) -> float | None:
    """The ``*_displacement`` fields are already divided by the image diagonal."""
    if displacement_diagonals is None:
        return None
    if not math.isfinite(dt_seconds) or dt_seconds <= 0:
        return None
    return scale_intensity(float(displacement_diagonals), dt_seconds, exponent)


def analyze_timed_flow_sequence(
    video: str,
    prompt: str,
    sequence: TimedFrameSequence,
    flow_provider: Callable[[np.ndarray, np.ndarray], np.ndarray] | Any,
    *,
    motion_target_override: str | None = None,
    config: AuditConfig | None = None,
) -> AuditVideoResult:
    config = config or AuditConfig()
    components = config.components()
    if config.scalar_score_hook is not None and not config.scalar_score_source:
        raise ValueError("scalar_score_hook requires scalar_score_source")
    target = parse_motion_target(prompt, motion_target_override)
    if not components["task_routing"] or config.ablation == AuditAblation.WITHOUT_PROMPT_ROUTING:
        target = PromptTargetDecision(MotionTarget.GENERIC.value, "ablation_without_prompt_routing")
    transitions: list[TransitionEvidence] = []
    compute_flow = flow_provider.compute_flow if hasattr(flow_provider, "compute_flow") else flow_provider

    for frame_a, frame_b in zip(sequence.frames[:-1], sequence.frames[1:]):
        dt = frame_b.timestamp_seconds - frame_a.timestamp_seconds
        if not math.isfinite(dt):
            transitions.append(_invalid_transition(frame_a, frame_b, sequence.frame_shape, "non_finite_dt"))
            continue
        if dt <= 0:
            transitions.append(_invalid_transition(frame_a, frame_b, sequence.frame_shape, "non_positive_dt"))
            continue
        try:
            flow = np.asarray(compute_flow(frame_a.frame, frame_b.frame), dtype=np.float64)
            if flow.shape[:2] != sequence.frame_shape or flow.ndim != 3 or flow.shape[2] != 2:
                raise ValueError(f"flow shape {flow.shape} does not match frame shape {sequence.frame_shape}")
            decomposition = None
            if components["source_decomposition"]:
                decomposition = decompose_motion(flow, config.affine, compensate_global=config.ablation != AuditAblation.WITHOUT_GLOBAL_COMPENSATION)
                apparent_stats, global_stats, residual_stats = decomposition.apparent_statistics, decomposition.global_statistics, decomposition.residual_statistics
            else:
                apparent_stats, global_stats, residual_stats = motion_statistics(flow), None, None
            apparent_displacement, apparent_speed = normalized_speed(
                apparent_stats.top5_mean, sequence.frame_shape, dt
            )
            if components["source_decomposition"]:
                global_displacement, global_speed = normalized_speed(global_stats.top5_mean, sequence.frame_shape, dt)
                residual_displacement, residual_speed = normalized_speed(residual_stats.top5_mean, sequence.frame_shape, dt)
            else:
                global_displacement = global_speed = residual_displacement = residual_speed = None
            if not components["time_normalization"] or config.ablation == AuditAblation.WITHOUT_TIME_NORMALIZATION:
                apparent_speed = apparent_displacement
                if components["source_decomposition"]:
                    global_speed, residual_speed = global_displacement, residual_displacement
            transitions.append(
                TransitionEvidence(
                    source_frame_index_a=frame_a.source_frame_index,
                    source_frame_index_b=frame_b.source_frame_index,
                    timestamp_a=frame_a.timestamp_seconds,
                    timestamp_b=frame_b.timestamp_seconds,
                    dt_seconds=dt,
                    frame_shape=sequence.frame_shape,
                    valid=True,
                    invalid_reason=None,
                    apparent_displacement=apparent_displacement,
                    global_displacement=global_displacement,
                    residual_displacement=residual_displacement,
                    apparent_speed=apparent_speed,
                    global_speed=global_speed,
                    residual_speed=residual_speed,
                    apparent_magnitude_statistics=apparent_stats,
                    global_magnitude_statistics=global_stats,
                    residual_magnitude_statistics=residual_stats,
                    affine=None if decomposition is None else decomposition.affine,
                )
            )
        except ValueError as exc:
            transitions.append(_invalid_transition(frame_a, frame_b, sequence.frame_shape, f"{type(exc).__name__}: {exc}"))

    lag_evidence = (
        _measure_lag_scaling(sequence, transitions, compute_flow, config)
        if config.lag_scaling_enabled
        and components["time_normalization"]
        and config.ablation
        not in (AuditAblation.WITHOUT_TIME_NORMALIZATION, AuditAblation.WITHOUT_LAG_SCALING)
        else None
    )
    exponent, exponent_source = resolve_time_normalization_exponent(components, config, lag_evidence)
    # The threshold and the significance flags live in the same normalisation as
    # the intensity, so the static/moving decision does not drift with the frame
    # rate the way it does when a pixel threshold meets a per-frame displacement.
    threshold = derive_motion_threshold(sequence.frame_shape, config, exponent)
    threshold_value = threshold.value

    def _significant(displacement: float | None, dt: float) -> bool | None:
        value = _intensity(displacement, dt, exponent)
        return None if value is None else value > threshold_value

    transitions = [
        replace(
            transition,
            apparent_intensity=_intensity(transition.apparent_displacement, transition.dt_seconds, exponent),
            global_intensity=_intensity(transition.global_displacement, transition.dt_seconds, exponent),
            residual_intensity=_intensity(transition.residual_displacement, transition.dt_seconds, exponent),
            significant_motion_threshold=threshold_value,
            apparent_significant=_significant(transition.apparent_displacement, transition.dt_seconds),
            global_significant=_significant(transition.global_displacement, transition.dt_seconds),
            residual_significant=_significant(transition.residual_displacement, transition.dt_seconds),
        )
        for transition in transitions
    ]

    coverage_threshold = threshold.value if components["duration_persistence"] else None
    apparent = aggregate_channel(
        transitions, "apparent", coverage_threshold,
        duration_weighted=components["duration_persistence"], intensity_field="apparent_intensity",
    )
    camera = aggregate_channel(
        transitions, "camera", coverage_threshold,
        duration_weighted=components["duration_persistence"], intensity_field="global_intensity",
    )
    residual = aggregate_channel(
        transitions, "residual", coverage_threshold,
        duration_weighted=components["duration_persistence"], intensity_field="residual_intensity",
    )
    if config.ablation == AuditAblation.WITHOUT_CONTINUOUS_PERSISTENCE_AGGREGATION:
        from ..schemas import ChannelEvidence

        apparent = ChannelEvidence("apparent", apparent.motion_intensity, None, apparent.valid_transition_count, apparent.valid_duration)
        camera = ChannelEvidence("camera", camera.motion_intensity, None, camera.valid_transition_count, camera.valid_duration)
        residual = ChannelEvidence("residual", residual.motion_intensity, None, residual.valid_transition_count, residual.valid_duration)
    include_persistence = config.ablation != AuditAblation.WITHOUT_CONTINUOUS_PERSISTENCE_AGGREGATION
    selected_channel, task_evidence = _route_evidence(
        target,
        apparent,
        camera,
        residual,
        include_persistence=include_persistence,
    )
    score = None
    scalar_score_source = None
    sampled_count = len(sequence.frames)
    official_count = official_count_num(sampled_count)
    effective_count = (
        audit_effective_count_num(sampled_count)
        if config.boundary_fix_enabled
        else (official_count if sampled_count > 1 else None)
    )

    if apparent.valid_transition_count == 0:
        status = "insufficient_transitions"
        failure_reason = "insufficient_transitions"
        score = None
    elif target.target == MotionTarget.UNKNOWN.value:
        status = "unresolved"
        failure_reason = "unknown_motion_target"
        score = None
    else:
        status = "succeeded"
        failure_reason = None
        score = _benchmark_scalar_score(task_evidence)
        scalar_score_source = (
            "task_relevant_motion_evidence.motion_intensity"
            if score is not None
            else None
        )
        if config.scalar_score_hook is not None:
            score = _optional_scalar_score(task_evidence, config)
            scalar_score_source = config.scalar_score_source
    return AuditVideoResult(
        video=video,
        prompt=prompt,
        status=status,
        failure_reason=failure_reason,
        score=score,
        target_decision=target,
        sampled_source_frame_indices=tuple(frame.source_frame_index for frame in sequence.frames),
        timestamps=tuple(frame.timestamp_seconds for frame in sequence.frames),
        timestamp_source=sequence.timestamp_source,
        source_fps=sequence.source_fps,
        sampling_interval=sequence.sampling_interval,
        frame_shape=sequence.frame_shape,
        transitions=tuple(transitions),
        threshold=threshold,
        apparent=apparent,
        camera=camera,
        residual=residual,
        selected_evidence_channel=selected_channel,
        task_relevant_motion_evidence=task_evidence,
        scalar_score_source=(
            scalar_score_source
        ),
        scalar_score_independently_calibrated=(
            config.scalar_score_independently_calibrated
            if config.scalar_score_hook is not None
            else False
        ),
        official_count_num=official_count,
        audit_effective_count_num=effective_count,
        boundary_fix_applied=config.boundary_fix_enabled and sampled_count > 1 and official_count == 0,
        component_provenance=components,
        lag_scaling=lag_evidence,
        time_normalization_exponent=exponent,
        time_normalization_exponent_source=exponent_source,
        time_normalization_exponent_independently_calibrated=(
            config.lag_exponent is not None
            and config.lag_exponent_source is not None
            and config.lag_exponent_independently_calibrated
        ),
    )


def audit_result_payload(result: AuditVideoResult, level: DiagnosticsLevel) -> dict[str, Any]:
    return {
        "video": result.video,
        "prompt": result.prompt,
        "backend": "audit",
        "score": result.score,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "error": None,
        "prompt_motion_target": result.target_decision.target,
        "selected_evidence_channel": result.selected_evidence_channel,
        "apparent_intensity": result.apparent.motion_intensity,
        "camera_intensity": result.camera.motion_intensity,
        "residual_intensity": result.residual.motion_intensity,
        "apparent_coverage": result.apparent.temporal_coverage,
        "camera_coverage": result.camera.temporal_coverage,
        "residual_coverage": result.residual.temporal_coverage,
        "task_relevant_motion_evidence": result.task_relevant_motion_evidence,
        "score_metadata": {
            "source": result.scalar_score_source,
            "independently_calibrated": result.scalar_score_independently_calibrated,
            "status": (
                "succeeded_scalarized"
                if result.score is not None
                else ("configured_but_not_applicable" if result.scalar_score_source else "not_available")
            ),
            "scalarization": "task_relevant_motion_evidence.motion_intensity",
        },
        "diagnostics": serialize_audit_result(result, level),
    }
