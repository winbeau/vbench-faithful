from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TimedFrame:
    frame: Any = field(repr=False, compare=False)
    timestamp_seconds: float
    source_frame_index: int


@dataclass(frozen=True)
class TimedFrameSequence:
    frames: tuple[TimedFrame, ...]
    frame_shape: tuple[int, int]
    source_fps: float | None
    sampling_interval: int
    timestamp_source: str


@dataclass(frozen=True)
class MotionStatistics:
    top5_mean: float
    mean: float
    median: float
    maximum: float


@dataclass(frozen=True)
class AffineFitDiagnostics:
    affine_parameters: tuple[tuple[float, float, float], tuple[float, float, float]]
    inlier_ratio: float
    candidate_count: int
    initial_candidate_count: int
    candidate_strategy: str
    fallback: bool
    fallback_reason: str | None
    residual_median: float
    residual_mad: float


@dataclass(frozen=True)
class MotionDecomposition:
    apparent_flow: Any = field(repr=False, compare=False)
    global_flow: Any = field(repr=False, compare=False)
    residual_flow: Any = field(repr=False, compare=False)
    apparent_statistics: MotionStatistics
    global_statistics: MotionStatistics
    residual_statistics: MotionStatistics
    affine: AffineFitDiagnostics


@dataclass(frozen=True)
class MotionThreshold:
    value: float
    units: str
    source: str
    independently_calibrated: bool


@dataclass(frozen=True)
class TransitionEvidence:
    source_frame_index_a: int
    source_frame_index_b: int
    timestamp_a: float
    timestamp_b: float
    dt_seconds: float
    frame_shape: tuple[int, int]
    valid: bool
    invalid_reason: str | None
    apparent_displacement: float | None = None
    global_displacement: float | None = None
    residual_displacement: float | None = None
    apparent_speed: float | None = None
    global_speed: float | None = None
    residual_speed: float | None = None
    significant_motion_threshold: float | None = None
    apparent_significant: bool | None = None
    global_significant: bool | None = None
    residual_significant: bool | None = None
    apparent_magnitude_statistics: MotionStatistics | None = None
    global_magnitude_statistics: MotionStatistics | None = None
    residual_magnitude_statistics: MotionStatistics | None = None
    affine: AffineFitDiagnostics | None = None

    def to_dict(self, include_affine: bool = True) -> dict[str, Any]:
        result = {
            "source_frame_index_a": self.source_frame_index_a,
            "source_frame_index_b": self.source_frame_index_b,
            "timestamp_a": self.timestamp_a,
            "timestamp_b": self.timestamp_b,
            "dt_seconds": self.dt_seconds,
            "frame_shape": list(self.frame_shape),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "apparent_displacement": self.apparent_displacement,
            "global_displacement": self.global_displacement,
            "residual_displacement": self.residual_displacement,
            "apparent_speed": self.apparent_speed,
            "global_speed": self.global_speed,
            "residual_speed": self.residual_speed,
            "significant_motion_threshold": self.significant_motion_threshold,
            "moving_static_decision": {
                "apparent": self.apparent_significant,
                "camera": self.global_significant,
                "residual": self.residual_significant,
            },
        }
        if self.apparent_magnitude_statistics is not None:
            result["magnitude_statistics"] = {
                "apparent": asdict(self.apparent_magnitude_statistics),
                "camera": asdict(self.global_magnitude_statistics),
                "residual": asdict(self.residual_magnitude_statistics),
            }
        if include_affine and self.affine is not None:
            result["global_estimation"] = {
                "affine_parameters": [list(row) for row in self.affine.affine_parameters],
                "inlier_ratio": self.affine.inlier_ratio,
                "candidate_count": self.affine.candidate_count,
                "initial_candidate_count": self.affine.initial_candidate_count,
                "candidate_strategy": self.affine.candidate_strategy,
                "fallback": self.affine.fallback,
                "fallback_reason": self.affine.fallback_reason,
                "residual_median": self.affine.residual_median,
                "residual_mad": self.affine.residual_mad,
            }
        return result


@dataclass(frozen=True)
class ChannelEvidence:
    channel: str
    motion_intensity: float | None
    temporal_coverage: float | None
    valid_transition_count: int
    valid_duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "motion_intensity": self.motion_intensity,
            "temporal_coverage": self.temporal_coverage,
            "valid_transition_count": self.valid_transition_count,
            "valid_duration": self.valid_duration,
        }


@dataclass(frozen=True)
class PromptTargetDecision:
    target: str
    source: str
    matched_subject_cues: tuple[str, ...] = ()
    matched_camera_cues: tuple[str, ...] = ()
    matched_generic_cues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "source": self.source,
            "matched_subject_cues": list(self.matched_subject_cues),
            "matched_camera_cues": list(self.matched_camera_cues),
            "matched_generic_cues": list(self.matched_generic_cues),
        }


@dataclass(frozen=True)
class AuditVideoResult:
    video: str
    prompt: str
    status: str
    failure_reason: str | None
    score: float | None
    target_decision: PromptTargetDecision
    sampled_source_frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    timestamp_source: str
    source_fps: float | None
    sampling_interval: int
    frame_shape: tuple[int, int]
    transitions: tuple[TransitionEvidence, ...]
    threshold: MotionThreshold
    apparent: ChannelEvidence
    camera: ChannelEvidence
    residual: ChannelEvidence
    selected_evidence_channel: str | None
    task_relevant_motion_evidence: dict[str, Any] | None
    scalar_score_source: str | None
    scalar_score_independently_calibrated: bool
    official_count_num: int
    audit_effective_count_num: int | None
    boundary_fix_applied: bool


@dataclass(frozen=True)
class OfficialVideoResult:
    video: str
    sampled_source_frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    timestamp_source: str
    source_fps: float
    sampling_interval: int
    frame_shape: tuple[int, int]
    sampled_frame_count: int
    raw_flow_top5_mean: tuple[float, ...]
    official_threshold: float
    official_moving_flags: tuple[bool, ...]
    official_moving_count: int
    official_count_num: int
    official_video_boolean: bool
