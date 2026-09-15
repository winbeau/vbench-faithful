from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class OrderedRelationQuery:
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class Detection:
    detection_id: int
    label: str
    box: Box
    confidence: float | None = None

    def __post_init__(self) -> None:
        if len(self.box) != 4:
            raise ValueError("box must contain x0, y0, x1, y1")
        x0, y0, x1, y1 = self.box
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid non-positive box: {self.box}")


@dataclass(frozen=True)
class RoleBoundCandidates:
    query: OrderedRelationQuery
    subject_candidates: tuple[Detection, ...]
    object_candidates: tuple[Detection, ...]
    frame_index: int


@dataclass(frozen=True)
class GeometryEvidence:
    dx: float
    dy: float
    iou: float
    axis_dominance: bool
    direction_match: bool
    score: float


@dataclass(frozen=True)
class FrameAuditResult:
    frame_index: int
    query: OrderedRelationQuery
    detections: tuple[Detection, ...]
    subject_candidate_ids: tuple[int, ...]
    object_candidate_ids: tuple[int, ...]
    assigned_subject_id: int | None
    assigned_object_id: int | None
    selected_subject_box: Box | None
    selected_object_box: Box | None
    geometry: GeometryEvidence | None
    frame_score: float
    frame_reason: str
    role_identity_ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoAuditResult:
    video: str
    prompt: str
    query: OrderedRelationQuery
    sampled_frame_indices: tuple[int, ...]
    frame_results: tuple[FrameAuditResult, ...]
    video_score: float
    aggregation_method: str = "mean_over_16_officially_sampled_frames"
    scored_frame_count: int | None = None

    @property
    def valid_frame_count(self) -> int:
        return sum(result.geometry is not None for result in self.frame_results)

    @property
    def detected_frame_count(self) -> int:
        """Frames where the detector returned a candidate for both roles.

        This is the denominator of the detection-conditioned aggregation: it
        removes detector drop-outs instead of scoring them as relation
        violations, so an abstention is no longer indistinguishable from a
        wrong direction.
        """
        return sum(
            bool(result.subject_candidate_ids) and bool(result.object_candidate_ids)
            for result in self.frame_results
        )

    @property
    def missing_subject_count(self) -> int:
        return sum(result.frame_reason == "missing_subject" for result in self.frame_results)

    @property
    def missing_object_count(self) -> int:
        return sum(result.frame_reason == "missing_object" for result in self.frame_results)

    @property
    def ambiguous_role_count(self) -> int:
        return sum(result.role_identity_ambiguous for result in self.frame_results)

    def to_dict(self, include_frames: bool = True) -> dict[str, Any]:
        result = {
            "video": self.video,
            "prompt": self.prompt,
            "subject": self.query.subject,
            "relation": self.query.relation,
            "object": self.query.object,
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "frame_scores": [frame.frame_score for frame in self.frame_results],
            "valid_frame_count": self.valid_frame_count,
            "detected_frame_count": self.detected_frame_count,
            "scored_frame_count": self.scored_frame_count,
            "missing_subject_count": self.missing_subject_count,
            "missing_object_count": self.missing_object_count,
            "ambiguous_role_count": self.ambiguous_role_count,
            "aggregation_method": self.aggregation_method,
            "video_score": self.video_score,
        }
        if include_frames:
            result["frame_results"] = [frame.to_dict() for frame in self.frame_results]
        return result
