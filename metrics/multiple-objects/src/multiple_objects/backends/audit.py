from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..conditions import parse_target_objects
from ..metric import aggregate_video, frame_object_evidence, official_frame_decision
from ..schemas import FrameDetections, FrameEvidence, MultipleObjectsConfig, frame_evidence_to_dict


def evaluate_detections(
    video: str | Path,
    metadata: dict[str, Any],
    frame_detections: list[list[Any] | FrameDetections],
    config: MultipleObjectsConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MultipleObjectsConfig()
    targets = parse_target_objects(metadata)
    frames: list[FrameEvidence] = []
    for index, frame in enumerate(frame_detections):
        if isinstance(frame, FrameDetections):
            detections = frame.detections
            official_labels: tuple[str, ...] | None = frame.official_labels
            decision_source = "separate_official_threshold_inference"
        else:
            detections = frame
            official_labels = None
            decision_source = "model_free_score_proxy"
        matched, confidences, score, weakest, weakest_confidence = frame_object_evidence(targets, detections, cfg)
        frames.append(
            FrameEvidence(
                index, targets, matched, confidences,
                official_frame_decision(
                    targets, detections, cfg.official_threshold,
                    official_labels=official_labels,
                ),
                score, weakest, weakest_confidence,
                tuple(official_labels or ()), decision_source,
            )
        )
    video_score = aggregate_video(frame.repaired_frame_score for frame in frames)
    return {
        "video": str(video),
        "prompt": str(metadata.get("prompt", "")),
        "backend": "audit",
        "score": video_score,
        "status": "succeeded",
        "diagnostics": {
            "target_objects": list(targets),
            "frames": [
                {
                    **frame_evidence_to_dict(frame),
                    "official_threshold": cfg.official_threshold,
                    "repair_candidate_threshold": cfg.repair_candidate_threshold,
                    "threshold_margin": {
                        target: frame.per_target_official_threshold_score[target]
                        - cfg.official_threshold
                        for target in targets
                    },
                    "threshold_margin_basis": "official_pre_threshold_roi_score",
                    "evidence_score_source": "official_grit_roi_threshold_score",
                    "aggregation_mode": cfg.aggregation_mode,
                    "softmin_beta": cfg.softmin_beta,
                }
                for frame in frames
            ],
            "final_video_score": video_score,
            "frame_score_sum": sum(frame.repaired_frame_score for frame in frames),
            "frame_count": len(frames),
            "aggregation": "arithmetic_mean_of_frame_scores",
            "official_threshold": cfg.official_threshold,
            "repair_candidate_threshold": cfg.repair_candidate_threshold,
            "evidence_score_source": "official_grit_roi_threshold_score",
            "aggregation_mode": cfg.aggregation_mode,
            "softmin_beta": cfg.softmin_beta,
        },
    }


def evaluate_audit_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    config: MultipleObjectsConfig,
) -> list[dict[str, Any]]:
    from .vbench import OfficialGrITDetector

    detector = OfficialGrITDetector(
        device,
        model_weight,
        repair_candidate_threshold=config.repair_candidate_threshold,
    )
    return [
        evaluate_detections(video, dict(metadata[video.name]), detector.detect_video(video), config)
        for video in videos
    ]
