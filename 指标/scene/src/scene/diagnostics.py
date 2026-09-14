from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class FrameDiagnostic:
    frame_index: int
    global_score: float
    regional_scores: list[float]
    environment_mean: float
    final_frame_score: float
    scene_label: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["frame_index"] = int(self.frame_index)
        data["global_score"] = float(self.global_score)
        data["regional_scores"] = [float(value) for value in self.regional_scores]
        data["environment_mean"] = float(self.environment_mean)
        data["final_frame_score"] = float(self.final_frame_score)
        return data


def diagnostics_payload(frame_diagnostics: list[FrameDiagnostic], sampled_frame_indices: list[int]) -> dict[str, Any]:
    return {
        "sampled_frame_indices": [int(index) for index in sampled_frame_indices],
        "frame_diagnostics": [item.to_dict() for item in frame_diagnostics],
        "frame_scores": [item.final_frame_score for item in frame_diagnostics],
    }
