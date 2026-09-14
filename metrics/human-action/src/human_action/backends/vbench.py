from __future__ import annotations

from pathlib import Path
from typing import Any

from vbench_audit_core.upstream import (
    import_official_module as _import_official_module,
    inspect_upstream as _inspect_upstream,
    verify_upstream as _verify_upstream,
)
from ..models import LockedUmtClassifier, OFFICIAL_NUM_FRAMES, OFFICIAL_THRESHOLD, OFFICIAL_TOP_K
from ..schemas import OfficialActionResult

DEFAULT_WEIGHT = Path.home() / ".cache/vbench/umt/l16_ptk710_ftk710_ftk400_f16_res224.pth"

def inspect_upstream(path: Path | None = None):
    return _inspect_upstream(path)


def verify_upstream(path: Path | None = None):
    return _verify_upstream(path, dimension="human_action")


def import_official_module(path: Path | None = None):
    return _import_official_module("human_action", path)


def official_target_from_filename(video: str | Path) -> str:
    return str(video).split("/")[-1].lower().split("-")[0].split("person is ")[-1].split("_")[0]


def official_decision(
    probabilities: Any,
    categories: tuple[str, ...],
    target_action: str,
) -> tuple[bool, tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    import torch

    values = probabilities if torch.is_tensor(probabilities) else torch.as_tensor(probabilities)
    if tuple(values.shape) != (400,):
        raise ValueError(f"expected 400 Kinetics probabilities, got {tuple(values.shape)}")
    top_values, top_indices = torch.topk(
        values.unsqueeze(0), OFFICIAL_TOP_K, dim=1
    )
    indices = top_indices.squeeze(0).tolist()
    rounded = tuple(round(float(value), 4) for value in top_values.squeeze(0).tolist())
    top_actions = tuple(categories[int(index)] for index in indices)
    accepted = tuple(action for action, value in zip(top_actions, rounded) if value >= OFFICIAL_THRESHOLD)
    return target_action in accepted, top_actions, rounded, accepted


class OfficialHumanActionEvaluator:
    """Per-video plumbing around the locked Official UMT behavior."""

    def __init__(
        self,
        device: Any,
        model_weight: Path,
        upstream_path: Path | None = None,
        *,
        classifier: Any | None = None,
    ):
        self.module, self.upstream_state = import_official_module(upstream_path)
        self.classifier = classifier or LockedUmtClassifier(device, model_weight, self.module)

    def evaluate_video(self, video: Path) -> OfficialActionResult:
        if hasattr(self.classifier, "predict_official_video_tensor"):
            probabilities = self.classifier.predict_official_video_tensor(video)
        else:
            probabilities = self.classifier.predict_official_video(video)
        target = official_target_from_filename(video)
        matched, actions, values, accepted = official_decision(
            probabilities, self.classifier.categories, target
        )
        return OfficialActionResult(
            video=str(video),
            target_action=target,
            target_source="official_filename_parser",
            sampled_frame_count=OFFICIAL_NUM_FRAMES,
            rounded_top5_actions=actions,
            rounded_top5_probabilities=values,
            accepted_actions=accepted,
            threshold=OFFICIAL_THRESHOLD,
            matched=matched,
        )
