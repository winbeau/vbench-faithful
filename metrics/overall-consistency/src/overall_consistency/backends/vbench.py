from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from vbench_audit_core.upstream import (
    UPSTREAM_SHA,
    import_official_module as _import_official_module,
    inspect_upstream as _inspect_upstream,
    resolve_upstream_path,
    verify_upstream as _verify_upstream,
)

UPSTREAM_PATH = None
DEFAULT_CHECKPOINT = Path.home() / ".cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth"
NUM_FRAMES = 8
SAMPLE_MODE = "middle"
INPUT_SIZE = 224


def upstream_path() -> Path:
    return resolve_upstream_path()


def checkpoint_path() -> Path:
    import os
    return Path(os.environ.get("VBENCH_AUDIT_VICLIP_WEIGHT", str(DEFAULT_CHECKPOINT))).expanduser()


def inspect_upstream(path: Path | None = None) -> dict[str, Any]:
    return _inspect_upstream(path).__dict__


def verify_upstream(path: Path | None = None) -> dict[str, Any]:
    return _verify_upstream(path, dimension="overall_consistency").__dict__


def import_official_module(path: Path | None = None) -> tuple[Any, dict[str, Any]]:
    module, state = _import_official_module("overall_consistency", path)
    return module, state.__dict__


def sample_middle_indices(num_frames: int, video_length: int) -> list[int]:
    if num_frames <= 0 or video_length <= 0:
        raise ValueError("num_frames and video_length must be positive")
    count = min(num_frames, video_length)
    intervals = [int(index * video_length / count) for index in range(count + 1)]
    indices = [(intervals[index] + intervals[index + 1] - 1) // 2 for index in range(count)]
    indices.extend([indices[-1]] * (num_frames - len(indices)))
    return indices


def official_entry(video: Path, metadata_item: Mapping[str, Any]) -> dict[str, Any]:
    prompt = metadata_item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("overall consistency requires a non-empty prompt")
    return {"prompt": prompt, "video_list": [str(video.resolve())]}


class OfficialEvaluator:
    """Execute the locked `overall_consistency()` function without approximating it."""

    def __init__(self, device: Any, checkpoint: Path, upstream: Path | None = None, module: Any | None = None, encoder: Any | None = None):
        self.device = device
        self.module, self.state = (module, None) if module is not None else import_official_module(upstream)
        if encoder is None:
            from ..models import LockedViCLIPEncoder

            encoder = LockedViCLIPEncoder(self.module, device, checkpoint)
        self.encoder = encoder

    def evaluate_video(self, video: Path, metadata_item: Mapping[str, Any]) -> float:
        entry = official_entry(video, metadata_item)
        dataset_score, video_results = self.module.overall_consistency(
            self.encoder.model, [entry], self.encoder.tokenizer, self.device, sample=SAMPLE_MODE
        )
        score = float(video_results[0]["video_results"])
        if score != score or score in (float("inf"), float("-inf")):
            raise ValueError("official evaluator returned a non-finite score")
        if float(dataset_score) != score:
            raise RuntimeError("single-video official dataset score differs from video score")
        return score
