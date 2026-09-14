from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from vbench_audit_core.upstream import (
    UPSTREAM_SHA,
    import_official_module as _import_official_module,
    resolve_upstream_path,
    verify_upstream as _verify_upstream,
)

UPSTREAM_PATH = None
DEFAULT_TAG2TEXT_CONFIG = {
    "pretrained": "caption_model/tag2text_swin_14m.pth",
    "image_size": 384,
    "vit": "swin_b",
}


def upstream_path() -> Path:
    return resolve_upstream_path()


def verify_upstream(path: Path | None = None) -> dict[str, Any]:
    state = _verify_upstream(path, dimension="scene")
    return state.__dict__


def extract_scene_label(metadata_item: Mapping[str, Any]) -> str:
    """Read the official auxiliary_info.scene label, accepting its nested shape."""
    value: Any = metadata_item.get("dimension_metadata", metadata_item.get("auxiliary_info"))
    if isinstance(value, Mapping) and "scene" in value:
        value = value["scene"]
    while isinstance(value, Mapping) and "scene" in value:
        value = value["scene"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scene metadata must contain a non-empty scene label")
    return value.strip()


def sample_middle_indices(num_frames: int, video_length: int) -> list[int]:
    if num_frames <= 0 or video_length <= 0:
        raise ValueError("num_frames and video_length must be positive")
    count = min(num_frames, video_length)
    # Match np.linspace(0, video_length, count + 1).astype(int), including its
    # floating-step rounding behavior before truncation.
    step = video_length / count
    intervals = [int(i * step) for i in range(count + 1)]
    intervals[-1] = video_length
    indices = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(count)]
    if len(indices) < num_frames:
        indices.extend([indices[-1]] * (num_frames - len(indices)))
    return indices


def official_frame_match(scene_label: str, caption: str) -> bool:
    return all(token in caption for token in scene_label.split(" "))


def video_score(frame_matches: list[bool | int | float]) -> float:
    if not frame_matches:
        raise ValueError("cannot score an empty frame sequence")
    # Official inputs are booleans; repaired modes provide continuous frame
    # support. Numeric averaging preserves both contracts.
    return sum(float(value) for value in frame_matches) / len(frame_matches)


def dataset_score(video_results: list[Mapping[str, Any]]) -> float:
    frame_count = sum(int(item.get("frame_count", len(item.get("frame_results", [])))) for item in video_results)
    success_count = sum(int(item.get("success_frame_count", sum(bool(v) for v in item.get("frame_results", [])))) for item in video_results)
    if frame_count <= 0:
        raise ValueError("cannot aggregate a dataset without frames")
    return success_count / frame_count


def official_metadata_entry(video: Path, metadata_item: Mapping[str, Any]) -> dict[str, Any]:
    label = extract_scene_label(metadata_item)
    return {
        "prompt_en": metadata_item.get("prompt", ""),
        "dimension": ["scene"],
        "video_list": [str(video.resolve())],
        "auxiliary_info": {"scene": {"scene": {"scene": label}}},
    }


class OfficialVBenchSceneEvaluator:
    """Thin wrapper around the locked VBench 1.0 Scene evaluator."""

    def __init__(
        self,
        device: Any,
        tag2text_config: Mapping[str, Any] | None = None,
        upstream: Path | None = None,
        compute: Callable[..., Any] | None = None,
    ):
        self.device = device
        self.upstream = upstream or upstream_path()
        self.state = verify_upstream(self.upstream)
        self.module = _import_official_module("scene", self.upstream)[0] if compute is None else None
        self.compute = compute
        config = dict(DEFAULT_TAG2TEXT_CONFIG)
        config.update(tag2text_config or {})
        import os
        explicit_weight = os.environ.get("VBENCH_AUDIT_TAG2TEXT_WEIGHT")
        cache_dir = os.environ.get("VBENCH_CACHE_DIR", str(Path.home() / ".cache" / "vbench"))
        if explicit_weight:
            config["pretrained"] = explicit_weight
        elif config["pretrained"] == DEFAULT_TAG2TEXT_CONFIG["pretrained"]:
            config["pretrained"] = str(Path(cache_dir) / config["pretrained"])
        self.config = config
        self.model = None

    def _model(self) -> Any:
        if self.model is None:
            from vbench.third_party.tag2Text.tag2text import tag2text_caption

            self.model = tag2text_caption(**self.config).to(self.device)
            self.model.eval()
        return self.model

    def evaluate_video(self, video: Path, metadata_item: Mapping[str, Any]) -> Mapping[str, Any]:
        entry = official_metadata_entry(video, metadata_item)
        compute_scene = self.compute
        if compute_scene is None:
            # compute_scene constructs a model on every call.  Workers reuse
            # one evaluator, so call the official per-video function with the
            # cached model while retaining the exact frame/caption formula.
            label = extract_scene_label(metadata_item)
            direct_entry = {
                "prompt": metadata_item.get("prompt", ""),
                "video_list": [str(video.resolve())],
                "auxiliary_info": {"scene": {"scene": label}},
            }
            score, rows = self.module.scene(self._model(), [direct_entry], self.device)
            return rows[0]
        with tempfile.TemporaryDirectory(prefix="scene-official-") as root:
            path = Path(root) / "input.json"
            path.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
            raw = compute_scene(str(path), self.device, self.config)
        return raw[1][0]
