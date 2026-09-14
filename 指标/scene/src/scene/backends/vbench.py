from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

UPSTREAM_PATH = Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", "/root/vbench1"))
UPSTREAM_REMOTE = "https://github.com/msy625/VBench.git"
UPSTREAM_BRANCH = "master"
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"
OFFLINE_BUNDLE_ORIGIN = "/root/vbench1.bundle"
DEFAULT_TAG2TEXT_CONFIG = {
    "pretrained": "caption_model/tag2text_swin_14m.pth",
    "image_size": 384,
    "vit": "swin_b",
}


def upstream_path() -> Path:
    return Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()


def verify_upstream(path: Path | None = None) -> dict[str, Any]:
    path = path or upstream_path()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()

    remote = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current")
    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    origin_ok = remote in {UPSTREAM_REMOTE, OFFLINE_BUNDLE_ORIGIN}
    if not (origin_ok and branch == UPSTREAM_BRANCH and sha == UPSTREAM_SHA):
        raise RuntimeError(f"upstream identity mismatch: {(remote, branch, sha)}")
    if dirty:
        raise RuntimeError("upstream worktree is dirty; refusing official evaluation")
    return {"path": str(path.resolve()), "remote": remote, "branch": branch, "sha": sha, "dirty": dirty}


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
        resolved = str(self.upstream.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        self.module = importlib.import_module("vbench.scene") if compute is None else None
        self.compute = compute
        config = dict(DEFAULT_TAG2TEXT_CONFIG)
        config.update(tag2text_config or {})
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
        with tempfile.TemporaryDirectory(prefix="scene-official-") as root:
            path = Path(root) / "input.json"
            path.write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")
            compute_scene = self.compute or self.module.compute_scene
            raw = compute_scene(str(path), self.device, self.config)
        return raw[1][0]
