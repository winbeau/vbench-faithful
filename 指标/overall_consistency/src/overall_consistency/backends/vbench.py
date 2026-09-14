from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

UPSTREAM_PATH = Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", "/root/vbench1"))
UPSTREAM_REMOTE = "https://github.com/msy625/VBench.git"
UPSTREAM_BRANCH = "master"
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"
OFFLINE_BUNDLE_ORIGIN = "/root/vbench1.bundle"
DEFAULT_CHECKPOINT = Path.home() / ".cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth"
NUM_FRAMES = 8
SAMPLE_MODE = "middle"
INPUT_SIZE = 224


def upstream_path() -> Path:
    return Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()


def checkpoint_path() -> Path:
    return Path(os.environ.get("VBENCH_AUDIT_VICLIP_WEIGHT", str(DEFAULT_CHECKPOINT))).expanduser()


def inspect_upstream(path: Path | None = None) -> dict[str, Any]:
    path = path or upstream_path()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()

    return {
        "path": str(path.resolve()),
        "remote": git("remote", "get-url", "origin"),
        "branch": git("branch", "--show-current"),
        "sha": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def verify_upstream(path: Path | None = None) -> dict[str, Any]:
    state = inspect_upstream(path)
    origin_ok = state["remote"] in {UPSTREAM_REMOTE, OFFLINE_BUNDLE_ORIGIN}
    if not (origin_ok and state["branch"] == UPSTREAM_BRANCH and state["sha"] == UPSTREAM_SHA):
        raise RuntimeError(f"upstream identity mismatch: {state}")
    if state["dirty"]:
        raise RuntimeError("upstream worktree is dirty; refusing official evaluation")
    return state


def import_official_module(path: Path | None = None) -> tuple[Any, dict[str, Any]]:
    path = path or upstream_path()
    state = verify_upstream(path)
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.overall_consistency")
    module_path = Path(module.__file__).resolve()
    if path.resolve() not in module_path.parents:
        raise RuntimeError(f"wrong vbench import: {module_path}")
    return module, state


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
