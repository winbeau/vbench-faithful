from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

UPSTREAM_PATH = Path("/home/msy625/vbench1")
UPSTREAM_REMOTE = "https://github.com/msy625/VBench.git"
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"
DEFAULT_CONFIG = UPSTREAM_PATH / "vbench/third_party/amt/cfgs/AMT-S.yaml"
DEFAULT_WEIGHT = Path.home() / ".cache/vbench/amt_model/amt-s.pth"


def inspect_upstream(path: Path = UPSTREAM_PATH) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()
    return {"path": str(path.resolve()), "remote": git("remote", "get-url", "origin"), "branch": git("branch", "--show-current"), "sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def import_official_module(path: Path = UPSTREAM_PATH):
    state = inspect_upstream(path)
    if (state["remote"], state["branch"], state["sha"]) != (UPSTREAM_REMOTE, "master", UPSTREAM_SHA):
        raise RuntimeError("upstream VBench identity mismatch")
    if state["dirty"]:
        raise RuntimeError("upstream VBench worktree is dirty")
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.motion_smoothness")
    if path.resolve() not in Path(module.__file__).resolve().parents:
        raise RuntimeError(f"wrong official import: {module.__file__}")
    return module, state


def metadata_to_official_entries(videos: list[Path]) -> list[dict[str, Any]]:
    return [{"prompt_en": "", "dimension": ["motion_smoothness"], "video_list": [str(video.resolve())]} for video in videos]


def evaluate_official_reference(
    videos: list[Path], device: Any, config: Path = DEFAULT_CONFIG, checkpoint: Path = DEFAULT_WEIGHT,
    *, compute: Callable[..., Any] | None = None, upstream_path: Path = UPSTREAM_PATH,
) -> tuple[Any, dict[str, Any] | None]:
    if compute is None and not checkpoint.is_file():
        raise FileNotFoundError(f"official AMT checkpoint not found: {checkpoint}")
    state = None
    if compute is None:
        module, state = import_official_module(upstream_path)
        compute = module.compute_motion_smoothness
    with tempfile.TemporaryDirectory(prefix="motion-smooth-official-") as root:
        metadata = Path(root) / "official_input.json"
        metadata.write_text(json.dumps(metadata_to_official_entries(videos)), encoding="utf-8")
        result = compute(str(metadata), device, {"config": str(config), "ckpt": str(checkpoint)})
    return result, state
