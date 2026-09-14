from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import LockedUmtClassifier, OFFICIAL_NUM_FRAMES, OFFICIAL_THRESHOLD, OFFICIAL_TOP_K
from ..schemas import OfficialActionResult

UPSTREAM_PATH = Path("/root/vbench1")
UPSTREAM_BRANCH = "master"
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"
DEFAULT_WEIGHT = Path("/root/autodl-tmp/vbench-audit-storage/models/umt/l16_ptk710_ftk710_ftk400_f16_res224.pth")


@dataclass(frozen=True)
class UpstreamState:
    path: str
    remote: str
    branch: str
    sha: str
    dirty: bool
    submodules: tuple[str, ...]


def inspect_upstream(path: Path = UPSTREAM_PATH) -> UpstreamState:
    if not path.is_dir():
        raise RuntimeError(f"upstream repository does not exist: {path}")
    required_source = path / "vbench" / "human_action.py"
    if not required_source.is_file():
        raise RuntimeError(f"locked Human Action source is missing: {required_source}")
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    return UpstreamState(
        path=str(path.resolve()),
        remote=git("remote", "get-url", "origin"),
        branch=git("branch", "--show-current"),
        sha=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        submodules=tuple(line for line in git("submodule", "status").splitlines() if line),
    )


def verify_upstream(path: Path = UPSTREAM_PATH) -> UpstreamState:
    state = inspect_upstream(path)
    expected = (UPSTREAM_BRANCH, UPSTREAM_SHA)
    actual = (state.branch, state.sha)
    if actual != expected:
        raise RuntimeError(f"upstream identity mismatch: expected={expected}, actual={actual}")
    if state.dirty:
        raise RuntimeError("upstream worktree is dirty; refusing Human Action evaluation")
    return state


def import_official_module(path: Path = UPSTREAM_PATH):
    state = verify_upstream(path)
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.human_action")
    if path.resolve() not in Path(module.__file__).resolve().parents:
        raise RuntimeError(f"wrong vbench import: {module.__file__}")
    return module, state


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
        upstream_path: Path = UPSTREAM_PATH,
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
