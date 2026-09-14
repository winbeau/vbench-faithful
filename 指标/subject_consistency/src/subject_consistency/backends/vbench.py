from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_PATH = Path("/home/msy625/vbench1")
UPSTREAM_REMOTE = "https://github.com/msy625/VBench.git"
UPSTREAM_BRANCH = "master"
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"


@dataclass(frozen=True)
class UpstreamState:
    path: str
    remote: str
    branch: str
    sha: str
    dirty: bool
    source_type: str


def _source_type(origin: str) -> str:
    if origin == UPSTREAM_REMOTE:
        return "github"
    candidate = Path(origin).expanduser()
    if candidate.suffix == ".bundle" and candidate.is_file():
        return "local_bundle"
    return "untrusted"


def inspect_upstream(path: Path | None = None) -> UpstreamState:
    path = path or Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()

    origin = git("remote", "get-url", "origin")
    return UpstreamState(
        path=str(path.resolve()),
        remote=origin,
        branch=git("branch", "--show-current"),
        sha=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        source_type=_source_type(origin),
    )


def verify_upstream(path: Path | None = None) -> UpstreamState:
    state = inspect_upstream(path)
    if state.source_type not in {"github", "local_bundle"}:
        raise RuntimeError(f"untrusted upstream origin: {state.remote!r}")
    expected = (UPSTREAM_BRANCH, UPSTREAM_SHA)
    actual = (state.branch, state.sha)
    if actual != expected:
        raise RuntimeError(f"upstream identity mismatch: expected={expected}, actual={actual}")
    if state.dirty:
        raise RuntimeError("upstream worktree is dirty; refusing Subject Consistency evaluation")
    return state


def import_official_module(path: Path | None = None):
    path = path or Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()
    state = verify_upstream(path)
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.subject_consistency")
    module_path = Path(module.__file__).resolve()
    if path.resolve() not in module_path.parents:
        raise RuntimeError(f"wrong vbench import: {module_path}")
    return module, state


def _similarity_parts(features):
    if getattr(features, "ndim", None) != 2:
        raise ValueError("features must have shape [T, D]")
    if int(features.shape[0]) < 2:
        raise ValueError("Subject Consistency requires at least two frames")
    similarities = (features @ features.transpose(0, 1)).clamp_min(0.0)
    return similarities.diagonal(offset=1), similarities[0, 1:]


def official_subject_consistency(features) -> float:
    """VBench1.0: mean_t 0.5*(previous cosine + fixed-first cosine)."""

    previous, first = _similarity_parts(features)
    return float((0.5 * previous + 0.5 * first).mean().item())


def official_diagnostics(features):
    from ..diagnostics import OfficialDiagnostics

    previous, first = _similarity_parts(features)
    local_score = float(previous.mean().item())
    first_anchor_score = float(first.mean().item())
    return OfficialDiagnostics(
        num_frames=int(features.shape[0]),
        local_score=local_score,
        global_score=first_anchor_score,
        final_score=float((0.5 * previous + 0.5 * first).mean().item()),
    )
