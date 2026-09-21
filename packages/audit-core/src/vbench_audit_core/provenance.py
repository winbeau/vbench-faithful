"""Small, model-free provenance helpers shared by metric entry points."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any
from .paths import workspace_root


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace_root(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def collect_provenance(
    videos: list[Path],
    metadata_path: Path | None,
    model_config_path: Path | None,
    *,
    upstream_status: str = "not_checked",
) -> dict[str, Any]:
    """Return one stable envelope for inputs and the current code revision.

    Model loading and upstream verification deliberately stay outside this
    helper.  A skeleton run can therefore produce auditable ``not_checked``
    provenance without importing CUDA or downloading anything.
    """

    try:
        root = workspace_root()
    except ValueError:
        # An independently installed wheel can run with an explicit --output
        # outside any checkout. Retain input provenance without inventing a
        # workspace revision or making result writing depend on one.
        root = None
    if root is None:
        package = Path(__file__).resolve().parent
        sources = {"installed/audit-core/" + str(p.relative_to(package)): _sha256(p)
                   for p in sorted(package.rglob("*.py"))}
        code_scope = "installed audit-core source only; no workspace checkout found"
    else:
        code_files = sorted([*root.glob("packages/*/src/**/*.py"), *root.glob("metrics/*/src/**/*.py")])
        sources = {str(p.relative_to(root)): _sha256(p) for p in code_files}
        code_scope = "containing checkout HEAD; source hashes identify editable or copied worktree code"
    return {
        "code_sha": _code_sha(),
        "code_root": str(root) if root is not None else None,
        "source_file_sha256": sources,
        "code_sha_scope": code_scope,
        "input_sha256": {str(video): _sha256(video) for video in videos},
        "metadata_sha256": _sha256(metadata_path),
        "model_config_sha256": _sha256(model_config_path),
        "upstream": {"status": upstream_status},
    }


__all__ = ["collect_provenance"]
