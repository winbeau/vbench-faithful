"""Workspace and output path helpers shared by every metric CLI."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Return the repository root containing the workspace configuration.

    Resolution is performed at call time so an installed package does not
    freeze an environment variable or the caller's current directory at
    import time.  ``VBENCH_AUDIT_WORKSPACE`` is an explicit override useful
    for editable and isolated installations.
    """

    override = os.environ.get("VBENCH_AUDIT_WORKSPACE")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise ValueError(f"VBENCH_AUDIT_WORKSPACE is not a directory: {candidate}")
        return candidate

    origin = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve()
    if origin.is_file():
        origin = origin.parent
    for parent in (origin, *origin.parents):
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8")
            except OSError:
                continue
            if "[tool.uv.workspace]" in text or "[tool.uv]" in text and "members" in text:
                return parent
    raise ValueError(
        "cannot locate the vbench-audit workspace; set VBENCH_AUDIT_WORKSPACE "
        "or pass an explicit --output directory"
    )


def output_base(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit output base or the workspace ``output`` folder."""

    if value is None:
        return workspace_root() / "output"
    return Path(value).expanduser().resolve()
