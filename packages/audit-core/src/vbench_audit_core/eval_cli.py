"""Workspace entry points for dimension packages using the YAML controller."""
from __future__ import annotations

import runpy
import sys

from .paths import workspace_root


def metric_main(dimension: str, argv: list[str] | None = None) -> int:
    """Use the same controller, cache and isolated workers as scripts/eval.py."""
    scripts = workspace_root() / "scripts"
    previous = sys.path[:]
    try:
        sys.path.insert(0, str(scripts))
        namespace = runpy.run_path(str(scripts / "eval.py"))
    finally:
        sys.path[:] = previous
    return namespace["main"](argv, dimension=dimension)
