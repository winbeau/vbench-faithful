"""Reproducible loading and provenance checks for the locked VBench checkout.

This module deliberately does not import ``vbench`` at module import time.  The
official repository has a large optional dependency closure and, more
importantly, importing a module from a different checkout in the same Python
process would make a parity result untrustworthy.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Python 3.10 requires the audit-core 'tomli' dependency") from exc


@dataclass(frozen=True)
class UpstreamSpec:
    dimension: str
    module: str
    entrypoint: str
    source: str
    source_sha256: str


@dataclass(frozen=True)
class UpstreamConfig:
    url: str
    sha: str
    dimensions: Mapping[str, UpstreamSpec]


@dataclass(frozen=True)
class UpstreamState:
    path: str
    remote: str
    branch: str
    sha: str
    dirty: bool
    submodules: tuple[str, ...]
    source_hashes: Mapping[str, str]

    @property
    def source_type(self) -> str:
        return "github" if self.remote == UPSTREAM_URL else "unknown"


def _workspace_root() -> Path:
    """Use the shared runtime path resolver for all source lookup."""
    try:
        from .paths import workspace_root  # type: ignore
    except ImportError as exc:
        raise RuntimeError("audit-core paths.workspace_root is unavailable") from exc
    value = workspace_root()
    if value is None:
        raise RuntimeError("audit-core paths.workspace_root returned no workspace")
    return Path(value).resolve()


def config_path() -> Path:
    return _workspace_root() / "configs" / "upstream.toml"


def load_config(path: Path | None = None) -> UpstreamConfig:
    selected = Path(path) if path is not None else config_path()
    try:
        data = tomllib.loads(selected.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"VBench upstream configuration is missing: {selected}") from exc
    root = data.get("upstream")
    if not isinstance(root, dict):
        raise ValueError(f"{selected} must define [upstream]")
    dimensions_raw = data.get("dimensions")
    if not isinstance(dimensions_raw, dict):
        raise ValueError(f"{selected} must define [dimensions.<name>] entries")
    dimensions: dict[str, UpstreamSpec] = {}
    for name, value in dimensions_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"invalid upstream dimension entry: {name}")
        try:
            dimensions[name] = UpstreamSpec(
                dimension=name,
                module=str(value["module"]),
                entrypoint=str(value["entrypoint"]),
                source=str(value["source"]),
                source_sha256=str(value["source_sha256"]),
            )
        except KeyError as exc:
            raise ValueError(f"upstream dimension {name!r} is incomplete") from exc
    return UpstreamConfig(url=str(root["url"]), sha=str(root["sha"]), dimensions=dimensions)


# Compatibility exports are generated from the config rather than copied into
# each adapter.  They are informational only; verification always rereads the
# configuration at call time.
_CONFIG = load_config()
UPSTREAM_URL = _CONFIG.url
UPSTREAM_SHA = _CONFIG.sha
UPSTREAM_REMOTE = UPSTREAM_URL


def upstream_spec(dimension: str, path: Path | None = None) -> UpstreamSpec:
    config = load_config(path)
    try:
        return config.dimensions[dimension]
    except KeyError as exc:
        raise KeyError(f"unknown VBench dimension: {dimension!r}") from exc


def resolve_upstream_path(path: Path | str | None = None) -> Path:
    """Resolve one checkout for every dimension.

    An explicit argument wins, followed by ``VBENCH_AUDIT_UPSTREAM``.  The
    sibling checkout is the reproducible default.  ``VBENCH1_ROOT`` is retained
    only as a migration fallback and emits a warning so old shell setups are
    visible in run logs.
    """
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get("VBENCH_AUDIT_UPSTREAM")
    if configured:
        return Path(configured).expanduser().resolve()
    legacy = os.environ.get("VBENCH1_ROOT")
    if legacy:
        warnings.warn(
            "VBENCH1_ROOT is deprecated; use VBENCH_AUDIT_UPSTREAM",
            DeprecationWarning,
            stacklevel=2,
        )
        return Path(legacy).expanduser().resolve()
    return (_workspace_root().parent / "VBench").resolve()


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect VBench git checkout {path}: {exc}") from exc


def inspect_upstream(path: Path | str | None = None) -> UpstreamState:
    selected = resolve_upstream_path(path)
    config = load_config()
    if not selected.is_dir():
        raise FileNotFoundError(f"official VBench checkout does not exist: {selected}")
    remote = _git(selected, "remote", "get-url", "origin")
    branch = _git(selected, "branch", "--show-current")
    sha = _git(selected, "rev-parse", "HEAD")
    dirty = bool(_git(selected, "status", "--porcelain"))
    submodule_text = _git(selected, "submodule", "status")
    hashes: dict[str, str] = {}
    for name, spec in config.dimensions.items():
        source = selected / spec.source
        if not source.is_file():
            raise FileNotFoundError(f"official {name} source is missing: {source}")
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return UpstreamState(
        path=str(selected),
        remote=remote,
        branch=branch,
        sha=sha,
        dirty=dirty,
        submodules=tuple(line for line in submodule_text.splitlines() if line),
        source_hashes=hashes,
    )


def verify_upstream(path: Path | str | None = None, *, dimension: str | None = None) -> UpstreamState:
    config = load_config()
    state = inspect_upstream(path)
    if state.remote != config.url:
        raise RuntimeError(f"VBench origin mismatch: expected={config.url!r}, actual={state.remote!r}")
    if state.sha != config.sha:
        raise RuntimeError(f"VBench SHA mismatch: expected={config.sha}, actual={state.sha}")
    if state.dirty:
        raise RuntimeError("VBench checkout is dirty; refusing official evaluation")
    selected = config.dimensions.values() if dimension is None else (upstream_spec(dimension),)
    for spec in selected:
        actual = state.source_hashes[spec.dimension]
        if actual != spec.source_sha256:
            raise RuntimeError(
                f"VBench source hash mismatch for {spec.dimension}: "
                f"expected={spec.source_sha256}, actual={actual}"
            )
    return state


def _module_origin(module: Any) -> Path | None:
    origin = getattr(module, "__file__", None)
    return Path(origin).resolve() if origin else None


def _assert_no_wrong_cached_vbench(selected: Path) -> None:
    """Reject a process that already imported another VBench checkout."""
    for name, module in tuple(sys.modules.items()):
        if name != "vbench" and not name.startswith("vbench."):
            continue
        origin = _module_origin(module)
        if origin is not None and selected not in origin.parents:
            raise RuntimeError(
                f"wrong cached VBench module {name!r} from {origin}; "
                f"selected checkout is {selected}"
            )


def import_official_module(
    dimension: str,
    path: Path | str | None = None,
    *,
    module_name: str | None = None,
) -> tuple[Any, UpstreamState]:
    """Verify and import one locked official module from the selected checkout."""
    spec = upstream_spec(dimension)
    state = verify_upstream(path, dimension=dimension)
    selected = Path(state.path)
    _assert_no_wrong_cached_vbench(selected)
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))
    importlib.invalidate_caches()
    name = module_name or spec.module
    module = importlib.import_module(name)
    origin = _module_origin(module)
    if origin is None or selected not in origin.parents:
        raise RuntimeError(f"wrong VBench import for {name}: {origin}")
    return module, state


def source_hashes(path: Path | str | None = None) -> dict[str, str]:
    return dict(verify_upstream(path).source_hashes)
