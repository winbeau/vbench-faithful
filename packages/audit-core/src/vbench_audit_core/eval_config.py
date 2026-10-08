"""Strict, flat YAML configuration for the unified evaluation command.

Loading a configuration does not load media, models, or create output folders.
Media paths inside the input JSON/JSONL retain ``paper_common.load_inputs``'s
contract: relative to the input manifest, unless ``video_root`` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import os
from pathlib import Path
import re

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent, MappingStartEvent, ScalarEvent
from yaml.nodes import MappingNode, SequenceNode

from .paths import workspace_root


PAPER_DIMENSIONS = (
    "scene", "human_action", "object_class", "subject_consistency",
    "background_consistency", "dynamic_degree", "spatial_relationship",
    "multiple_objects", "color",
)
OFFICIAL_DIMENSIONS = PAPER_DIMENSIONS + (
    "motion_smoothness", "temporal_flickering", "aesthetic_quality",
    "imaging_quality", "temporal_style", "overall_consistency", "appearance_style",
)
_ALIASES = {
    "dynamics_degree": "dynamic_degree",
    "multiplt_object": "multiple_objects",
    "color_consistency": "color",
}
_KEYS = {
    "version", "input", "assets", "output", "video_root", "backend",
    "dimensions", "gpus", "cache_dir", "env_dir", "reuse",
}
_PATH_KEYS = {"input", "assets", "output", "video_root", "cache_dir", "env_dir"}
_ENVIRONMENT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_GPU_UUID = re.compile(
    r"(?:GPU|MIG)-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)


@dataclass(frozen=True)
class EvalConfig:
    version: int
    input: Path
    assets: Path
    output: Path
    video_root: Path | None
    backend: str
    dimensions: tuple[str, ...]
    gpus: tuple[int | str, ...]
    cache_dir: Path
    env_dir: Path
    reuse: bool

    def as_dict(self) -> dict:
        """Return the resolved configuration using only JSON-compatible values."""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        return result


class _ConfigLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        event = self.peek_event()
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ConstructorError(None, None, "YAML aliases and anchors are not supported", event.start_mark)
        if parent is not None and isinstance(event, MappingStartEvent):
            raise ConstructorError(None, None, "nested YAML mappings are not supported", event.start_mark)
        if isinstance(parent, SequenceNode) and not isinstance(event, ScalarEvent):
            raise ConstructorError(None, None, "YAML lists must contain scalar values, not nested collections", event.start_mark)
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, "expected a YAML mapping", node.start_mark)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConstructorError(None, None, "configuration keys must be strings", key_node.start_mark)
            if key in result:
                raise ConstructorError(None, None, f"duplicate configuration key: {key}", key_node.start_mark)
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Do not apply YAML 1.1's surprising conversion of yes/no/on/off into booleans.
# Copy the resolver table so other SafeLoader users in the process are untouched.
_ConfigLoader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in values if tag != "tag:yaml.org,2002:bool"]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf"),
)


def _check_keys(values: dict, source: str) -> None:
    if any(not isinstance(key, str) for key in values):
        raise ValueError(f"{source}: configuration keys must be strings")
    unknown = sorted(set(values) - _KEYS)
    if unknown:
        raise ValueError(f"{source}: unknown configuration keys: {', '.join(unknown)}")


def _path(value, key: str, base: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{key}: expected a nonempty path string")

    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"{key}: environment variable {name} is not set")
        return os.environ[name]

    expanded = _ENVIRONMENT.sub(replace, str(value))
    if "${" in expanded:
        raise ValueError(f"{key}: only ${{ENV_NAME}} environment substitution is supported")
    if not expanded.strip() or "\x00" in expanded:
        raise ValueError(f"{key}: expanded path must be nonempty and contain no NUL bytes")
    try:
        path = base / Path(expanded).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{key}: cannot resolve path: {exc}") from exc
    if key == "output" and (path.exists() or path.is_symlink()):
        raise ValueError("output: expected a new directory; the configured path already exists")
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{key}: cannot resolve path: {exc}") from exc


def _dimensions(value) -> tuple[str, ...]:
    if isinstance(value, str):
        if value == "all":
            return OFFICIAL_DIMENSIONS
        if value == "paper":
            return PAPER_DIMENSIONS
        raise ValueError("dimensions: expected 'all', 'paper', or a nonempty list of dimension names")
    if not isinstance(value, list) or not value:
        raise ValueError("dimensions: expected 'all', 'paper', or a nonempty list of dimension names")
    result = []
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError("dimensions: each dimension name must be a string")
        name = raw.replace("-", "_")
        name = _ALIASES.get(name, name)
        if name not in OFFICIAL_DIMENSIONS:
            raise ValueError(f"dimensions: unknown dimension {raw!r}")
        if name in result:
            raise ValueError(f"dimensions: duplicate dimension {name!r}")
        result.append(name)
    return tuple(result)


def _gpus(value) -> tuple[int | str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("gpus: expected a nonempty list of nonnegative integers or GPU-/MIG- UUIDs")
    seen = set()
    for gpu in value:
        if type(gpu) is int and gpu >= 0:
            identity = gpu
        elif isinstance(gpu, str) and _GPU_UUID.fullmatch(gpu):
            identity = gpu.casefold()
        else:
            raise ValueError("gpus: each device must be a nonnegative integer or a complete GPU-/MIG- UUID")
        if identity in seen:
            raise ValueError(f"gpus: duplicate device {gpu!r}")
        seen.add(identity)
    return tuple(value)


def load_config(path: Path, overrides: dict | None = None) -> EvalConfig:
    """Parse configuration and apply explicitly provided CLI overrides.

    File paths are relative to the YAML directory. Paths in ``overrides`` are
    relative to the caller's current directory. Required input/assets files are
    not opened here; their identities and contents are checked by the evaluator.
    """
    path = Path(path).expanduser().resolve()
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid evaluation YAML {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected one YAML mapping")
    _check_keys(raw, str(path))
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("overrides: expected a dictionary")
    _check_keys(overrides, "overrides")
    values = {**raw, **overrides}
    for key in ("input", "assets"):
        if key not in values:
            raise ValueError(f"{key}: required configuration field is missing")
    version = values.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("version: expected integer 1")
    backend = values.get("backend", "ours")
    if not isinstance(backend, str) or backend not in {"ours", "official", "origin", "repair", "both"}:
        raise ValueError("backend: expected 'ours', 'official', 'repair', 'origin', or 'both'")
    backend = {"ours": "repair", "official": "origin"}.get(backend, backend)
    reuse = values.get("reuse", True)
    if type(reuse) is not bool:
        raise ValueError("reuse: expected a boolean (true or false)")
    defaults = {
        "output": Path("../output/eval") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "video_root": None,
        "cache_dir": Path("~/.cache/vbench-repair"),
    }
    if "env_dir" not in values:
        defaults["env_dir"] = workspace_root() / ".venvs/metrics"
    resolved = {}
    for key in _PATH_KEYS:
        value = values.get(key, defaults.get(key))
        resolved[key] = None if key == "video_root" and value is None else _path(
            value, key, Path.cwd() if key in overrides else path.parent,
        )
    if resolved["input"].suffix not in {".json", ".jsonl"}:
        raise ValueError("input: expected a JSON or JSONL manifest (.json or .jsonl)")
    if resolved["output"].exists() or resolved["output"].is_symlink():
        raise ValueError("output: expected a new directory; the configured path already exists")
    return EvalConfig(
        version=version, backend=backend, reuse=reuse,
        dimensions=_dimensions(values.get("dimensions", "all")),
        gpus=_gpus(values.get("gpus", [0])), **resolved,
    )
