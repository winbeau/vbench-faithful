"""Small, per-dimension venvs backed by existing, read-only model runtimes.

This is dependency sharing, not a package resolver or a filesystem sandbox. The
base runtimes must remain immutable while workers run. Package metadata and
startup paths are fingerprinted on every preparation; model weights and all
installed package payloads are deliberately not rehashed or copied here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_SCHEMA = "vbench-shared-environment/1"
_RECEIPT = "environment.json"
_MARKER = "VBENCH_ENVIRONMENT_JSON:"

# Run under the source interpreter: root Python and model Python need not have
# the same ABI. -I excludes caller PYTHONPATH/user-site; -B avoids base writes.
_IDENTIFY = r'''
import hashlib, importlib.metadata, json, os, pathlib, site, sys, sysconfig

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

paths = list(dict.fromkeys(os.path.realpath(p) for p in sys.path if p))
sites = list(dict.fromkeys(os.path.realpath(p) for p in site.getsitepackages()
                          if os.path.isdir(p)))
sites += [p for p in paths if pathlib.Path(p).name in ("site-packages", "dist-packages")
          and p not in sites and os.path.isdir(p)]
distributions = []
for directory in paths:
    root = pathlib.Path(directory)
    for info in sorted([*root.glob("*.dist-info"), *root.glob("*.egg-info")]):
        dist = importlib.metadata.Distribution.at(info)
        files = sorted(p for p in info.rglob("*") if p.is_file()) if info.is_dir() else [info]
        distributions.append({"path": str(info), "name": dist.metadata.get("Name"),
                              "version": dist.version,
                              "metadata_sha256": {str(p.relative_to(info)) if info.is_dir() else info.name:
                                                  digest(p) for p in files}})
startup = {}
for directory in sites:
    for p in sorted(pathlib.Path(directory).glob("*.pth")):
        startup[str(p)] = digest(p)
cfg = pathlib.Path(sys.prefix) / "pyvenv.cfg"
data = {"executable": sys.executable, "real_executable": os.path.realpath(sys.executable),
        "executable_sha256": digest(pathlib.Path(sys.executable)),
        "version": sys.version, "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag, "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"), "platform": sysconfig.get_platform(),
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "sys_path": paths, "site_packages": sites, "distributions": distributions,
        "startup_sha256": startup, "pyvenv_cfg_sha256": digest(cfg) if cfg.is_file() else None}
print("VBENCH_ENVIRONMENT_JSON:" + json.dumps(data, sort_keys=True))
'''

# This module lives in the thin venv, not the shared runtime. Executable .pth
# files still work for external model packages, including editable installs.
# Metric editables are removed both from sys.path and setuptools' import maps.
_BRIDGE = r'''
import json
import os
from pathlib import Path
import site
import sys

_active = False
_explicit = set()

def _metric_path(value):
    parts = Path(value).resolve().parts
    return any(parts[i] == "metrics" and parts[i + 2] == "src"
               for i in range(len(parts) - 2))

def clean():
    sys.path[:] = [p for p in sys.path if not p or not _metric_path(p)
                   or os.path.realpath(p) in _explicit]
    for name, module in list(sys.modules.items()):
        if name.startswith("__editable__"):
            mapping = getattr(module, "MAPPING", {})
            for key, value in list(mapping.items()):
                if _metric_path(value):
                    del mapping[key]
            namespaces = getattr(module, "NAMESPACES", {})
            for key, values in list(namespaces.items()):
                kept = [p for p in values if not _metric_path(p)]
                if kept:
                    namespaces[key] = kept
                else:
                    del namespaces[key]
        location = getattr(module, "__file__", None)
        if location and _metric_path(location):
            # Never inherit a module eagerly imported by a base startup hook.
            sys.modules.pop(name, None)
    sys.path_importer_cache.clear()

def activate():
    global _active, _explicit
    if _active:
        return
    _active = True
    sys.dont_write_bytecode = True
    # The dispatcher owns source selection. Do not resurrect selected sources
    # from the base: only explicit caller PYTHONPATH entries may survive.
    _explicit = {os.path.realpath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p}
    receipt = json.loads((Path(sys.prefix) / "environment.json").read_text())
    base = receipt["base"]
    for directory in base["site_packages"]:
        site.addsitedir(directory)
    for path in base["sys_path"]:
        if path not in sys.path and not _metric_path(path):
            sys.path.append(path)
    clean()
'''
_PTH = "import _vbench_shared_runtime; _vbench_shared_runtime.activate()\n"
# Keep base sitecustomize from reintroducing metric paths after .pth processing.
# Its dependency search paths have already been captured by the source probe.
_CUSTOMIZE = "import _vbench_shared_runtime\n_vbench_shared_runtime.clean()\n"
_INSPECT = r'''
import json, site, sys, sysconfig
print("VBENCH_ENVIRONMENT_JSON:" + json.dumps({
    "prefix": sys.prefix, "version": sys.version, "soabi": sysconfig.get_config_var("SOABI"),
    "site_packages": site.getsitepackages(), "sys_path": sys.path}))
'''


def _clean_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "UV_")) and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _probe(python: Path, code: str) -> dict:
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", code], env=_clean_environment(),
        text=True, capture_output=True, check=True, timeout=120,
    )
    lines = [line[len(_MARKER):] for line in completed.stdout.splitlines() if line.startswith(_MARKER)]
    if len(lines) != 1:
        raise RuntimeError(f"Could not identify Python runtime: {python}")
    return json.loads(lines[0])


def _metric_path(value: str) -> bool:
    parts = Path(value).resolve().parts
    return any(parts[i] == "metrics" and parts[i + 2] == "src" for i in range(len(parts) - 2))


def _validate_runtime(folder: Path, base: dict) -> dict:
    runtime = _probe(folder / "bin/python", _INSPECT)
    if (Path(runtime["prefix"]).resolve() != folder.resolve()
            or runtime["version"] != base["version"] or runtime["soabi"] != base["soabi"]):
        raise RuntimeError(f"Thin environment has a different Python ABI or prefix: {folder}")
    if any(_metric_path(p) for p in runtime["sys_path"] if p):
        raise RuntimeError(f"Thin environment inherited metric source paths: {folder}")
    return runtime


def _generated_files(site_packages: Path) -> dict[Path, str]:
    return {site_packages / "_vbench_shared_runtime.py": _BRIDGE,
            site_packages / "_vbench_shared_runtime.pth": _PTH,
            site_packages / "sitecustomize.py": _CUSTOMIZE}


def prepare_environment(dimension: str, role: str, base_python: Path, env_root: Path) -> Path:
    """Return an isolated venv Python sharing the selected model dependencies.

    ``dimension`` is the dispatcher's canonical underscore name and ``role`` is
    ``visual`` or ``semantic``. No dependency installation, interpreter download
    or mutation of the base environment occurs. Existing fingerprints are kept;
    base Python/package metadata changes select a new sibling directory. A bad
    existing receipt fails closed instead of replacing a possibly running venv.

    Callers set PYTHONPATH to the selected metric and shared source packages.
    Shared dependencies are not hermetically copied: an in-place base mutation
    while a worker is running is outside this contract.
    """
    if not isinstance(dimension, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", dimension):
        raise ValueError("dimension must be a canonical underscore name")
    if role not in {"visual", "semantic"}:
        raise ValueError("role must be visual or semantic")
    # Resolving the executable symlink would bypass the source venv's packages.
    base_python = Path(os.path.abspath(Path(base_python).expanduser()))
    if not base_python.is_file():
        raise FileNotFoundError(base_python)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare evaluation environments")
    env_root = Path(env_root).expanduser().resolve()
    base = _probe(base_python, _IDENTIFY)
    base_prefix = Path(base["prefix"]).resolve()
    if env_root == base_prefix or base_prefix in env_root.parents:
        raise ValueError("env_root must be outside the shared base runtime")
    parent = env_root / dimension / role
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".prepare.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Recheck after waiting for another process; its preparation may have
        # overlapped an externally managed base environment update.
        base = _probe(base_python, _IDENTIFY)
        identity = {"schema": _SCHEMA, "dimension": dimension, "role": role,
                    "base_python": str(base_python), "base": base,
                    "bridge_sha256": hashlib.sha256((_BRIDGE + _PTH + _CUSTOMIZE).encode()).hexdigest()}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = {**identity, "fingerprint": fingerprint}
        destination = parent / fingerprint
        if destination.exists():
            try:
                if json.loads((destination / _RECEIPT).read_text()) != receipt:
                    raise ValueError("receipt differs")
                runtime = _validate_runtime(destination, base)
                local_site = Path(runtime["site_packages"][0])
                if any(path.read_text() != value for path, value in _generated_files(local_site).items()):
                    raise ValueError("shared dependency bridge differs")
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"Invalid cached evaluation environment; use a new env_root: {destination}") from exc
            return destination / "bin/python"

        staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint[:12]}-", dir=parent))
        try:
            subprocess.run(
                [uv, "venv", "--python", str(base_python), "--no-project", "--no-config",
                 "--offline", "--no-python-downloads", "--relocatable", str(staging)],
                env=_clean_environment(), text=True, capture_output=True, check=True, timeout=120,
            )
            runtime = _probe(staging / "bin/python", _INSPECT)
            local_site = Path(runtime["site_packages"][0])
            if staging not in local_site.parents:
                raise RuntimeError("uv did not create an isolated site-packages directory")
            (staging / _RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            for path, value in _generated_files(local_site).items():
                path.write_text(value)
            _validate_runtime(staging, base)
            if _probe(base_python, _IDENTIFY) != base:
                raise RuntimeError("Shared base runtime changed while preparing the environment; retry when stable")
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination / "bin/python"


__all__ = ["prepare_environment"]
