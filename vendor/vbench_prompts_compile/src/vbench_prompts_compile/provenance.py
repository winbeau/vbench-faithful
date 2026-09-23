"""Run provenance helpers shared by the training and evaluation entry points.

Torch/transformers are imported lazily so importing this module stays cheap in
the base development environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path | str = REPO_ROOT) -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # pragma: no cover - git may be absent
            return "unknown"

    return {"commit": run("rev-parse", "HEAD"), "dirty": run("status", "--porcelain")}


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "<unset>"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_total_memory_mib"] = int(properties.total_memory / (1024 * 1024))
            info["device_index_visible"] = 0
    except Exception:  # pragma: no cover
        info["torch"] = None
    for name in ("transformers", "trl", "peft", "datasets"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover
            info[name] = None
    return info
