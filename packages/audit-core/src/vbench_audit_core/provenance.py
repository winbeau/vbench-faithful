"""Small, model-free provenance helpers shared by metric entry points."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


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
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
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

    return {
        "code_sha": _code_sha(),
        "input_sha256": {str(video): _sha256(video) for video in videos},
        "metadata_sha256": _sha256(metadata_path),
        "model_config_sha256": _sha256(model_config_path),
        "upstream": {"status": upstream_status},
    }


__all__ = ["collect_provenance"]
