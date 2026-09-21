from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vbench_audit_core.schemas import VideoResult
from ..runtime import evaluate_videos

METRIC = "background-consistency"


def evaluate_batch(backend: str, videos: Sequence[Path], metadata: Mapping[str, Mapping[str, Any]], device: str | None, config: Mapping[str, Any]) -> list[VideoResult]:
    return evaluate_videos(backend, videos, metadata, device, config, method="official")
