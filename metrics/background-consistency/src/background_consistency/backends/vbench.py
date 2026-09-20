from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vbench_audit_core.contracts import not_implemented_batch
from vbench_audit_core.schemas import VideoResult

METRIC = "background-consistency"


def evaluate_batch(backend: str, videos: Sequence[Path], metadata: Mapping[str, Mapping[str, Any]], device: str | None, config: Mapping[str, Any]) -> list[VideoResult]:
    return not_implemented_batch(METRIC, backend, videos, variant="official", reason="VBench compute_background_consistency adapter is deferred")
