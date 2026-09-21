from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vbench_audit_core.contracts import not_implemented_batch
from vbench_audit_core.schemas import VideoResult

METRIC = "color"


def evaluate_batch(backend: str, videos: Sequence[Path], metadata: Mapping[str, Mapping[str, Any]], device: str | None, config: Mapping[str, Any]) -> list[VideoResult]:
    from ..runtime import evaluate_audit
    variant = config.get("runtime", {}).get("audit_variant", "repair")
    return evaluate_audit(videos, metadata, device, config,
                          variant="repair" if variant == "diagnostic" else variant)
