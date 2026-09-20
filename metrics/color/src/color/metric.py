from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vbench_audit_core.contracts import not_implemented_batch
from vbench_audit_core.schemas import VideoResult

from .backends import audit, repair, vbench

METRIC = "color"


def evaluate_batch(
    backend: str,
    videos: Sequence[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: str | None,
    config: Mapping[str, Any],
) -> list[VideoResult]:
    """Shared batch shape; GRiT binding and color predicates are deferred."""

    if backend == "vbench":
        return vbench.evaluate_batch(backend, videos, metadata, device, config)
    if backend == "audit":
        variant = config.get("runtime", {}).get("audit_variant", "diagnostic")
        return repair.evaluate_batch(backend, videos, metadata, device, config) if variant == "repair" else audit.evaluate_batch(backend, videos, metadata, device, config)
    return not_implemented_batch(METRIC, backend, videos, reason=f"unknown backend: {backend}")


evaluate_batch.requires_cuda = False


__all__ = ["METRIC", "evaluate_batch"]
