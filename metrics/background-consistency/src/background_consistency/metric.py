from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vbench_audit_core.contracts import not_implemented_batch
from vbench_audit_core.schemas import VideoResult

from .backends import audit, repair, vbench

METRIC = "background-consistency"


def evaluate_batch(
    backend: str,
    videos: Sequence[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: str | None,
    config: Mapping[str, Any],
) -> list[VideoResult]:
    """Real model-backed evaluation with one explicit result per input."""

    inference = None
    if config.get("model", {}).get("clip", {}).get("checkpoint"):
        from .inference import configure_inference
        inference = configure_inference(config.get("runtime", {}).get("seed", 0))
    if backend == "vbench":
        results = vbench.evaluate_batch(backend, videos, metadata, device, config)
    elif backend == "audit":
        variant = config.get("runtime", {}).get("audit_variant", "repair")
        results = audit.evaluate_batch(backend, videos, metadata, device, config) if variant in ("diagnostic", "aggregation") else repair.evaluate_batch(backend, videos, metadata, device, config)
    else:
        return not_implemented_batch(METRIC, backend, videos, reason=f"unknown backend: {backend}")
    if inference is not None:
        for result in results:
            result.metric = {**(result.metric or {}), "inference": inference}
    return results


evaluate_batch.requires_cuda = True


__all__ = ["METRIC", "evaluate_batch"]
