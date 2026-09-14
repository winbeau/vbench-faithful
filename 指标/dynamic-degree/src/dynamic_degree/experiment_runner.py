"""Thin manifest-to-backend dispatcher; it owns no Dynamic Degree mathematics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .diagnostics import DiagnosticsLevel
from .metric import evaluate_audit_batch, evaluate_vbench_batch
from .backends.audit import AuditConfig, AuditVariant


class VariantNotReady(RuntimeError):
    pass


def evaluate_variant(video_path: Path, prompt: str, target_type: str, metric_variant: str, *, device: Any, model_weight: Path) -> dict[str, Any]:
    """Run only an implementation that exists; unresolved variants fail explicitly."""
    video = Path(video_path)
    if metric_variant == 'official':
        # Official VBench Dynamic Degree has no prompt routing.
        return evaluate_vbench_batch([video], {video.name: {'prompt': prompt}}, device, model_weight)[0]
    variants = {
        'time_only': AuditVariant.TIME_ONLY,
        'source_only': AuditVariant.SOURCE_ONLY,
        'duration_only': AuditVariant.DURATION_ONLY,
        'source_time': AuditVariant.SOURCE_TIME,
        'full': AuditVariant.FULL,
    }
    if metric_variant in variants:
        metadata = {video.name: {'prompt': prompt, 'motion_target': target_type.lower()}}
        return evaluate_audit_batch([video], metadata, device, model_weight, DiagnosticsLevel.COMPACT, config=AuditConfig(variant=variants[metric_variant]))[0]
    raise ValueError(f'unknown metric_variant: {metric_variant}')
