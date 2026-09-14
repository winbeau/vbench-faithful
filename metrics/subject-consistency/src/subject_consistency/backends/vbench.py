from __future__ import annotations

from pathlib import Path

from vbench_audit_core.upstream import (
    UPSTREAM_SHA,
    UPSTREAM_URL,
    import_official_module as _import_official_module,
    inspect_upstream as _inspect_upstream,
    verify_upstream as _verify_upstream,
)

UPSTREAM_REMOTE = UPSTREAM_URL
UPSTREAM_BRANCH = ""  # branch names are intentionally not part of identity


def inspect_upstream(path: Path | None = None):
    return _inspect_upstream(path)


def verify_upstream(path: Path | None = None):
    return _verify_upstream(path, dimension="subject_consistency")


def import_official_module(path: Path | None = None):
    return _import_official_module("subject_consistency", path)


def _similarity_parts(features):
    if getattr(features, "ndim", None) != 2:
        raise ValueError("features must have shape [T, D]")
    if int(features.shape[0]) < 2:
        raise ValueError("Subject Consistency requires at least two frames")
    similarities = (features @ features.transpose(0, 1)).clamp_min(0.0)
    return similarities.diagonal(offset=1), similarities[0, 1:]


def official_subject_consistency(features) -> float:
    """VBench1.0: mean_t 0.5*(previous cosine + fixed-first cosine)."""

    previous, first = _similarity_parts(features)
    return float((0.5 * previous + 0.5 * first).mean().item())


def official_diagnostics(features):
    from ..diagnostics import OfficialDiagnostics

    previous, first = _similarity_parts(features)
    local_score = float(previous.mean().item())
    first_anchor_score = float(first.mean().item())
    return OfficialDiagnostics(
        num_frames=int(features.shape[0]),
        local_score=local_score,
        global_score=first_anchor_score,
        final_score=float((0.5 * previous + 0.5 * first).mean().item()),
    )
