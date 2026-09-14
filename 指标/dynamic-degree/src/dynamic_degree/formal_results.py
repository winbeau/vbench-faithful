"""Formal-batch result classification without changing evaluator semantics."""
from __future__ import annotations

import math
from typing import Any, Mapping


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def valid_structured_evidence(evidence: Any) -> bool:
    """Return whether the enabled components produced their required evidence."""
    if not isinstance(evidence, Mapping):
        return False
    provenance = evidence.get("component_provenance")
    routing = evidence.get("routing")
    if not isinstance(provenance, Mapping) or not isinstance(routing, Mapping):
        return False
    if not _finite_number(evidence.get("apparent_intensity")):
        return False
    if provenance.get("source_decomposition"):
        if not all(_finite_number(evidence.get(name)) for name in ("camera_intensity", "residual_intensity")):
            return False
    if provenance.get("duration_persistence"):
        required = ["apparent_coverage"]
        if provenance.get("source_decomposition"):
            required.extend(("camera_coverage", "residual_coverage"))
        if not all(_finite_number(evidence.get(name)) for name in required):
            return False
    return True


def classify_evaluation_result(result: Mapping[str, Any]) -> str:
    """Classify a persisted result as scalar, structured, failed, or unresolved.

    ``unknown_motion_target`` is an evaluator routing outcome, not a failed
    structured computation, when all enabled component evidence is present.
    The raw evaluator status is retained separately by the stage runner.
    """
    failure = result.get("failure_reason")
    structured_ok = valid_structured_evidence(result.get("structured_evidence"))
    if failure is not None and not (failure == "unknown_motion_target" and structured_ok):
        return "failed_or_abstained"
    if result.get("score") is not None:
        return "succeeded_scalar"
    if structured_ok:
        return "succeeded_structured"
    return "unresolved"


def normalize_batch_result(result: dict[str, Any]) -> dict[str, Any]:
    """Attach batch status and clear only the known routing pseudo-failure."""
    classification = classify_evaluation_result(result)
    result["evaluation_kind"] = classification
    if classification == "succeeded_structured" and result.get("failure_reason") == "unknown_motion_target":
        result["raw_evaluator_status"] = result.get("status")
        result["raw_evaluator_failure_reason"] = result.get("failure_reason")
        result["status"] = "succeeded_structured"
        result["failure_reason"] = None
    elif classification == "succeeded_scalar":
        result["status"] = "succeeded_scalar"
    return result
