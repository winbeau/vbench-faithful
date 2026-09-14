"""Schema-validated, append-safe Dynamic Degree evaluation records.

This module deliberately performs no metric inference and is shared by future
official and ablation runners.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

REQUIRED_FIELDS = (
    "base_id", "derived_id", "split", "dimension", "intervention_family",
    "intervention_level", "prompt", "target_type", "expected_relation",
    "metric_name", "metric_variant", "score", "seed", "failure_or_abstention",
)
TARGET_TYPES = {"SUBJECT", "CAMERA", "GENERIC", "BOTH", "UNKNOWN"}
METRIC_VARIANTS = {"official", "time_only", "source_only", "duration_only", "source_time", "full"}
RESUME_KEY = ("base_id", "derived_id", "metric_name", "metric_variant", "target_type")


def validate_evaluation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized record or raise ValueError before it reaches JSONL."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    extras = set(record) - set(REQUIRED_FIELDS)
    if missing or extras:
        raise ValueError(f"evaluation record fields mismatch: missing={missing}, extras={sorted(extras)}")
    value = dict(record)
    if not isinstance(value["base_id"], str) or not value["base_id"]:
        raise ValueError("base_id must be a non-empty string")
    if not isinstance(value["derived_id"], str) or not value["derived_id"]:
        raise ValueError("derived_id must be a non-empty string")
    if value["split"] not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if value["dimension"] != "dynamic_degree" or value["metric_name"] != "dynamic_degree":
        raise ValueError("dimension and metric_name must be dynamic_degree")
    if value["target_type"] not in TARGET_TYPES:
        raise ValueError("unsupported target_type")
    if value["metric_variant"] not in METRIC_VARIANTS:
        raise ValueError("unsupported metric_variant")
    if value["score"] is not None and (not isinstance(value["score"], (int, float)) or isinstance(value["score"], bool)):
        raise ValueError("score must be numeric or null")
    if value["score"] is None and value["failure_or_abstention"] is None:
        raise ValueError("null score requires failure_or_abstention")
    if value["score"] is not None and value["failure_or_abstention"] is not None:
        raise ValueError("successful score requires null failure_or_abstention")
    return value


def completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    return {tuple(json.loads(line)[field] for field in RESUME_KEY) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_evaluation_record(path: str | Path, record: Mapping[str, Any]) -> bool:
    """Append one completed logical evaluation; False means it already exists."""
    path = Path(path)
    normalized = validate_evaluation_record(record)
    key = tuple(normalized[field] for field in RESUME_KEY)
    if key in completed_keys(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
    return True
