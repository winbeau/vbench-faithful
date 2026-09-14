"""Validated JSONL records for Dynamic Degree experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FIELDS = (
    'base_id', 'derived_id', 'split', 'dimension', 'intervention_family',
    'intervention_level', 'prompt', 'target_type', 'expected_relation',
    'metric_name', 'metric_variant', 'score', 'seed', 'failure_or_abstention',
)
VARIANTS = frozenset({'official', 'time_only', 'source_only', 'duration_only', 'source_time', 'full'})
TARGET_TYPES = frozenset({'SUBJECT', 'CAMERA', 'GENERIC', 'BOTH', 'UNKNOWN'})


class RecordValidationError(ValueError):
    pass


def record_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(str(record[name]) for name in ('base_id', 'derived_id', 'metric_name', 'metric_variant', 'target_type'))


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if set(record) != set(FIELDS):
        missing, extra = set(FIELDS) - set(record), set(record) - set(FIELDS)
        raise RecordValidationError(f'record fields mismatch: missing={sorted(missing)} extra={sorted(extra)}')
    result = dict(record)
    if result['split'] not in {'dev', 'test'} or result['dimension'] != 'dynamic_degree' or result['metric_name'] != 'dynamic_degree':
        raise RecordValidationError('invalid split, dimension, or metric_name')
    if result['metric_variant'] not in VARIANTS or result['target_type'] not in TARGET_TYPES:
        raise RecordValidationError('invalid metric_variant or target_type')
    if not isinstance(result['base_id'], str) or not isinstance(result['derived_id'], str) or not result['base_id'] or not result['derived_id']:
        raise RecordValidationError('base_id and derived_id must be nonempty strings')
    if result['score'] is not None and (not isinstance(result['score'], (int, float)) or isinstance(result['score'], bool)):
        raise RecordValidationError('score must be numeric or null')
    if result['failure_or_abstention'] is not None and not isinstance(result['failure_or_abstention'], str):
        raise RecordValidationError('failure_or_abstention must be string or null')
    return result


@dataclass
class EvaluationRecordWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = self._read_keys()

    def _read_keys(self) -> set[tuple[str, str, str, str, str]]:
        keys: set[tuple[str, str, str, str, str]] = set()
        if not self.path.exists():
            return keys
        for line_number, line in enumerate(self.path.read_text(encoding='utf-8').splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = validate_record(json.loads(line))
            except (json.JSONDecodeError, RecordValidationError) as exc:
                raise RecordValidationError(f'malformed existing record at line {line_number}: {exc}') from exc
            key = record_key(record)
            if key in keys:
                raise RecordValidationError(f'duplicate existing record at line {line_number}: {key}')
            keys.add(key)
        return keys

    def append(self, record: Mapping[str, Any]) -> bool:
        validated = validate_record(record)
        key = record_key(validated)
        if key in self._keys:
            return False
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(validated, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
        self._keys.add(key)
        return True
