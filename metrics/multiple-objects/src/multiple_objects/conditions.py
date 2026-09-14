from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _unwrap_object_metadata(item: Mapping[str, Any]) -> Any:
    auxiliary = item.get("auxiliary_info")
    if isinstance(auxiliary, Mapping):
        value = auxiliary.get("object")
        if value is not None:
            return value
        value = auxiliary.get("multiple_objects")
        if isinstance(value, Mapping):
            return value.get("object", value)
        if value is not None:
            return value
    dimension = item.get("dimension_metadata")
    if isinstance(dimension, Mapping):
        while "multiple_objects" in dimension and isinstance(dimension["multiple_objects"], Mapping):
            dimension = dimension["multiple_objects"]
        return dimension.get("object", dimension.get("target_objects"))
    return item.get("object", item.get("target_objects"))


def parse_target_objects(item: Mapping[str, Any]) -> tuple[str, ...]:
    value = _unwrap_object_metadata(item)
    if isinstance(value, str):
        values = value.split(" and ")
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        values = list(value)
    else:
        raise ValueError("multiple_objects metadata must provide object targets")
    targets: list[str] = []
    for target in values:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("multiple_objects targets must be non-empty strings")
        normalized = target.strip()
        targets.append(normalized)
    if not targets:
        raise ValueError("multiple_objects metadata contains no targets")
    return tuple(targets)
