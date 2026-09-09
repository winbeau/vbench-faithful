from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import ConditionSet, SemanticCondition


def _explicit_value(metadata: Mapping[str, Any]) -> tuple[bool, Any]:
    if "semantic_conditions" in metadata:
        return True, metadata["semantic_conditions"]
    dimension = metadata.get("dimension_metadata")
    if isinstance(dimension, Mapping) and "semantic_conditions" in dimension:
        return True, dimension["semantic_conditions"]
    return False, None


def fallback_conditions(
    prompt: str,
    warning: str | None = None,
    *,
    reason: str = "semantic_conditions_missing",
) -> ConditionSet:
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    warnings = (warning,) if warning else ()
    # Preserve the exact official prompt. Stripping here would make the fallback
    # condition use different ViCLIP text input than the global branch.
    return ConditionSet(
        (SemanticCondition(text=prompt, condition_type=None, source="fallback"),),
        "fallback",
        warnings,
        reason,
    )


def normalize_conditions(metadata: Mapping[str, Any], prompt: str) -> ConditionSet:
    """Prefer explicit conditions; otherwise conservatively keep the full prompt."""
    present, raw = _explicit_value(metadata)
    if not present:
        return fallback_conditions(prompt)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return fallback_conditions(
            prompt,
            "semantic_conditions must be a list; used full-prompt fallback",
            reason="semantic_conditions_malformed",
        )

    normalized: list[SemanticCondition] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            text = item.strip()
            condition_type = None
        elif isinstance(item, Mapping):
            text_value = item.get("text")
            type_value = item.get("type", item.get("condition_type"))
            if not isinstance(text_value, str) or (type_value is not None and not isinstance(type_value, str)):
                return fallback_conditions(
                    prompt,
                    f"semantic_conditions[{index}] is malformed; used full-prompt fallback",
                    reason="semantic_conditions_malformed",
                )
            text = text_value.strip()
            condition_type = (type_value.strip() or None) if isinstance(type_value, str) else None
        else:
            return fallback_conditions(
                prompt,
                f"semantic_conditions[{index}] is malformed; used full-prompt fallback",
                reason="semantic_conditions_malformed",
            )
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(SemanticCondition(text=text, condition_type=condition_type, source="provided"))

    if not normalized:
        return fallback_conditions(
            prompt,
            "semantic_conditions contained no usable text; used full-prompt fallback",
            reason="semantic_conditions_empty_or_no_usable_text",
        )
    return ConditionSet(tuple(normalized), "provided")
