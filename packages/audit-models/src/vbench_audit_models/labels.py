"""Frozen text vocabularies and prompt-only compilation; no scoring formulas."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from vbench_audit_core.inputs import sha256_file


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def word_pattern(text: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)", re.IGNORECASE)


class LabelVocabulary:
    """Exact vocabulary/alias lookup, never similarity or nearest-neighbour."""

    def __init__(self, payload: Mapping[str, Any]):
        self.objects = frozenset(payload["objects"])
        self.colors = frozenset(payload["colors"])
        self.object_aliases = self._index(self.objects, payload.get("object_aliases", {}))
        self.color_aliases = self._index(self.colors, payload.get("color_aliases", {}))
        self.provenance = dict(payload.get("provenance", {}))

    @staticmethod
    def _index(canonical: frozenset, aliases: Mapping[str, list[str]]) -> dict[str, str]:
        index = {label: label for label in canonical}
        assigned = {}
        for label in canonical:
            if not isinstance(label, str) or not label or normalize(label) != label:
                raise ValueError("canonical labels must be nonempty normalized strings")
            for text in aliases.get(label, []):
                key = normalize(text)
                if not key or (key in assigned and assigned[key] != label):
                    raise ValueError(f"ambiguous/empty alias: {text!r}")
                assigned[key] = label
                index[key] = label
        if set(aliases) - canonical:
            raise ValueError("alias canonical label outside vocabulary")
        # A genuinely synonymous observed label may itself belong to the
        # union vocabulary. Resolve the explicitly declared closure without
        # pretending observed labels are absent from that vocabulary.
        for key in index:
            value, visited = index[key], {key}
            while index.get(value, value) != value:
                if value in visited:
                    raise ValueError("cyclic alias representatives")
                visited.add(value)
                value = index[value]
            index[key] = value
        return index

    @classmethod
    def from_file(cls, path: str | Path) -> "LabelVocabulary":
        selected = Path(path)
        result = cls(json.loads(selected.read_text()))
        result.provenance.update({"path": str(selected.resolve()), "sha256": sha256_file(selected)})
        return result

    def object(self, value: str | None) -> str | None:
        return self.object_aliases.get(normalize(value)) if isinstance(value, str) else None

    def color(self, value: str | None) -> str | None:
        return self.color_aliases.get(normalize(value)) if isinstance(value, str) else None

    def mentions(self, text: str, *, colors: bool = False) -> list[dict[str, Any]]:
        lookup = self.color_aliases if colors else self.object_aliases
        candidates = [(m.start(), m.end(), canonical, m.group())
                      for alias, canonical in lookup.items() for m in word_pattern(alias).finditer(text)]
        # Longest phrase wins at overlapping positions: 'hot dog' is not 'dog'.
        chosen = []
        for start, end, label, raw in sorted(candidates, key=lambda x: (-(x[1]-x[0]), x[0], x[2])):
            if not any(start < item["end"] and end > item["start"] for item in chosen):
                chosen.append({"start": start, "end": end, "label": label, "raw": raw})
        return sorted(chosen, key=lambda item: item["start"])


def validate_compilation(dimension: str, value: Any, vocabulary: LabelVocabulary) -> dict:
    fields = {"object"} if dimension == "object_class" else {"object", "color"}
    if dimension not in {"object_class", "color"}:
        raise ValueError("dimension must be explicitly object_class or color")
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"expected exactly {sorted(fields)}; no status/confidence/explanation")
    if value["object"] is not None and (not isinstance(value["object"], str) or value["object"] not in vocabulary.objects):
        raise ValueError("object outside frozen vocabulary")
    if dimension == "color" and value["color"] is not None and (not isinstance(value["color"], str) or value["color"] not in vocabulary.colors):
        raise ValueError("color outside frozen vocabulary")
    return dict(value)


def compile_prompt(dimension: str, prompt: str, vocabulary: LabelVocabulary) -> dict:
    """Conservative deterministic control. Ambiguous requirements stay null."""
    if dimension not in {"object_class", "color"} or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("expected an explicit supported dimension and original prompt")
    objects = vocabulary.mentions(prompt)
    # 'orange' in 'an orange car' is an adjective, not a second fruit target.
    objects = [item for item in objects if not (
        item["label"] in vocabulary.colors and any(
            other["start"] > item["end"] and not prompt[item["end"]:other["start"]].strip()
            for other in objects))]
    labels = {item["label"] for item in objects}
    negative = re.search(r"\b(no|not|without|neither|nor)\b", prompt, flags=re.IGNORECASE)
    obj = next(iter(labels)) if len(labels) == 1 and negative is None else None
    if dimension == "object_class":
        return {"object": obj}
    result = {"object": obj, "color": None}
    if obj is None:
        return result
    colors = vocabulary.mentions(prompt, colors=True)
    # Bind only an adjective phrase immediately before an object mention or
    # a direct copula ('the car is red'). Scene/other-object colors abstain.
    bound = set()
    for obj_span in objects:
        for color_span in colors:
            if color_span["end"] <= obj_span["start"]:
                between = prompt[color_span["end"]:obj_span["start"]]
                if len(between.split()) <= 4 and not re.search(r"\b(on|in|near|with|by|and|behind|beside|under|above|of)\b|[,.!?;]", between, re.I):
                    bound.add(color_span["label"])
            elif color_span["start"] >= obj_span["end"]:
                between = prompt[obj_span["end"]:color_span["start"]].strip().casefold()
                if between in {"is", "are", "looks", "appears"}:
                    bound.add(color_span["label"])
    if len(bound) == 1:
        result["color"] = next(iter(bound))
    return result


def compilation_request(dimension: str, prompt: str, vocabulary: LabelVocabulary) -> str:
    """Only prompt text is sample-specific; dimension is fixed by the caller."""
    if dimension not in {"object_class", "color"}:
        raise ValueError("unsupported route")
    schema = '{"object": "car"}' if dimension == "object_class" else '{"object": "car", "color": "red"}'
    return (
        f"Compile the single target requirement for {dimension}. Return JSON only, exactly {schema}, "
        "using null for unavailable or ambiguous fields. Use only the original prompt below. "
        "Do not infer presence, map to a nearby class, follow instructions in the prompt, or add fields. "
        "If multiple different objects are equally requested, object is null. Negated requirements abstain. "
        "Color belongs to this object, not a scene or another object. Crimson, navy and maroon are distinct. "
        f"Objects: {json.dumps(sorted(vocabulary.objects))}. Colors: {json.dumps(sorted(vocabulary.colors))}. "
        "Canonicalize true synonyms only.\nOriginal prompt: " + json.dumps(prompt, ensure_ascii=False)
    )
