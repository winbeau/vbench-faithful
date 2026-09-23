"""Task contracts, record construction and validation.

Model-free (standard library only) so the base development environment can run
every check without torch or network access. The same functions are reused by
the cleaning scripts, the teacher ingest path and the training data loader, so
the JSONL contract has exactly one definition.

A dataset record keeps provenance in the wrapper; the *target* stays minimal,
because that is what the student model must produce.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

TASKS: tuple[str, ...] = ("spatial", "action", "objects", "scene")
SCENE_LABELS: tuple[str, ...] = ("supported", "contradicted", "insufficient")

# Relations with a deterministic geometry predicate in the audit backend.
DIRECTION_RELATIONS: tuple[str, ...] = ("left", "right", "above", "below")
# Accepted but not backed by the direction predicate: weak supervision only.
WEAK_RELATIONS: tuple[str, ...] = (
    "on",
    "under",
    "beneath",
    "underneath",
    "near",
    "next to",
    "beside",
    "behind",
    "in front of",
    "over",
    "inside",
    "outside",
    "between",
)
ALL_RELATIONS: tuple[str, ...] = DIRECTION_RELATIONS + WEAK_RELATIONS

INPUT_KEYS: Mapping[str, tuple[str, ...]] = {
    "spatial": ("prompt",),
    "action": ("prompt",),
    "objects": ("prompt",),
    "scene": ("prompt", "caption"),
}
TARGET_KEYS: Mapping[str, tuple[str, ...]] = {
    "spatial": ("relationships",),
    "action": ("actions",),
    "objects": ("entities",),
    "scene": (),
}

QUALITY_LEVELS: tuple[str, ...] = (
    "engineering_only",
    "synthetic_weak",
    "teacher_candidate_unreviewed",
    "teacher_reviewed",
    "llm_annotated_arbitrated",
    "gold",
)

# Action names outside Kinetics-400 are reported as this sentinel instead of
# being guessed into a class (decision B2). Downstream treats it as "no K400
# match", never as a category to score against.
ACTION_OTHER = "other"

# Leading words dropped when canonicalising an object name (decision B3).
ENTITY_DETERMINERS = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "some", "several", "many",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "both", "all", "another", "each", "every", "no", "his", "her", "its", "their",
        "our", "your", "my", "there", "few", "lots", "pair",
    }
)
TEACHER_QUALITY = "teacher_candidate_unreviewed"

TOP_LEVEL_KEYS = frozenset(
    {"sample_id", "task", "input", "target", "source", "source_id", "group_id", "quality", "meta"}
)
RELATIONSHIP_KEYS = frozenset({"subject", "relation", "object"})

WORD_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")
PUNCT_EDGE = " \t\n.,;:!?\"'()[]{}"


class RecordError(ValueError):
    """Raised when a record cannot be canonicalised; carries error codes."""

    def __init__(self, errors: Sequence[str], sample_id: str | None = None) -> None:
        self.errors = tuple(errors)
        self.sample_id = sample_id
        super().__init__(f"invalid record {sample_id or '<unknown>'}: {', '.join(self.errors)}")


def normalize_space(text: str) -> str:
    return WS_RE.sub(" ", str(text)).strip()


def word_count(text: str) -> int:
    """Whitespace-token count; the engineering definition fixed by S4."""
    stripped = normalize_space(text)
    return len(stripped.split(" ")) if stripped else 0


def char_count(text: str) -> int:
    return len(str(text))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Byte-stable serialisation used for hashing and for model targets."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_phrase(text: str) -> str:
    """Lowercase, collapse whitespace and strip edge punctuation."""
    return normalize_space(str(text)).strip(PUNCT_EDGE).lower()


def phrase_tokens(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


# Singular nouns that merely end in "s": the plural rule must not strip them.
SINGULAR_S_WORDS = frozenset(
    {
        "bus", "circus", "glass", "grass", "dress", "class", "gas", "cactus", "virus", "canvas",
        "lens", "house", "mouse", "course", "horse", "plus", "status", "analysis", "business",
        "campus", "corps", "atlas", "basis", "chaos", "chess", "cross", "iris", "justice",
        "office", "practice", "promise", "purpose", "release", "service", "surface", "tennis",
    }
)

IRREGULAR_PLURALS = {
    "leaves": "leaf", "wolves": "wolf", "knives": "knife", "lives": "life", "wives": "wife",
    "shelves": "shelf", "scarves": "scarf", "halves": "half", "loaves": "loaf", "thieves": "thief",
    "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth", "children": "child",
    "people": "person", "men": "man", "women": "woman", "oxen": "ox", "indices": "index",
}


def stem(token: str) -> str:
    """Very small English stemmer for grounding checks (not a tokenizer).

    Handles plural/singular and gerund/finite verb pairs such as
    ``plays``/``playing``/``play`` so a canonical action label can be grounded in
    a prompt that uses a different verb form. Doubled consonants are only reduced
    after stripping ``-ing`` (so ``grass`` keeps its double s).
    """
    result = token
    if result in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[result]
    if result in SINGULAR_S_WORDS:
        return result
    if len(result) > 4 and result.endswith("ies"):
        result = result[:-3] + "y"
    elif len(result) > 4 and result.endswith("ves"):
        result = result[:-3] + "f"
    elif len(result) > 5 and result.endswith("ing"):
        result = result[:-3]
        if len(result) > 3 and result[-1] == result[-2] and result[-1] not in "aeiou":
            result = result[:-1]
    elif len(result) > 3 and result.endswith("es") and not result.endswith("ses"):
        result = result[:-2]
    elif len(result) > 3 and result.endswith("s") and not result.endswith("ss"):
        result = result[:-1]
    return result


def singular(token: str) -> str:
    """Backwards-compatible alias for :func:`stem`."""
    return stem(token)


def span_support(prompt: str, phrase: str) -> str:
    """How a phrase is grounded in the prompt.

    Returns one of ``exact`` (case-insensitive word-boundary substring),
    ``morph`` (same with simple singular fallback per token), ``tokens`` (all
    content tokens present separately, order ignored), ``partial`` (at least
    60% of content tokens present *and* the head token present) or ``none``.

    ``none`` is a hard failure; ``partial`` is a review warning, because
    canonical labels (for example K400 names) do not have to appear verbatim in
    a prompt while a hallucinated object loses its head token entirely.
    """
    prompt_norm = normalize_space(prompt).lower()
    phrase_norm = normalize_phrase(phrase)
    if not phrase_norm:
        return "none"
    if re.search(r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])", prompt_norm):
        return "exact"
    phrase_stems = [stem(t) for t in phrase_tokens(phrase_norm)]
    prompt_stems = [stem(t) for t in phrase_tokens(prompt_norm)]
    if phrase_stems and _sublist(phrase_stems, prompt_stems):
        return "morph"
    if phrase_stems and all(t in prompt_stems for t in phrase_stems):
        return "tokens"
    if phrase_stems:
        present = sum(1 for t in phrase_stems if t in prompt_stems)
        head = phrase_stems[-1]
        if head in prompt_stems and present * 5 >= len(phrase_stems) * 3:
            return "partial"
    return "none"


def _sublist(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for index, token in enumerate(haystack):
        if token == first and list(haystack[index : index + len(needle)]) == list(needle):
            return True
    return False


def canonical_entity_name(name: str) -> str:
    """Canonical object name: no leading determiner, singular head, lowercase.

    Deterministic and auditable; the raw mention is kept elsewhere when needed.
    """
    tokens = normalize_phrase(name).split(" ")
    while tokens and tokens[0] in ENTITY_DETERMINERS:
        tokens = tokens[1:]
    if not tokens:
        return normalize_phrase(name)
    tokens[-1] = singular(tokens[-1])
    return " ".join(tokens)


HEAD_NOUN_CUTS = frozenset(
    {
        "in", "on", "at", "with", "of", "near", "behind", "under", "beside", "by", "from", "to",
        "holding", "wearing", "carrying", "riding", "sitting", "standing", "walking", "looking",
        "next", "inside", "outside", "over", "above", "below", "between", "and", "that", "who",
        "sleeping", "lying", "flying", "playing", "running", "jumping", "sitting", "standing",
    }
)


def head_noun(phrase: str) -> str:
    """Cheap deterministic head noun of a region phrase.

    ``a man in a blue shirt`` -> ``man``; ``a stack of books`` -> ``stack``;
    ``two wooden benches`` -> ``bench``. Used for weak object targets only.
    """
    tokens = normalize_phrase(phrase).split(" ")
    while tokens and tokens[0] in ENTITY_DETERMINERS:
        tokens = tokens[1:]
    head: list[str] = []
    for token in tokens:
        if token in HEAD_NOUN_CUTS:
            break
        head.append(token)
    if not head:
        head = tokens[:1]
    if not head:
        return normalize_phrase(phrase)
    head[-1] = singular(head[-1])
    return " ".join(head)


def make_sample_id(task: str, source: str, source_id: str, prompt: str) -> str:
    digest = sha256_text(canonical_json({"task": task, "source": source, "source_id": str(source_id), "prompt": prompt}))[:12]
    return f"{task}-{source}-{source_id}-{digest}"


def length_bucket(words: int) -> str:
    if words <= 14:
        return "1-14"
    if words <= 49:
        return "15-49"
    if words <= 99:
        return "50-99"
    if words <= 200:
        return "100-200"
    if words <= 400:
        return "201-400"
    return "401+"


def dedup_preserving_order(items: Iterable[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for item in items:
        key = normalize_phrase(item)
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(normalize_space(item))
    return kept, dropped


def canonical_target(
    task: str,
    target: Any,
    *,
    action_resolver: Callable[[str], str | None] | None = None,
) -> tuple[Any, list[str], list[str]]:
    """Normalise a target. Returns ``(target, errors, warnings)``."""
    errors: list[str] = []
    warnings: list[str] = []
    if task == "scene":
        label = target.get("label") if isinstance(target, Mapping) else target
        label = normalize_phrase(label) if isinstance(label, str) else ""
        if label not in SCENE_LABELS:
            errors.append("scene_label")
            return None, errors, warnings
        return label, errors, warnings
    if not isinstance(target, Mapping):
        errors.append("schema_types")
        return None, errors, warnings
    keys = set(target)
    expected = set(TARGET_KEYS[task])
    if keys != expected:
        errors.append("schema_keys")
        return None, errors, warnings
    if task == "spatial":
        rows = target["relationships"]
        if not isinstance(rows, list):
            errors.append("schema_types")
            return None, errors, warnings
        kept: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != RELATIONSHIP_KEYS:
                errors.append("schema_types")
                continue
            subject = normalize_space(row["subject"])
            obj = normalize_space(row["object"])
            relation = normalize_phrase(row["relation"])
            if not subject or not obj or not relation:
                errors.append("empty_value")
                continue
            if normalize_phrase(subject) == normalize_phrase(obj):
                errors.append("self_relation")
                continue
            if relation not in ALL_RELATIONS:
                errors.append("relation_not_allowed")
                continue
            if relation not in DIRECTION_RELATIONS:
                warnings.append("weak_relation")
            key = (normalize_phrase(subject), relation, normalize_phrase(obj))
            if key in seen:
                warnings.append("duplicate_relation")
                continue
            seen.add(key)
            kept.append({"subject": subject, "relation": relation, "object": obj})
        if errors and not kept:
            return None, errors, warnings
        return {"relationships": kept}, errors, warnings
    if task == "action":
        rows = target["actions"]
        if not isinstance(rows, list) or not all(isinstance(x, str) for x in rows):
            errors.append("schema_types")
            return None, errors, warnings
        kept, dropped = dedup_preserving_order(rows)
        if dropped:
            warnings.append("duplicate_value")
        resolved: list[str] = []
        for name in kept:
            if normalize_phrase(name) == ACTION_OTHER:
                warnings.append("other_action")
                resolved.append(ACTION_OTHER)
                continue
            if action_resolver is not None:
                canonical = action_resolver(name)
                if canonical is None:
                    errors.append("action_not_in_k400")
                    continue
                if normalize_phrase(canonical) != normalize_phrase(name):
                    warnings.append("action_alias_used")
                resolved.append(canonical)
            else:
                resolved.append(name)
        kept, _ = dedup_preserving_order(resolved)
        if errors and not kept:
            return None, errors, warnings
        return {"actions": kept}, errors, warnings
    # objects
    rows = target["entities"]
    if not isinstance(rows, list) or not all(isinstance(x, str) for x in rows):
        errors.append("schema_types")
        return None, errors, warnings
    kept, dropped = dedup_preserving_order(rows)
    if dropped:
        warnings.append("duplicate_value")
    if not kept:
        errors.append("empty_value")
        return None, errors, warnings
    return {"entities": kept}, errors, warnings


def validate_input(task: str, payload: Any) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {}, ["schema_types"]
    expected = set(INPUT_KEYS[task])
    if set(payload) != expected:
        errors.append("schema_keys")
        return {}, errors
    clean: dict[str, str] = {}
    for key in INPUT_KEYS[task]:
        value = payload[key]
        if not isinstance(value, str) or not normalize_space(value):
            errors.append("empty_value")
            continue
        clean[key] = normalize_space(value)
    return clean, errors


@dataclass(frozen=True)
class Validation:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_record(
    record: Mapping[str, Any],
    *,
    action_resolver: Callable[[str], str | None] | None = None,
    check_spans: bool = True,
) -> Validation:
    """Validate a dataset record; returns error/warning codes (never raises)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(record, Mapping):
        return Validation(("schema_types",), ())
    unknown = set(record) - TOP_LEVEL_KEYS
    if unknown:
        errors.append("unknown_top_level_key")
    missing = {"sample_id", "task", "input", "target", "source", "source_id", "group_id", "quality"} - set(record)
    if missing:
        errors.append("missing_required_key")
        return Validation(tuple(errors), tuple(warnings))
    task = record["task"]
    if task not in TASKS:
        errors.append("unknown_task")
        return Validation(tuple(errors), tuple(warnings))
    quality = record["quality"]
    if quality not in QUALITY_LEVELS:
        errors.append("unknown_quality")
    for key in ("sample_id", "source", "source_id", "group_id"):
        if not isinstance(record[key], str) or not normalize_space(record[key]):
            errors.append("empty_value")
    clean_input, input_errors = validate_input(task, record["input"])
    errors.extend(input_errors)
    target, target_errors, target_warnings = canonical_target(task, record["target"], action_resolver=action_resolver)
    errors.extend(target_errors)
    warnings.extend(target_warnings)
    if target is None:
        return Validation(tuple(errors), tuple(warnings))
    if check_spans:
        prompt = clean_input.get("prompt", "")
        if task == "spatial":
            for row in target["relationships"]:
                for field in ("subject", "object"):
                    support = span_support(prompt, row[field])
                    if support == "none":
                        errors.append(f"unsupported_span:{field}")
                    elif support == "partial":
                        warnings.append(f"partial_span:{field}")
        elif task == "action":
            for name in target["actions"]:
                if normalize_phrase(name) == ACTION_OTHER:
                    continue
                support = span_support(prompt, name)
                if support == "none":
                    errors.append("unsupported_span:action")
                elif support == "partial":
                    warnings.append("partial_span:action")
        elif task == "objects":
            # Gold records keep the raw mentions; canonical names need not appear verbatim.
            meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
            mentions = meta.get("entity_mentions")
            span_checked = bool(mentions)
            for name in (mentions or target["entities"]):
                support = span_support(prompt, str(name))
                if support == "none":
                    # LLM-gold rows were span-verified on their raw mentions when they
                    # were created; a canonical form that no longer matches literally
                    # ("cosmic circus" -> "cosmic circu") must not block training.
                    if record.get("quality") == "llm_annotated_arbitrated":
                        warnings.append("span_unverifiable:entity")
                    else:
                        errors.append("unsupported_span:entity")
                elif support == "partial":
                    warnings.append("partial_span:entity")
    return Validation(tuple(errors), tuple(warnings))


def canonicalize_record(
    record: Mapping[str, Any],
    *,
    action_resolver: Callable[[str], str | None] | None = None,
    check_spans: bool = True,
) -> dict[str, Any]:
    """Return a normalised copy of ``record`` or raise :class:`RecordError`."""
    validation = validate_record(record, action_resolver=action_resolver, check_spans=check_spans)
    if not validation.ok:
        raise RecordError(validation.errors, str(record.get("sample_id", "")) or None)
    task = record["task"]
    clean_input, _ = validate_input(task, record["input"])
    target, _, _ = canonical_target(task, record["target"], action_resolver=action_resolver)
    out: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "task": task,
        "input": clean_input,
        "target": target,
        "source": record["source"],
        "source_id": str(record["source_id"]),
        "group_id": record["group_id"],
        "quality": record["quality"],
    }
    meta = record.get("meta")
    if meta is not None:
        if not isinstance(meta, Mapping):
            raise RecordError(("schema_types",), out["sample_id"])
        out["meta"] = dict(meta)
    return out


def make_record(
    *,
    task: str,
    sample_input: Mapping[str, str],
    target: Any,
    source: str,
    source_id: str,
    group_id: str,
    quality: str,
    meta: Mapping[str, Any] | None = None,
    action_resolver: Callable[[str], str | None] | None = None,
    check_spans: bool = True,
) -> dict[str, Any]:
    """Build a validated record; raises :class:`RecordError` when invalid."""
    if task not in TASKS:
        raise RecordError(("unknown_task",))
    record: dict[str, Any] = {
        "sample_id": make_sample_id(task, source, source_id, sample_input.get("prompt", "")),
        "task": task,
        "input": dict(sample_input),
        "target": target,
        "source": source,
        "source_id": str(source_id),
        "group_id": group_id,
        "quality": quality,
    }
    if meta is not None:
        record["meta"] = dict(meta)
    return canonicalize_record(record, action_resolver=action_resolver, check_spans=check_spans)


def record_key(record: Mapping[str, Any]) -> str:
    """Exact-duplicate key: task + input + target, ignoring provenance."""
    return sha256_text(
        canonical_json({"task": record["task"], "input": record["input"], "target": record["target"]})
    )


def add_length_meta(record: dict[str, Any]) -> dict[str, Any]:
    """Attach engineering length statistics (S4) without touching the target."""
    prompt = record["input"].get("prompt", "")
    caption = record["input"].get("caption", "")
    words = word_count(prompt)
    meta = dict(record.get("meta", {}))
    meta.update(
        {
            "prompt_words": words,
            "prompt_chars": char_count(prompt),
            "caption_words": word_count(caption),
            "length_bucket": length_bucket(words),
        }
    )
    record["meta"] = meta
    return record


def write_jsonl(path: Any, records: Iterable[Mapping[str, Any]]) -> int:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
            count += 1
    return count


def read_jsonl(path: Any) -> list[dict[str, Any]]:
    from pathlib import Path

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
