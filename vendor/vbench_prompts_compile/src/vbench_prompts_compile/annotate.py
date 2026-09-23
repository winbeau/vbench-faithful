"""LLM annotation with arbitration (decision B1) and vision scene labels (B5).

Pipeline per item:

1. pass A  -- deterministic instruction, temperature 0;
2. pass B  -- an independent second reviewer (same rules, different wording and
   temperature 0.6) so genuine disagreement is possible;
3. arbitration -- only when A and B disagree: a third call sees both answers and
   must decide, with the same output schema.

Outputs carry ``quality="llm_annotated_arbitrated"`` and full provenance (provider,
model, pass answers, tokens, latency). These are **LLM labels**: they are stronger
than the earlier template weak supervision because they are schema-checked,
span-checked and arbitrated, but they are still not human gold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .llm import ChatClient, ChatResponse, LLMError
from .records import (
    ACTION_OTHER,
    ALL_RELATIONS,
    SCENE_LABELS,
    canonical_entity_name,
    canonical_json,
    canonical_target,
    make_record,
    normalize_phrase,
    normalize_space,
    sha256_text,
)

ANNOTATION_QUALITY = "llm_annotated_arbitrated"

SYSTEM_TEXT = (
    "You label one text-to-video prompt for a research dataset. Use only what the prompt states. "
    "Never guess, never add objects, actions or relations that are not written, never explain. "
    "Answer with a single JSON object and nothing else."
)
SYSTEM_TEXT_REVIEWER = (
    "You are a second independent annotator for the same text-to-video prompt. "
    "Apply the rules exactly as written; do not try to agree with anyone. "
    "Answer with a single JSON object and nothing else."
)
SYSTEM_VISION = (
    "You annotate a single video frame for a scene-verification dataset. "
    "Judge the frame itself; the automatic caption may be wrong. Never explain. "
    "Answer with a single JSON object and nothing else."
)

INSTRUCTIONS: Mapping[str, str] = {
    "spatial": (
        "Extract every explicit spatial relation between two named entities that the prompt states. "
        "Allowed relation words: left, right, above, below, on, under, beneath, underneath, near, "
        "next to, beside, behind, in front of, over, inside, outside, between. "
        "Use the entity words as they appear in the prompt. "
        'Answer exactly {"relationships":[{"subject":"...","relation":"...","object":"..."}]}. '
        'If the prompt states no spatial relation between two named entities, answer {"relationships":[]}. '
        "Do not turn an intransitive phrase (\"a cloud floats above\") into a relation."
    ),
    "action": (
        "List every action performed by a person, animal or moving object in the prompt. "
        "Write each action as a short verb phrase using the prompt's own words "
        '(for example "playing guitar", "climbing a tall tree", "assembling furniture"). '
        "Do not decide whether the phrase belongs to any fixed vocabulary and do not output objects. "
        'Answer exactly {"actions":["..."]}. If the prompt states no action, answer {"actions":[]}.'
    ),
    "objects": (
        "Extract the concrete physical things the prompt explicitly mentions: people, animals, objects "
        "and visible scene elements. Rules: output a short canonical noun phrase in lowercase, singular, "
        "without articles or possessives (cat, dog, bicycle, coffee mug, stack of books); "
        "never add things that are only implied; exclude camera and post-production vocabulary "
        "(camera, close-up, tracking shot, 3D animation, depth of field, cinematic). "
        'Answer exactly {"entities":["..."]}.'
    ),
    "scene": (
        "You are given one frame of a generated video, the text prompt it was generated from and an "
        "automatic caption of that frame. Work in two steps. "
        "Step 1: decide whether the prompt requires a particular scene (a place or environment such as a "
        "beach, ocean, forest, street, bathroom, office). If the prompt does not require any scene, answer "
        '{"label":"insufficient"} immediately, whatever the frame shows. '
        "Step 2: only if a scene is required, compare it with the frame. "
        'Answer exactly {"label":"supported"} or {"label":"contradicted"} or {"label":"insufficient"}. '
        "supported: the frame clearly shows the scene the prompt requires. "
        "contradicted: the frame clearly shows a different scene (for example the prompt requires an ocean "
        "and the frame shows a river with a canoe). "
        "insufficient: a scene is required but the frame does not show enough to decide."
    ),
}


@dataclass
class Item:
    item_id: str
    task: str
    prompt: str
    group_id: str
    bucket: str
    caption: str | None = None
    image: str | None = None
    source: str = "unknown"
    source_id: str = "0"
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "task": self.task,
            "prompt": self.prompt,
            "caption": self.caption,
            "image": self.image,
            "group_id": self.group_id,
            "bucket": self.bucket,
            "source": self.source,
            "source_id": self.source_id,
            "prompt_sha256": sha256_text(self.prompt),
            **({"meta": self.meta} if self.meta else {}),
        }


def build_user_text(item: Item, *, reviewer: bool = False, instruction_override: str | None = None) -> str:
    instruction = instruction_override or INSTRUCTIONS[item.task]
    if reviewer:
        instruction = (
            f"{instruction}\nSecond reviewer note: re-read the prompt before answering; "
            "if the first reading produced no item, only report an empty result when the prompt truly "
            "contains none."
        )
    parts = [instruction]
    parts.append(f"Prompt: {normalize_space(item.prompt)}")
    if item.task == "scene":
        if item.caption:
            parts.append(f"Automatic caption: {normalize_space(item.caption)}")
        if item.image and instruction_override is None:
            parts.append("The image above is the frame to judge.")
    return "\n\n".join(parts)


def parse_annotation(task: str, text: str) -> tuple[Any | None, list[str]]:
    """Parse one annotation answer into a target-shaped value."""
    from .teacher import parse_teacher_text

    return parse_teacher_text(task, text)


def validate_annotation(
    task: str,
    value: Any,
    item: Item,
    *,
    action_resolver: Callable[[str], str | None] | None,
) -> tuple[Any | None, list[str], list[str]]:
    """Schema + vocabulary + span validation with an auditable sanitisation trail.

    Returns ``(target, errors, notes)``. Unsupported mentions are **dropped and
    recorded** in ``notes`` rather than silently kept (the project rule forbids
    silently claiming a complete target set). A pass with nothing left to keep is
    rejected so arbitration can resolve it.

    Action answers that cannot be mapped to Kinetics-400 become the ``other``
    sentinel (decision B2) instead of being dropped or guessed.
    """
    from .records import span_support

    errors: list[str] = []
    notes: list[str] = []
    if value is None:
        return None, ["empty_value"], notes

    if task == "objects":
        if not isinstance(value, Mapping) or not isinstance(value.get("entities"), list):
            return None, ["schema_types"], notes
        kept: list[str] = []
        for raw in value["entities"]:
            mention = str(raw)
            if span_support(item.prompt, mention) == "none":
                notes.append(f"dropped_unsupported_entity:{canonical_entity_name(mention)}")
                continue
            kept.append(canonical_entity_name(mention))
        if not kept:
            return None, ["no_supported_entity"], notes
        target, target_errors, _ = canonical_target("objects", {"entities": kept})
        notes.extend(target_errors)
        return (None, ["schema_types"], notes) if target is None else (target, [], notes)

    if task == "spatial":
        raw_rows = value.get("relationships") if isinstance(value, Mapping) else None
        if not isinstance(raw_rows, list):
            return None, ["schema_types"], notes
        kept_rows: list[dict[str, str]] = []
        for row in raw_rows:
            if not isinstance(row, Mapping):
                notes.append("dropped_malformed_relation")
                continue
            relation = normalize_phrase(str(row.get("relation", "")))
            subject = normalize_space(str(row.get("subject", "")))
            obj = normalize_space(str(row.get("object", "")))
            if not relation or not subject or not obj:
                notes.append("dropped_malformed_relation")
                continue
            if relation not in ALL_RELATIONS:
                notes.append(f"dropped_relation_word:{relation}")
                continue
            if span_support(item.prompt, subject) == "none" or span_support(item.prompt, obj) == "none":
                notes.append(f"dropped_unsupported_endpoint:{subject}|{obj}")
                continue
            kept_rows.append({"subject": subject, "relation": relation, "object": obj})
        if not kept_rows and raw_rows:
            return None, ["no_supported_relation"], notes
        target, target_errors, _ = canonical_target("spatial", {"relationships": kept_rows})
        return (None, ["schema_types"], notes) if target is None else (target, [], notes)

    if task == "action":
        raw_actions = value.get("actions") if isinstance(value, Mapping) else None
        if not isinstance(raw_actions, list) or not all(isinstance(x, str) for x in raw_actions):
            return None, ["schema_types"], notes
        resolved: list[str] = []
        for raw in raw_actions:
            name = normalize_space(raw)
            if not name:
                continue
            if normalize_phrase(name) == ACTION_OTHER:
                resolved.append(ACTION_OTHER)
                continue
            canonical = action_resolver(name) if action_resolver is not None else None
            if canonical is None:
                notes.append(f"mapped_to_other:{name}")
                resolved.append(ACTION_OTHER)
                continue
            if span_support(item.prompt, canonical) == "none":
                notes.append(f"dropped_unsupported_action:{canonical}")
                continue
            resolved.append(canonical)
        if not resolved and raw_actions:
            return None, ["no_supported_action"], notes
        target, target_errors, _ = canonical_target("action", {"actions": resolved}, action_resolver=action_resolver)
        return (None, ["schema_types"], notes) if target is None else (target, [], notes)

    # scene
    target, target_errors, _ = canonical_target("scene", value)
    if target is None:
        return None, target_errors or ["scene_label"], notes
    return target, [], notes


@dataclass
class AnnotationResult:
    item_id: str
    status: str  # accepted | arbitrated | rejected
    target: Any | None
    passes: list[dict[str, Any]]
    agreement: bool
    errors: list[str]
    usage: dict[str, int] = field(default_factory=dict)


def _call(
    client: ChatClient,
    item: Item,
    *,
    reviewer: bool,
    temperature: float,
    max_tokens: int,
    system: str,
    extra_user: str | None = None,
    instruction_override: str | None = None,
) -> ChatResponse:
    user = build_user_text(item, reviewer=reviewer, instruction_override=instruction_override)
    if extra_user:
        user = f"{user}\n\n{extra_user}"
    images = [item.image] if (item.image and item.task == "scene") else []
    return client.chat(user=user, system=system, images=images, max_tokens=max_tokens, temperature=temperature)


def annotate_item(
    client: ChatClient,
    item: Item,
    *,
    action_resolver: Callable[[str], str | None] | None = None,
    max_tokens: int = 384,
    reviewer_temperature: float = 0.6,
    charge: Callable[[Mapping[str, Any]], int] | None = None,
    instruction_override: str | None = None,
    system_override: str | None = None,
) -> AnnotationResult:
    """Two passes + arbitration; every provider call is charged by the caller."""
    passes: list[dict[str, Any]] = []
    answers: list[Any | None] = []
    valid: list[Any | None] = []
    default_system = SYSTEM_TEXT if item.task != "scene" else SYSTEM_VISION
    reviewer_system = SYSTEM_TEXT_REVIEWER if item.task != "scene" else SYSTEM_VISION
    if system_override:
        default_system = reviewer_system = system_override
    for index, (reviewer, temperature, system) in enumerate(((False, 0.0, default_system), (True, reviewer_temperature, reviewer_system))):
        entry: dict[str, Any] = {"pass": index + 1, "reviewer": reviewer, "temperature": temperature}
        if charge is not None:
            charge({"item_id": item.item_id, "pass": index + 1, "provider": client.provider, "model": client.model})
        try:
            response = _call(client, item, reviewer=reviewer, temperature=temperature, max_tokens=max_tokens, system=system, instruction_override=instruction_override)
        except LLMError as error:
            entry.update({"error": str(error)})
            passes.append(entry)
            answers.append(None)
            valid.append(None)
            continue
        parsed, parse_errors = parse_annotation(item.task, response.text)
        target, errors, notes = validate_annotation(item.task, parsed, item, action_resolver=action_resolver)
        entry.update(
            {
                **response.as_metadata(),
                "raw_sha256": sha256_text(response.text),
                "raw_value": parsed,
                "value": target,
                "errors": errors + parse_errors,
                "notes": notes,
            }
        )
        passes.append(entry)
        answers.append(target)
        valid.append(target if not errors else None)

    if valid[0] is not None and valid[1] is not None and canonical_json(valid[0]) == canonical_json(valid[1]):
        return AnnotationResult(item.item_id, "accepted", valid[0], passes, True, [], _usage(passes))

    # Anything else (disagreement, one invalid pass, both invalid) goes to arbitration.
    if True:
        if charge is not None:
            charge({"item_id": item.item_id, "pass": 3, "provider": client.provider, "model": client.model, "kind": "arbitration"})
        first_errors = passes[0].get("errors", [])
        second_errors = passes[1].get("errors", [])
        extra = (
            "Two annotators produced different or invalid answers.\n"
            f"Annotator A: {canonical_json(valid[0]) if valid[0] is not None else 'invalid'}  (problems: {', '.join(first_errors) or 'none'})\n"
            f"Annotator B: {canonical_json(valid[1]) if valid[1] is not None else 'invalid'}  (problems: {', '.join(second_errors) or 'none'})\n"
            "Produce the single correct answer for the rules above. Remember: a relation must be one of the allowed words, "
            f'an action that is not a Kinetics-400 class must be written as "{ACTION_OTHER}", and entities must be plain '
            "lowercase singular noun phrases taken from the prompt. Answer with the same JSON schema only."
        )
        try:
            response = _call(
                client,
                item,
                reviewer=False,
                temperature=0.0,
                max_tokens=max_tokens,
                system=system_override or (SYSTEM_TEXT if item.task != "scene" else SYSTEM_VISION),
                extra_user=extra,
                instruction_override=instruction_override,
            )
            parsed, parse_errors = parse_annotation(item.task, response.text)
            target, errors, notes = validate_annotation(item.task, parsed, item, action_resolver=action_resolver)
            passes.append({"pass": 3, "arbitration": True, **response.as_metadata(), "raw_sha256": sha256_text(response.text), "raw_value": parsed, "value": target, "errors": errors + parse_errors, "notes": notes})
            if target is not None and not errors:
                return AnnotationResult(item.item_id, "arbitrated", target, passes, False, [], _usage(passes))
            return AnnotationResult(item.item_id, "rejected", None, passes, False, errors or ["arbitration_failed"], _usage(passes))
        except LLMError as error:
            passes.append({"pass": 3, "arbitration": True, "error": str(error)})
            return AnnotationResult(item.item_id, "rejected", None, passes, False, ["arbitration_transport_error"], _usage(passes))

    survivor = valid[0] if valid[0] is not None else valid[1]
    if survivor is not None:
        other = passes[0] if valid[0] is None else passes[1]
        return AnnotationResult(item.item_id, "accepted", survivor, passes, False, list(other.get("errors", [])), _usage(passes))
    return AnnotationResult(item.item_id, "rejected", None, passes, False, [e for entry in passes for e in entry.get("errors", [])] or ["no_valid_pass"], _usage(passes))


def _usage(passes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "requests": len(passes),
        "prompt_tokens": sum(int(p.get("prompt_tokens", 0) or 0) for p in passes),
        "completion_tokens": sum(int(p.get("completion_tokens", 0) or 0) for p in passes),
    }


def result_to_record(result: AnnotationResult, item: Item, *, action_resolver: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    if result.target is None:
        raise ValueError(f"cannot build a record from a {result.status} result: {item.item_id}")
    sample_input = {"prompt": normalize_space(item.prompt)}
    if item.task == "scene":
        sample_input["caption"] = normalize_space(item.caption or "")
    raw_mentions: list[str] | None = None
    if item.task == "objects" and isinstance(result.target, Mapping):
        accepted = {canonical_json(entry["value"]) for entry in result.passes if entry.get("value") is not None}
        for entry in reversed(result.passes):
            raw = entry.get("raw_value")
            if isinstance(raw, Mapping) and isinstance(raw.get("entities"), list) and canonical_json(entry.get("value")) in accepted:
                raw_mentions = [str(x) for x in raw["entities"]]
                break
    return make_record(
        task=item.task,
        sample_input=sample_input,
        target=result.target,
        source=item.source,
        source_id=item.source_id,
        group_id=item.group_id,
        quality=ANNOTATION_QUALITY,
        meta={
            "annotation": "llm_two_pass_arbitrated",
            "status": result.status,
            "agreement": result.agreement,
            "sanitisation_notes": sorted({note for entry in result.passes for note in entry.get("notes", [])}),
            "prompt_sha256": sha256_text(item.prompt),
            **({"image": item.image} if item.image else {}),
            **({"caption_sha256": sha256_text(item.caption)} if item.caption else {}),
            **({"entity_mentions": raw_mentions} if raw_mentions else {}),
            **item.meta,
        },
        action_resolver=action_resolver,
        check_spans=False,
    )

SYSTEM_REWRITE = (
    "You rewrite text-to-video prompts for a research dataset. You must preserve every entity, action "
    "and spatial relation of the original text and only add compatible descriptive detail (setting, "
    "lighting, style, camera). Never remove, replace or contradict a detail. Answer with a single JSON object."
)


def rewrite_instruction(min_words: int, max_words: int) -> str:
    return (
        f"Rewrite the prompt so that it contains between {min_words} and {max_words} words. "
        "Keep every entity, action and spatial relation of the original; you may add compatible detail "
        "(setting, lighting, style, camera) but never remove, rename or contradict anything. "
        'Answer exactly {"prompt":"..."} with a single string and nothing else.'
    )


@dataclass
class RewriteResult:
    text: str | None
    words: int
    in_range: bool
    errors: list[str]
    usage: dict[str, int]
    attempts: int = 1
    last_text: str | None = None


def rewrite_prompt(
    client: ChatClient,
    prompt: str,
    *,
    min_words: int,
    max_words: int,
    max_tokens: int = 700,
    charge: Callable[[Mapping[str, Any]], int] | None = None,
    item_id: str = "rewrite",
    attempts: int = 3,
) -> RewriteResult:
    """Ask for a rewrite of a given length, correcting the model if it misses.

    Length is part of the experiment (S4 buckets), so an out-of-range answer is
    retried with explicit feedback instead of being accepted with a tolerance.
    """
    from .records import word_count

    usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
    previous: RewriteResult | None = None
    for attempt in range(1, attempts + 1):
        if charge is not None:
            charge({"item_id": item_id, "kind": "rewrite", "attempt": attempt, "provider": client.provider, "model": client.model})
        usage["requests"] += 1
        user = f"{rewrite_instruction(min_words, max_words)}\n\nOriginal prompt: {normalize_space(prompt)}"
        if previous is not None and previous.last_text:
            user += (
                f"\n\nYour previous answer had {previous.words} words, which is outside the allowed range "
                f"{min_words}-{max_words}. Rewrite again and count the words before answering.\n"
                f"Previous answer: {previous.last_text}"
            )
        try:
            response = client.chat(user=user, system=SYSTEM_REWRITE, max_tokens=max_tokens, temperature=0.4)
        except LLMError as error:
            return RewriteResult(None, 0, False, [f"transport:{error}"], usage, attempt, previous.last_text if previous else None)
        usage["prompt_tokens"] += response.prompt_tokens
        usage["completion_tokens"] += response.completion_tokens
        parsed, parse_errors = parse_annotation("objects", response.text)  # reuse tolerant JSON parsing
        text = None
        if isinstance(parsed, Mapping) and isinstance(parsed.get("prompt"), str):
            text = normalize_space(str(parsed["prompt"]))
        if text is None:
            previous = RewriteResult(None, 0, False, parse_errors or ["missing_prompt_field"], usage, attempt, None)
            continue
        words = word_count(text)
        previous = RewriteResult(text, words, min_words <= words <= max_words, [], usage, attempt, text)
        if previous.in_range:
            return previous
    if previous is not None:
        previous.errors = [f"word_count_out_of_range:{previous.words}"]
        previous.usage = usage
    return previous or RewriteResult(None, 0, False, ["no_attempt"], usage, attempts, None)


GATE_SYSTEM = (
    "You classify prompts for a scene-verification dataset. A prompt *requires a scene* when it names a "
    "place or environment that the video must show (a bathroom, a beach, a forest, a city street, an "
    "office). A prompt that only describes people, objects, actions or style does not require a scene. "
    "Answer with a single JSON object and nothing else."
)
GATE_INSTRUCTION = (
    'Answer exactly {"requires_scene": true} or {"requires_scene": false}. '
    "true if the prompt names a place or environment the video must show; false otherwise."
)


def gate_scene_requirement(
    client: ChatClient,
    prompt: str,
    *,
    max_tokens: int = 40,
    charge: Callable[[Mapping[str, Any]], int] | None = None,
    item_id: str = "gate",
) -> tuple[bool | None, dict[str, int]]:
    """Cheap text-only gate: does this prompt require a scene at all?"""
    if charge is not None:
        charge({"item_id": item_id, "kind": "scene_gate", "provider": client.provider, "model": client.model})
    user = f"{GATE_INSTRUCTION}\n\nPrompt: {normalize_space(prompt)}"
    usage = {"requests": 1, "prompt_tokens": 0, "completion_tokens": 0}
    try:
        response = client.chat(user=user, system=GATE_SYSTEM, max_tokens=max_tokens, temperature=0.0)
    except LLMError:
        return None, usage
    usage["prompt_tokens"] += response.prompt_tokens
    usage["completion_tokens"] += response.completion_tokens
    parsed, _ = parse_annotation("objects", response.text)
    if isinstance(parsed, Mapping):
        for key in ("requires_scene", "requiresScene", "requires"):
            if key in parsed:
                value = parsed[key]
                if isinstance(value, bool):
                    return value, usage
                if isinstance(value, str) and value.strip().lower() in {"true", "false", "yes", "no"}:
                    return value.strip().lower() in {"true", "yes"}, usage
    return None, usage


def target_preserved(seed_target: Any, rewritten_target: Any, task: str) -> tuple[bool, list[str]]:
    """Every item of the seed target must still be present in the rewritten target."""
    if task == "scene":
        return (str(seed_target) == str(rewritten_target), [] if str(seed_target) == str(rewritten_target) else ["scene_label_changed"])
    if not isinstance(seed_target, Mapping) or not isinstance(rewritten_target, Mapping):
        return False, ["target_shape"]
    missing: list[str] = []
    for key, values in seed_target.items():
        if not isinstance(values, list):
            missing.append(f"unsupported_key:{key}")
            continue
        if key == "relationships":
            rewritten = {json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rewritten_target.get(key, [])}
            for row in values:
                if json.dumps(row, sort_keys=True, ensure_ascii=False) not in rewritten:
                    missing.append(f"relation:{row.get('subject')}-{row.get('relation')}-{row.get('object')}")
        else:
            rewritten = {normalize_phrase(str(x)) for x in rewritten_target.get(key, [])}
            for value in values:
                if normalize_phrase(str(value)) not in rewritten:
                    missing.append(f"{key}:{value}")
    return (not missing), missing

