"""Inference: one frozen backbone with three named LoRAs, plus the Scene model.

Hard routing only: the caller states the task; there is no automatic dimension
selection and no fallback to another adapter. Adapter switching mutates model
state, so the first implementation serialises requests instead of pretending to
be thread safe.

torch/transformers/peft are imported lazily so the module stays importable in
the base development environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .records import SCENE_LABELS, TASKS, canonical_json
from .teacher import parse_teacher_text

PARSE_TASKS = ("spatial", "action", "objects")


class RoutingError(RuntimeError):
    """Raised for unknown tasks, missing adapters or mismatched backbones."""


@dataclass(frozen=True)
class AdapterSpec:
    task: str
    path: str


@dataclass
class Prediction:
    task: str
    raw: str
    value: Any | None
    errors: tuple[str, ...]
    adapter: str | None = None
    postprocess: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        result = {"task": self.task, "raw": self.raw, "value": self.value, "errors": list(self.errors), "adapter": self.adapter}
        if self.postprocess is not None:
            result['postprocess'] = self.postprocess
        return result


def canonicalize_actions(value: Any, resolver) -> tuple[Any, int]:
    """Map generated action phrases onto the frozen Kinetics-400 vocabulary.

    Deterministic vocabulary lookup, not a semantic matcher: a phrase that resolves to
    a class name is replaced by that class, the ``other`` sentinel is kept, and anything
    unmappable becomes ``other`` (decision B2) instead of reaching the scorer as an
    invented class. Returns ``(value, changed_count)``.
    """
    from .records import ACTION_OTHER, normalize_phrase

    if not isinstance(value, Mapping) or not isinstance(value.get("actions"), list):
        return value, 0
    mapped: list[str] = []
    changed = 0
    for raw in value["actions"]:
        name = str(raw)
        if normalize_phrase(name) == ACTION_OTHER:
            mapped.append(ACTION_OTHER)
            continue
        canonical = resolver(name) if resolver is not None else None
        target = canonical if canonical else ACTION_OTHER
        if normalize_phrase(target) != normalize_phrase(name):
            changed += 1
        mapped.append(target)
    deduped: list[str] = []
    for name in mapped:
        if normalize_phrase(name) not in {normalize_phrase(existing) for existing in deduped}:
            deduped.append(name)
    return {"actions": deduped}, changed


def canonicalize_entities(value: Any) -> tuple[Any, int]:
    """Normalise generated entity names onto the project's canonical form.

    Deterministic string normalisation (determiner stripping, singular head), the same
    function used to build the training targets, so a correct answer phrased as
    "a cat" is not scored as a miss against "cat".
    """
    from .records import canonical_entity_name, normalize_phrase

    if not isinstance(value, Mapping) or not isinstance(value.get("entities"), list):
        return value, 0
    mapped: list[str] = []
    changed = 0
    for raw in value["entities"]:
        canonical = canonical_entity_name(str(raw))
        if normalize_phrase(canonical) != normalize_phrase(str(raw)):
            changed += 1
        if canonical and normalize_phrase(canonical) not in {normalize_phrase(existing) for existing in mapped}:
            mapped.append(canonical)
    return {"entities": mapped}, changed


def parse_output(task: str, text: str) -> tuple[Any | None, tuple[str, ...]]:
    """Strictly parse a generated target; never 'repair' a malformed answer."""
    value, errors = parse_teacher_text(task, text)
    if value is None:
        return None, tuple(errors)
    if task == "scene":
        label = str(value).strip().lower()
        if label not in SCENE_LABELS:
            return None, ("scene_label",)
        return label, ()
    if not isinstance(value, Mapping):
        return None, ("schema_types",)
    expected = {"spatial": {"relationships"}, "action": {"actions"}, "objects": {"entities"}}[task]
    if set(value) != expected:
        return None, ("schema_keys",)
    return dict(value), ()


def adapter_relation_space(path: str) -> str:
    directory = Path(path)
    config = directory / "training_config.json"
    if not config.exists() and directory.name.startswith("checkpoint-"):
        config = directory.parent / "training_config.json"
    if not config.exists():
        return "legacy"
    space = json.loads(config.read_text()).get("spatial_relation_space", "legacy")
    if space not in {"legacy", "four_directions"}:
        raise RoutingError(f"unknown adapter relation space: {space}")
    return space


def validate_direction_output(value):
    from .records import DIRECTION_RELATIONS
    if not isinstance(value, Mapping) or not isinstance(value.get("relationships"), list):
        return None, ("spatial_schema",)
    for relation in value["relationships"]:
        if not isinstance(relation, Mapping) or relation.get("relation") not in DIRECTION_RELATIONS:
            return None, ("spatial_relation_out_of_space",)
        if set(relation) != {"subject", "relation", "object"} or not all(isinstance(v, str) and v.strip() for v in relation.values()):
            return None, ("spatial_schema",)
    return value, ()


class AdapterRouter:
    """Base model + spatial/action/objects adapters with explicit hard routing."""

    def __init__(
        self,
        *,
        base_model_path: str,
        adapters: Mapping[str, str],
        base_revision: str | None = None,
        local_files_only: bool = True,
        dtype: str = "bfloat16",
        device_map: str | None = None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not adapters:
            raise RoutingError("no adapters configured")
        unknown = set(adapters) - set(PARSE_TASKS)
        if unknown:
            raise RoutingError(f"unknown adapter tasks: {sorted(unknown)}")
        self.base_model_path = base_model_path
        self.base_revision = base_revision
        self.adapters = dict(adapters)
        self.spatial_relation_space = adapter_relation_space(adapters["spatial"]) if "spatial" in adapters else "legacy"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, revision=base_revision, local_files_only=local_files_only, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            revision=base_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        first_task = sorted(adapters)[0]
        model = PeftModel.from_pretrained(base, adapters[first_task], adapter_name=first_task, is_trainable=False)
        for task in sorted(adapters):
            if task == first_task:
                continue
            model.load_adapter(adapters[task], adapter_name=task, is_trainable=False)
        model.eval()
        if device_map is None:
            model.to(self.device)
        self.model = model
        self.active_task: str | None = None

    def adapter_names(self) -> list[str]:
        return sorted(self.model.peft_config)

    def user_text(self, record: Mapping[str, Any]) -> str:
        from .training import user_text
        return user_text(record, spatial_relation_space=self.spatial_relation_space)

    def route(self, task: str) -> str:
        if task not in PARSE_TASKS:
            raise RoutingError(f"task {task!r} is not a parse task; use the Scene model instead")
        if task not in self.adapters:
            raise RoutingError(f"no adapter loaded for task {task!r}; loaded: {self.adapter_names()}")
        self.model.set_adapter(task)
        self.active_task = task
        return task

    def build_prompt(self, user: str) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], **kwargs)

    def generate(self, user: str, *, max_new_tokens: int = 256) -> str:
        import torch

        text = self.build_prompt(user)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs.pop("token_type_ids", None)  # Qwen-style models do not accept it
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict(
        self,
        task: str,
        user: str,
        *,
        max_new_tokens: int = 256,
        canonicalize_action_output: bool = False,
        canonicalize_entity_output: bool = False,
        action_interface: str = 'repair-v2.1',
        action_prompt: str | None = None,
    ) -> Prediction:
        if action_interface not in {'repair-v2.1', 'repair-v2', 'legacy-model'}:
            raise RoutingError('unknown Action interface: ' + str(action_interface))
        adapter = self.route(task)
        raw = self.generate(user, max_new_tokens=max_new_tokens)
        value, errors = parse_output(task, raw)
        postprocess = None
        if task == "spatial" and self.spatial_relation_space == "four_directions" and value is not None:
            value, direction_errors = validate_direction_output(value)
            errors = tuple(errors) + direction_errors
        if task == 'action' and action_interface in {'repair-v2.1', 'repair-v2'}:
            from .action_repair import compile_action
            from .sources import load_k400

            # Public callers pass the original prompt explicitly. The fallback
            # handles the exact training.user_text template used by old callers.
            _, marker, extracted = user.partition('\n\nPrompt: ')
            prompt = action_prompt if action_prompt is not None else (extracted if marker else user)
            if not hasattr(self, '_action_vocabulary'):
                self._action_vocabulary = load_k400()
            postprocess = compile_action(prompt, value, self._action_vocabulary,
                                         scope_guard=action_interface == 'repair-v2.1')
            value = postprocess['target']
            if postprocess['changed']:
                errors = tuple(errors) + ('action_repaired:' + postprocess['method'],)
        elif canonicalize_action_output and task == "action" and value is not None:
            from .sources import load_k400

            value, changed = canonicalize_actions(value, load_k400().resolve)
            if changed:
                errors = tuple(errors) + (f"action_canonicalized:{changed}",)
        if canonicalize_entity_output and task == "objects" and value is not None:
            value, changed = canonicalize_entities(value)
            if changed:
                errors = tuple(errors) + (f"entity_canonicalized:{changed}",)
        return Prediction(task=task, raw=raw, value=value, errors=errors, adapter=adapter, postprocess=postprocess)

    def predict_base(self, task: str, user: str, *, max_new_tokens: int = 256) -> Prediction:
        """Same prompt with every adapter disabled (un-finetuned control)."""
        with self.model.disable_adapter():
            raw = self.generate(user, max_new_tokens=max_new_tokens)
        value, errors = parse_output(task, raw)
        if task == "spatial" and self.spatial_relation_space == "four_directions" and value is not None:
            value, direction_errors = validate_direction_output(value)
            errors = tuple(errors) + direction_errors
        return Prediction(task=task, raw=raw, value=value, errors=errors, adapter=None)


class SceneVerifier:
    """Independent single-label model: prompt + caption -> supported/contradicted/insufficient."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        adapter_path: str | None = None,
        base_model_path: str | None = None,
        revision: str | None = None,
        local_files_only: bool = True,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_path is None and not (base_model_path and adapter_path):
            raise RoutingError("SceneVerifier needs either model_path or base_model_path+adapter_path")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer_source = base_model_path if adapter_path else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision, local_files_only=local_files_only, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            tokenizer_source, revision=revision, local_files_only=local_files_only, trust_remote_code=False, torch_dtype=torch_dtype
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        self.model = model.to(self.device)
        self.model.eval()
        self.adapter_path = adapter_path

    def predict_batch(self, pairs: Sequence[tuple[str, str]], *, max_new_tokens: int = 8) -> list[Prediction]:
        """Greedy left-padded decoding with the same label/newline contract.

        Batch size is a resource setting, never selected using label accuracy.
        All captions remain independent inputs, including repeated video frames.
        """
        import torch
        from .training import user_text

        if not pairs:
            return []
        texts = []
        for prompt, caption in pairs:
            user = user_text({'task': 'scene', 'input': {'prompt': prompt, 'caption': caption}})
            kwargs = {'tokenize': False, 'add_generation_prompt': True}
            try:
                text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': user}], enable_thinking=False, **kwargs)
            except TypeError:
                text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': user}], **kwargs)
            texts.append(text)
        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        inputs.pop('token_type_ids', None)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, temperature=None, top_p=None,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=['\n'], tokenizer=self.tokenizer)
        raws = self.tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        result = []
        for raw in raws:
            raw = raw.strip()
            value, errors = parse_output('scene', raw)
            result.append(Prediction(task='scene', raw=raw, value=value, errors=errors))
        return result

    def predict(self, prompt: str, caption: str, *, max_new_tokens: int = 8, stop_at_newline: bool = False) -> Prediction:
        """Single-label prediction.

        ``stop_at_newline`` enforces the "one label, nothing else" contract at
        decode time (the model is still free to emit a wrong label). It is off by
        default so the strict metric keeps reporting chatty outputs as failures;
        the smoke report gives both variants.
        """
        import torch

        from .training import user_text

        user = user_text({"task": "scene", "input": {"prompt": prompt, "caption": caption}})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], enable_thinking=False, **kwargs)
        except TypeError:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], **kwargs)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)
        generation: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if stop_at_newline:
            generation["stop_strings"] = ["\n"]
            generation["tokenizer"] = self.tokenizer
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation)
        raw = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        value, errors = parse_output("scene", raw)
        return Prediction(task="scene", raw=raw, value=value, errors=errors, adapter=None)
