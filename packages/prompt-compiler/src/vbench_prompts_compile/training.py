"""Training wrappers around the pinned TRL SFTTrainer and PEFT LoRA.

Import policy: this module must stay importable *without* torch so the base
development environment can validate data rendering and configuration. Every
heavy import happens inside a function.

What is deliberately not implemented here: optimizer loops, custom collators or
a vendored trainer. The project reuses ``trl.SFTTrainer`` and
``peft.LoraConfig`` and only fixes the parts the upstream script leaves open
(task instructions, completion-only masking, frozen base, adapter identity).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

from .records import (
    SCENE_LABELS,
    TASKS,
    canonical_json,
    length_bucket,
    read_jsonl,
    validate_record,
    word_count,
)

TASK_INSTRUCTIONS: Mapping[str, str] = {
    "spatial": (
        "Extract every explicit spatial relation between two named entities. "
        "Allowed relation words: left, right, above, below, on, under, near, next to, beside, "
        "behind, in front of, over, inside, outside, between. "
        'Answer with JSON only: {"relationships":[{"subject":"...","relation":"...","object":"..."}]}. '
        'If the prompt states no spatial relation, answer {"relationships":[]}.'
    ),
    "action": (
        "Extract the actions performed by a person or animal as Kinetics-400 action names. "
        'Answer with JSON only: {"actions":["..."]}. If the prompt states no action, answer {"actions":[]}.'
    ),
    "objects": (
        "Extract the concrete physical things explicitly mentioned: people, animals, objects and "
        'visible scene elements. Answer with JSON only: {"entities":["..."]}.'
    ),
    "scene": (
        "Decide whether the caption supports the scene required by the prompt. "
        "Answer with exactly one word: supported, contradicted or insufficient."
    ),
}

SPATIAL_DIRECTIONS_INSTRUCTION = (
    "Extract explicit image-plane directions between two named entities. "
    "The only allowed relation words are left, right, above, below. "
    "Map to the left of to left, to the right of to right, under/beneath/underneath to below, "
    "over/on top of/on the top of to above, and on the bottom of to below when a vertical relation is explicit. "
    "Bare on, beside, near, behind, in front of, inside and between do not specify one of these directions; omit them. "
    'Answer with JSON only: {"relationships":[{"subject":"...","relation":"...","object":"..."}]}. '
    'If no supported direction is stated, answer {"relationships":[]}.'
)


class TrainingConfigError(ValueError):
    """Raised for invalid or unsafe training configuration."""


@dataclass
class TrainingConfig:
    task: str
    model_name_or_path: str
    train_jsonl: str
    output_dir: str
    eval_jsonl: str | None = None
    model_revision: str | None = None
    tokenizer_name_or_path: str | None = None
    max_steps: int = 100
    max_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"
    seed: int = 20260919
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
    packing: bool = False
    completion_only_loss: bool = True
    local_files_only: bool = True
    allow_download: bool = False
    max_samples: int | None = None
    eval_fraction: float = 0.0
    dev_exact_match_every: int = 0
    dev_exact_match_samples: int = 16
    dev_exact_match_max_new_tokens: int = 96
    report_to: str = "none"
    dataset_num_proc: int = 1
    extra: dict[str, Any] = field(default_factory=dict)
    spatial_relation_space: str = "legacy"

    def validate(self) -> None:
        if self.spatial_relation_space not in {"legacy", "four_directions"}:
            raise TrainingConfigError("unknown spatial_relation_space")
        if self.task != "spatial" and self.spatial_relation_space != "legacy":
            raise TrainingConfigError("spatial_relation_space applies only to spatial")
        if self.task not in TASKS:
            raise TrainingConfigError(f"unknown task {self.task!r}")
        if self.max_steps <= 0:
            raise TrainingConfigError("max_steps must be positive")
        if self.max_length < 64:
            raise TrainingConfigError("max_length too small to hold a full sample")
        if not self.completion_only_loss:
            raise TrainingConfigError("completion_only_loss must stay enabled: targets are the only supervision")
        if self.packing:
            raise TrainingConfigError("packing must stay disabled for the first implementation")
        if self.local_files_only and self.allow_download:
            raise TrainingConfigError("local_files_only and allow_download are mutually exclusive")
        if not self.allow_download and not Path(self.model_name_or_path).exists() and not self.local_files_only:
            raise TrainingConfigError("model path does not exist; set allow_download or provide local weights")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainingConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise TrainingConfigError(f"unknown config keys: {sorted(unknown)}")
        config = cls(**dict(payload))
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Path | str) -> "TrainingConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data rendering
# ---------------------------------------------------------------------------

def user_text(record: Mapping[str, Any], *, spatial_relation_space: str | None = None) -> str:
    """Exactly what the student sees: prompt (and caption for scene) only."""
    task = record["task"]
    prompt = record["input"]["prompt"]
    if task == "scene":
        caption = record["input"].get("caption", "")
        return f"{TASK_INSTRUCTIONS['scene']}\n\nPrompt: {prompt}\nCaption: {caption}"
    space = spatial_relation_space or record.get("meta", {}).get("spatial_relation_space", "legacy")
    if task == "spatial" and space == "four_directions":
        return f"{SPATIAL_DIRECTIONS_INSTRUCTION}\n\nPrompt: {prompt}"
    return f"{TASK_INSTRUCTIONS[task]}\n\nPrompt: {prompt}"


def completion_text(record: Mapping[str, Any]) -> str:
    """The only supervised span: canonical JSON, or a bare scene label."""
    if record["task"] == "scene":
        return str(record["target"])
    return canonical_json(record["target"])


def load_records(
    path: Path | str,
    task: str,
    *,
    limit: int | None = None,
    action_resolver: Callable[[str], str | None] | None = None,
) -> list[dict[str, Any]]:
    """Load and re-validate JSONL records for one task."""
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        if record.get("task") != task:
            rejected.append({"sample_id": record.get("sample_id"), "reason": "task_mismatch"})
            continue
        validation = validate_record(record, action_resolver=action_resolver)
        if not validation.ok:
            rejected.append({"sample_id": record.get("sample_id"), "reason": ";".join(validation.errors)})
            continue
        rows.append(record)
        if limit is not None and len(rows) >= limit:
            break
    if rejected:
        raise TrainingConfigError(f"{len(rejected)} records failed validation; first: {rejected[0]}")
    return rows


def render_examples(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Instruction+prompt text (what the model sees) plus the raw prompt length.

    ``prompt_words`` keeps S4 length buckets tied to the *user prompt*, not to
    the task instruction that is prepended to it.
    """
    return [
        {
            "prompt": user_text(record),
            "completion": completion_text(record),
            "prompt_words": word_count(record["input"]["prompt"]),
        }
        for record in records
    ]


def group_split(records: Sequence[Mapping[str, Any]], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by ``group_id`` so paraphrases of one source never cross the split."""
    if eval_fraction <= 0:
        return list(records), []
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record["group_id"]), []).append(dict(record))
    ordered = sorted(groups)
    import random

    random.Random(seed).shuffle(ordered)
    eval_groups = max(1, int(len(ordered) * eval_fraction))
    eval_ids = set(ordered[:eval_groups])
    train = [row for gid in ordered if gid not in eval_ids for row in groups[gid]]
    evaluation = [row for gid in ordered if gid in eval_ids for row in groups[gid]]
    return train, evaluation


def length_report(tokenizer: Any, examples: Sequence[Mapping[str, str]], *, max_length: int) -> dict[str, Any]:
    """Serialize prompt+completion and report the real full-sequence budget."""
    lengths: list[int] = []
    over = 0
    buckets: dict[str, int] = {}
    for example in examples:
        text = example["prompt"] + example["completion"]
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        length = len(encoded) + 1  # EOS
        lengths.append(length)
        words = int(example.get("prompt_words", word_count(example["prompt"])))
        buckets[length_bucket(words)] = buckets.get(length_bucket(words), 0) + 1
        if length > max_length:
            over += 1
    if not lengths:
        return {"samples": 0}
    lengths.sort()

    def quantile(q: float) -> int:
        index = min(len(lengths) - 1, int(q * (len(lengths) - 1)))
        return lengths[index]

    return {
        "samples": len(lengths),
        "max_length": max_length,
        "over_budget": over,
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "max": lengths[-1],
        "word_buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Model / trainer construction (torch imported lazily)
# ---------------------------------------------------------------------------

def load_tokenizer(config: TrainingConfig):
    from transformers import AutoTokenizer

    source = config.tokenizer_name_or_path or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=config.model_revision,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_chat_template(tokenizer: Any, user: str, completion: str | None) -> tuple[str, str]:
    """Return ``(prompt_text, full_text)`` using one fixed no-thinking template."""
    messages = [{"role": "user", "content": user}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt_text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    if completion is None:
        return prompt_text, prompt_text
    return prompt_text, prompt_text + completion


def load_model(config: TrainingConfig):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def build_peft_config(config: TrainingConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(config: TrainingConfig, *, has_eval: bool):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        max_length=config.max_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        packing=config.packing,
        completion_only_loss=config.completion_only_loss,
        seed=config.seed,
        report_to=config.report_to,
        dataset_num_proc=config.dataset_num_proc,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.save_steps if has_eval else None,
        do_eval=has_eval,
        save_only_model=False,
        data_seed=config.seed,
        model_init_kwargs=None,
    )


def parameter_inventory(model: Any) -> dict[str, Any]:
    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    frozen = sum(param.numel() for _, param in model.named_parameters() if not param.requires_grad)
    return {
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(size for _, size in trainable),
        "frozen_parameters": frozen,
        "trainable_names_head": [name for name, _ in trainable[:8]],
    }


def _normalize_param_name(name: str) -> str:
    """Strip PEFT wrapper prefixes so base fingerprints are comparable."""
    prefixes = ("base_model.model.", "base_model.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def parameter_fingerprint(model: Any, *, adapter_only: bool = False) -> str:
    """Cheap deterministic digest of selected parameters.

    Samples the first and last element of every tensor instead of reading all
    weights: enough to detect an update, cheap on an 8B model. ``adapter_only``
    selects LoRA tensors; otherwise they are excluded (frozen-base check).
    """
    import hashlib

    import torch

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            if ("lora_" in name) != adapter_only:
                continue
            digest.update(_normalize_param_name(name).encode())
            digest.update(str(tuple(param.shape)).encode())
            digest.update(str(param.dtype).encode())
            flat = param.detach().reshape(-1).to(torch.float32)
            if flat.numel():
                digest.update(flat[0].cpu().numpy().tobytes())
                digest.update(flat[-1].cpu().numpy().tobytes())
    return digest.hexdigest()


def base_parameter_fingerprint(model: Any) -> str:
    """Digest of the frozen backbone (adapter tensors excluded)."""
    return parameter_fingerprint(model, adapter_only=False)


def adapter_parameter_fingerprint(model: Any) -> str:
    """Digest of the trainable LoRA tensors (must change during training)."""
    return parameter_fingerprint(model, adapter_only=True)


def masked_label_report(trainer: Any, tokenizer: Any, *, probe: str, expected_target: str) -> dict[str, Any]:
    """Inspect a real collated batch instead of trusting ``completion_only_loss``.

    ``probe`` is a distinctive substring of the *prompt* (must not be supervised);
    ``expected_target`` is the completion text (must be supervised).
    """
    dataset = trainer.train_dataset
    if dataset is None or len(dataset) == 0:
        raise TrainingConfigError("empty training dataset")
    row = dict(dataset[0])
    if "labels" in row and "input_ids" in row:
        input_ids = list(row["input_ids"])
        labels = list(row["labels"])
    else:
        batch = trainer.data_collator([row])
        input_ids = list(batch["input_ids"][0])
        labels = list(batch["labels"][0])
    labels = [(-100 if value is None else value) for value in labels]
    supervised_ids = [token for token, label in zip(input_ids, labels) if label != -100]
    decoded = tokenizer.decode(supervised_ids)
    prompt_supervised = sum(1 for label in labels[: max(1, len(labels) - len(supervised_ids))] if label != -100)

    def squash(text: str) -> str:
        return " ".join(text.split())

    return {
        "full_tokens": len(input_ids),
        "supervised_tokens": len(supervised_ids),
        "prompt_tokens": len(input_ids) - len(supervised_ids),
        "prompt_fully_masked": prompt_supervised == 0,
        "all_masked": len(supervised_ids) == 0,
        "target_present_in_supervision": squash(expected_target) in squash(decoded),
        "prompt_probe_leaked": squash(probe) in squash(decoded),
        "supervised_preview": squash(decoded)[:120],
    }


class DevExactMatchCallback:
    """Generation-based dev accuracy logged while training runs.

    TRL already logs token-level accuracy; this adds the metric the project
    actually cares about (exact canonical target match) on a fixed, group-
    disjoint dev slice, written to ``dev_metrics.jsonl`` in the run directory.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        output_dir: Path | str,
        max_new_tokens: int,
        every: int,
    ) -> None:
        from transformers import TrainerCallback

        self.records = list(records)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.every = int(every)
        self.path = self.output_dir / "dev_metrics.jsonl"

        outer = self

        class _Callback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):  # noqa: D102 - transformers signature
                every = outer.every
                if every <= 0 or state.global_step == 0 or state.global_step % every != 0:
                    return control
                model = kwargs.get("model")
                metrics = outer.evaluate(model, state.global_step)
                outer.append(metrics)
                return control

        self.callback = _Callback()

    def evaluate(self, model: Any, step: int) -> dict[str, Any]:
        import torch

        if model is None:
            return {"step": step, "error": "no_model"}
        was_training = model.training
        model.eval()
        exact = 0
        parseable = 0
        per_task: dict[str, list[int]] = {}
        with torch.no_grad():
            for record in self.records:
                prompt_text, _ = apply_chat_template(self.tokenizer, user_text(record), None)
                inputs = self.tokenizer(prompt_text, return_tensors="pt")
                inputs.pop("token_type_ids", None)
                inputs = inputs.to(model.device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                raw = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                target = record["target"] if record["task"] == "scene" else json.loads(canonical_json(record["target"]))
                parsed = _loose_parse(record["task"], raw)
                ok = parsed is not None and canonical_json(parsed) == canonical_json(target)
                parseable += int(parsed is not None)
                exact += int(ok)
                per_task.setdefault(record["task"], []).append(int(ok))
        if was_training:
            model.train()
        total = len(self.records)
        return {
            "step": step,
            "dev_samples": total,
            "dev_exact_match": exact / total if total else 0.0,
            "dev_parseable_rate": parseable / total if total else 0.0,
        }

    def append(self, metrics: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(metrics), ensure_ascii=False, sort_keys=True) + "\n")


def _loose_parse(task: str, raw: str) -> Any | None:
    """Parse a generation for the dev metric without importing inference.py."""
    text = raw.strip()
    if task == "scene":
        first = text.split()[0].strip(".,:;!\"'").lower() if text.split() else ""
        return first if first in SCENE_LABELS else None
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def train(config: TrainingConfig) -> dict[str, Any]:
    """Run one SFT job and return a machine-readable summary."""
    import torch
    from datasets import Dataset
    from trl import SFTTrainer

    from .sources import load_k400

    action_resolver = load_k400().resolve if config.task == "action" else None
    records = load_records(config.train_jsonl, config.task, limit=config.max_samples, action_resolver=action_resolver)
    if not records:
        raise TrainingConfigError(f"no usable records in {config.train_jsonl}")
    if config.eval_jsonl:
        eval_path = Path(config.eval_jsonl)
        eval_records = load_records(eval_path, config.task, limit=config.max_samples, action_resolver=action_resolver)
        if set(row["group_id"] for row in eval_records) & set(row["group_id"] for row in records):
            raise TrainingConfigError("eval_jsonl shares group_id families with the training data")
        train_records = records
    else:
        train_records, eval_records = group_split(records, config.eval_fraction, config.seed)
    if config.task == "spatial":
        for record in train_records + eval_records:
            record["meta"] = {**record.get("meta", {}), "spatial_relation_space": config.spatial_relation_space}
            if config.spatial_relation_space == "four_directions":
                from .records import DIRECTION_RELATIONS
                if any(r["relation"] not in DIRECTION_RELATIONS for r in record["target"]["relationships"]):
                    raise TrainingConfigError(f"out-of-space spatial target: {record['sample_id']}")
    tokenizer = load_tokenizer(config)
    processed: dict[str, list[dict[str, str]]] = {}
    for split_name, split_records in (("train", train_records), ("eval", eval_records)):
        rows = []
        for record in split_records:
            prompt_text, full_text = apply_chat_template(tokenizer, user_text(record), completion_text(record))
            rows.append(
                {
                    "prompt": prompt_text,
                    "completion": completion_text(record),
                    "text": full_text,
                    "prompt_words": word_count(record["input"]["prompt"]),
                }
            )
        processed[split_name] = rows
    report = length_report(
        tokenizer,
        [{"prompt": r["prompt"], "completion": r["completion"], "prompt_words": r["prompt_words"]} for r in processed["train"]],
        max_length=config.max_length,
    )
    if report.get("over_budget"):
        raise TrainingConfigError(f"{report['over_budget']} training samples exceed max_length={config.max_length}; refusing to truncate")

    model = load_model(config)
    callbacks = []
    dev_callback = None
    if config.dev_exact_match_every > 0 and eval_records:
        sample = eval_records[: config.dev_exact_match_samples]
        dev_callback = DevExactMatchCallback(
            sample,
            tokenizer,
            config.output_dir,
            config.dev_exact_match_max_new_tokens,
            config.dev_exact_match_every,
        )
        callbacks.append(dev_callback.callback)
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(config, has_eval=bool(eval_records)),
        train_dataset=Dataset.from_list(processed["train"]),
        eval_dataset=Dataset.from_list(processed["eval"]) if eval_records else None,
        peft_config=build_peft_config(config),
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )
    # Fingerprint the base *through the trainer*: identical naming and object on
    # both sides of training, so the frozen check cannot be fooled by wrappers.
    base_model = trainer.model.get_base_model() if hasattr(trainer.model, "get_base_model") else trainer.model
    before = base_parameter_fingerprint(base_model)
    before_adapter = adapter_parameter_fingerprint(trainer.model)
    inventory = parameter_inventory(trainer.model)
    if inventory["trainable_parameters"] == 0:
        raise TrainingConfigError("no trainable parameters: LoRA injection failed")
    first = processed["train"][0]
    probe = "Extract" if config.task != "scene" else "Caption:"
    mask_report = masked_label_report(trainer, tokenizer, probe=probe, expected_target=first["completion"])
    if mask_report["all_masked"] or not mask_report["prompt_fully_masked"]:
        raise TrainingConfigError(f"completion-only masking is wrong: {mask_report}")
    if not mask_report["target_present_in_supervision"] or mask_report["prompt_probe_leaked"]:
        raise TrainingConfigError(f"supervised span is not the target: {mask_report}")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "training_config.json").write_text(
        json.dumps(config.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    after = base_parameter_fingerprint(base_model)
    after_adapter = adapter_parameter_fingerprint(trainer.model)
    if dev_callback is not None:
        dev_callback.append(dev_callback.evaluate(trainer.model, int(trainer.state.global_step)))
    summary = {
        "task": config.task,
        "output_dir": config.output_dir,
        "train_samples": len(processed["train"]),
        "eval_samples": len(processed["eval"]),
        "train_records_groups": len({r["group_id"] for r in train_records}),
        "eval_records_groups": len({r["group_id"] for r in eval_records}),
        "max_steps": config.max_steps,
        "max_length": config.max_length,
        "length_report": report,
        "parameter_inventory": inventory,
        "mask_report": mask_report,
        "base_frozen": before == after,
        "adapter_changed": before_adapter != after_adapter,
        "train_metrics": {k: v for k, v in (result.metrics or {}).items()},
        "config": config.as_dict(),
    }
    (Path(config.output_dir) / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if not summary["base_frozen"]:
        raise TrainingConfigError("base parameters changed during LoRA training")
    if not summary["adapter_changed"]:
        raise TrainingConfigError("adapter parameters did not change: no gradient reached the LoRA weights")
    return summary


def write_run_manifest(run_dir: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(run_dir) / "run_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def copy_config(config_path: Path | str, run_dir: Path | str) -> None:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, target / "training_config.json")
