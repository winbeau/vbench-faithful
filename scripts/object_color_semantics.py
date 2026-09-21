#!/usr/bin/env python3
"""Versioned silver expansion, independent LoRA training and prompt-only inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from vbench_audit_core.inputs import sha256_file
from vbench_audit_models.labels import LabelVocabulary, compilation_request, validate_compilation
from scripts.object_color import save


def uid(value):
    return hashlib.sha256(value.encode()).hexdigest()


def silver_request(seed, vocabulary, categories):
    return (
        "Generate exactly eight labeled English prompt variations as silver training material. "
        "Use one of each requested category. Return JSON with only a records array; each record has "
        "exactly category, prompt, output. Output must obey the supplied minimal schema; no status/confidence. "
        "Keep the source subject for synonym/attributes/plural; do not treat hypernyms as synonyms. "
        "Negated requirements and equally requested different objects abstain with null. "
        "For color: only a color explicitly belonging to the sole target object; scene color does not count. "
        "Crimson/navy/maroon are distinct colors. Do not put instructions in the generated prompts.",
        json.dumps({"dimension": seed["dimension"], "source_prompt": seed["prompt"],
                    "source_gold": seed["gold"], "categories": categories,
                    "object_vocabulary": sorted(vocabulary.objects), "color_vocabulary": sorted(vocabulary.colors),
                    "schema": {"object": "string or null", **({"color": "string or null"} if seed["dimension"] == "color" else {})}}))


def validate_silver(seed, raw, vocabulary, categories):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"records"} or not isinstance(payload["records"], list):
        raise ValueError("expected only records array")
    if Counter(r.get("category") for r in payload["records"]) != Counter(categories):
        raise ValueError("must cover exactly the eight frozen categories")
    records = []
    for r in payload["records"]:
        if set(r) != {"category", "prompt", "output"} or not isinstance(r["prompt"], str) or not r["prompt"].strip():
            raise ValueError("invalid silver record")
        output = validate_compilation(seed["dimension"], r["output"], vocabulary)
        records.append({"sample_id": uid(seed["dimension"] + seed["source_group"] + r["prompt"]),
                        "source_group": seed["source_group"], "split": seed["split"],
                        "dimension": seed["dimension"], "prompt": r["prompt"], "target": output,
                        "category": r["category"], "quality": "silver", "reviewed": False})
    return records


def silver(args):
    compile_root = args.compile_root.resolve()
    sys.path.insert(0, str(compile_root / "src"))
    from vbench_prompts_compile.teacher import DeepSeekClient, BudgetLedger
    config = json.loads((args.config / "object_color_teacher.json").read_text())
    vocabulary_path = args.config / "object_color_vocabulary.json"
    vocabulary = LabelVocabulary.from_file(vocabulary_path)
    seeds = json.loads((args.config / "object_color_seeds.json").read_text())["rows"]
    output = args.output / "semantics"
    output.mkdir(parents=True, exist_ok=True)
    declaration = {"teacher_config": config, "teacher_config_sha256": sha256_file(args.config / "object_color_teacher.json"),
                   "vocabulary_sha256": sha256_file(vocabulary_path),
                   "seed_groups_sha256": sha256_file(args.config / "object_color_seeds.json"),
                   "client_sha256": sha256_file(compile_root / "src/vbench_prompts_compile/teacher.py"),
                   "human_review_slots": config["human_review_slots"], "human_review_completed": 0,
                   "group_split_precedes_expansion": True, "annotation_accuracy": "NOT MEASURED"}
    save(output / "declaration.json", declaration)
    client = DeepSeekClient(token_path=compile_root / "token.txt", base_url=config["base_url"],
                            model=config["api_model"], timeout_seconds=55)
    ledger = BudgetLedger(output, authorized_total=config["max_requests"], max_output_tokens=config["max_tokens"],
                          budget_path=output / "budget.json")
    identity_path = output / "teacher_identity.json"
    if not identity_path.exists():
        ledger.charge({"request_id": "identity-probe", "max_tokens": 32})
        probe = client.chat(system="Return JSON only.", user='Return {"ready":true}.', max_tokens=32, temperature=0)
        if not probe.system_fingerprint or probe.finish_reason != "stop":
            raise ValueError("teacher must expose a version fingerprint and complete response")
        save(identity_path, {"response_model": probe.model, "system_fingerprint": probe.system_fingerprint,
                            "run_date_utc": datetime.now(timezone.utc).isoformat(), "probe": probe.as_metadata(),
                            "version_scope": "API-reported model+fingerprint; not an immutable weights revision"})
    identity = json.loads(identity_path.read_text())
    tasks = []
    for seed in seeds:
        key = seed["dimension"] + "-" + seed["source_group"]
        system, user = silver_request(seed, vocabulary, config["categories"])
        request = {"request_id": key, "system": system, "user": user, "max_tokens": config["max_tokens"]}
        save(output / "requests" / (key + ".json"), request)
        if not (output / "responses" / (key + ".json")).exists():
            ledger.charge({"request_id": key, "request_sha256": uid(json.dumps(request, sort_keys=True)),
                           "max_tokens": config["max_tokens"]})
            tasks.append((key, system, user))

    def call(task):
        key, system, user = task
        try:
            response = client.chat(system=system, user=user, max_tokens=config["max_tokens"], temperature=0)
            item = {"request_id": key, "raw": response.text, "response": response.as_metadata()}
        except Exception as exc:
            item = {"request_id": key, "error": f"{type(exc).__name__}: {exc}"}
        save(output / "responses" / (key + ".json"), item)
        return item

    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for result in pool.map(call, tasks):
            ledger.append_result({k: v for k, v in result.items() if k != "raw"})
            print(json.dumps({"request_id": result["request_id"], "received": "raw" in result}), flush=True)
    rows, rejected = [], []
    for seed in seeds:
        key = seed["dimension"] + "-" + seed["source_group"]
        rows.append({"sample_id": uid(key), "source_group": seed["source_group"], "split": seed["split"],
                     "dimension": seed["dimension"], "prompt": seed["prompt"], "target": seed["gold"],
                     "category": "official_seed", "quality": "gold_seed", "reviewed": False})
        try:
            received = json.loads((output / "responses" / (key + ".json")).read_text())
            meta = received.get("response", {})
            if meta.get("model") != identity["response_model"] or meta.get("system_fingerprint") != identity["system_fingerprint"]:
                raise ValueError("teacher identity drift or failed request; quarantined")
            if meta["finish_reason"] != "stop":
                raise ValueError("truncated teacher output")
            rows.extend(validate_silver(seed, received["raw"], vocabulary, config["categories"]))
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"source_group": seed["source_group"], "dimension": seed["dimension"], "reason": str(exc)})
    prompt_groups = defaultdict(set)
    for r in rows:
        prompt_groups[(r["dimension"], " ".join(r["prompt"].casefold().split()))].add(r["source_group"])
    for r in rows:
        r["cross_source_duplicate"] = len(prompt_groups[(r["dimension"], " ".join(r["prompt"].casefold().split()))]) > 1
        r["training_eligible"] = r["split"] == "train" and not r["cross_source_duplicate"]
    save(output / "records.json", {"rows": rows, "rejected_requests": rejected})
    # Reserve test/development reviews, never promote teacher agreement to review.
    review = sorted([r for r in rows if r["split"] != "train" and r["quality"] == "silver"],
                    key=lambda r: uid("review:" + r["sample_id"]))[:config["human_review_slots"]]
    save(output / "human_review.pending.json", {"rows": [{**r, "reviewer": None, "human_target": None} for r in review]})
    save(output / "freeze.json", {"files": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*.json"))
                                             if p.name != "freeze.json"},
                                 "counts": dict(Counter(r["quality"] for r in rows)),
                                 "requests_charged": ledger.used, "rejected_requests": len(rejected),
                                 "human_review_completed": 0})


def train(args):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    protocol = json.loads((args.config / "object_color_protocol.json").read_text())["training"]
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    source = args.output / "semantics/records.json"
    records = json.loads(source.read_text())["rows"]
    chosen = [r for r in records if r["dimension"] == args.dimension and r["training_eligible"]]
    if not chosen or not any(r["quality"] == "silver" for r in chosen):
        raise ValueError("training needs frozen gold+silver records")
    train_groups = {r["source_group"] for r in chosen}
    if train_groups & {r["source_group"] for r in records if r["split"] != "train"}:
        raise ValueError("source group leakage")
    destination = args.output / "adapters" / args.dimension
    destination.mkdir(parents=True, exist_ok=False)
    set_seed(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    examples = []
    for row in chosen:
        target = validate_compilation(args.dimension, row["target"], vocabulary)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": compilation_request(args.dimension, row["prompt"], vocabulary)}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        completion = json.dumps(target, ensure_ascii=False, separators=(",", ":")) + tokenizer.eos_token
        if len(tokenizer(prompt + completion)["input_ids"]) > protocol["max_length"]:
            raise ValueError("sample exceeds training bound; no silent truncation")
        examples.append({"prompt": prompt, "completion": completion})
    save(destination / "training_manifest.json", {"protocol": protocol, "dimension": args.dimension,
         "data_sha256": sha256_file(source), "vocabulary": vocabulary.provenance,
         "base_path": str(args.base.resolve()),
         "base_file_sha256": {p.name: sha256_file(p) for p in sorted(args.base.iterdir())
                              if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}},
         "source_groups": sorted(train_groups), "n": len(examples), "input_fields": ["prompt"],
         "initial_adapter": "fresh independent random LoRA; no other task weights"})
    model = AutoModelForCausalLM.from_pretrained(str(args.base), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
    model.config.use_cache = False
    config = SFTConfig(output_dir=str(destination), max_steps=protocol["max_steps"],
        per_device_train_batch_size=protocol["batch_size"], gradient_accumulation_steps=protocol["gradient_accumulation"],
        learning_rate=protocol["learning_rate"], max_length=protocol["max_length"],
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True, packing=False, report_to="none", seed=protocol["seed"],
        logging_steps=10, save_steps=100, save_total_limit=2, dataset_num_proc=1)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=config,
        train_dataset=Dataset.from_list(examples), peft_config=LoraConfig(r=protocol["rank"],
        lora_alpha=protocol["alpha"], lora_dropout=protocol["dropout"], target_modules="all-linear", task_type="CAUSAL_LM"))
    result = trainer.train()
    trainer.save_model(str(destination))
    save(destination / "training_result.json", {"metrics": result.metrics,
         "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
         "adapter_sha256": sha256_file(destination / "adapter_model.safetensors")})


def compile_all(args):
    from vbench_audit_models.qwen import QwenPromptRouter
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    records = json.loads((args.output / "semantics/records.json").read_text())["rows"]
    adapters = {d: args.output / "adapters" / d for d in ("object_class", "color")}
    router = QwenPromptRouter.from_local(args.base, adapters)
    extra = json.loads(args.extra_prompts.read_text()) if args.extra_prompts else {}
    for dimension in ("object_class", "color"):
        prompts = sorted({r["prompt"] for r in records if r["dimension"] == dimension and r["split"] != "train"}
                         | set(extra.get(dimension, [])))
        for mode in ("base", "lora"):
            destination = args.output / "compiled" / f"{dimension}-{mode}.json"
            if destination.exists():
                raise FileExistsError(f"compiled output already exists: {destination}")
            outputs = [router.compile(dimension, prompt, vocabulary, mode=mode) for prompt in prompts]
            save(destination, {"dimension": dimension, "mode": mode, "input_fields": ["prompt"],
                 "vocabulary_sha256": vocabulary.provenance["sha256"], "model": router.provenance, "records": outputs})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["silver", "train", "compile"])
    parser.add_argument("--config", type=Path, default=Path("configs/four_dimension"))
    parser.add_argument("--output", type=Path, default=Path("output/object_color_20260920"))
    parser.add_argument("--compile-root", type=Path, default=Path("../vbench-prompts-compile"))
    parser.add_argument("--dimension", choices=["object_class", "color"])
    parser.add_argument("--base", type=Path)
    parser.add_argument("--extra-prompts", type=Path, help="optional dimension -> raw prompt list for frozen video query views")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    {"silver": silver, "train": train, "compile": compile_all}[args.command](args)


if __name__ == "__main__":
    main()
