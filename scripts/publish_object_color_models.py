#!/usr/bin/env python3
"""Publish the user-selected paper-final step 300, without claiming dev selection."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from publish_best_models import BASE_SHA, sha256


def run(args):
    selections = []
    for dimension in ["object_class", "color"]:
        source = args.source / dimension
        weight = source / "checkpoint-300/adapter_model.safetensors"
        training = json.loads((source / "training_result.json").read_text())
        digest = sha256(weight)
        if digest != training["adapter_sha256"] or sha256(source / "adapter_model.safetensors") != digest:
            raise ValueError("Step 300 differs from the documented final model")
        trainer = json.loads((source / "checkpoint-300/trainer_state.json").read_text())
        destination = args.stage / dimension
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / weight.name
        if not target.exists():
            os.link(weight, target)
        if sha256(target) != digest:
            raise ValueError("Existing staging model differs from selected step 300")
        for name in ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "merges.txt",
                     "vocab.json", "tokenizer.json", "chat_template.jinja", "training_manifest.json", "training_result.json"]:
            if (source / name).is_file():
                shutil.copy2(source / name, destination / name)
        raw_config = source / "checkpoint-300/adapter_config.json"
        shutil.copy2(raw_config, destination / "source_adapter_config.json")
        config = json.loads(raw_config.read_text())
        config.update(base_model_name_or_path="Qwen/Qwen3-8B", revision=BASE_SHA)
        (destination / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        selection = {"dimension": dimension, "version": "object-color-20260920", "selected_step": 300,
            "source_checkpoint": str(weight), "selection": "user_selected_paper_final_fixed_step",
            "development_selected": False, "development_evaluation_records": sum(
                any(k.startswith("eval_") for k in r) for r in trainer.get("log_history", [])),
            "test_used_for_selection": False, "sha256": digest, "bytes": weight.stat().st_size,
            "base_model": "Qwen/Qwen3-8B", "base_revision": BASE_SHA,
            "note": "The user selected the paper's fixed-step final model. No development best-checkpoint comparison was performed."}
        (destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        (destination / "README.md").write_text(f"# {dimension}: paper-final step 300\n\n"
            "Published at the user's explicit request. This is the paper's fixed 300-step final adapter, "
            "**not selected by development evaluation**. Retained steps 200 and 300 have no logged dev "
            "comparison. No test score was used to choose a checkpoint.\n\n"
            "The original training record, exact adapter hash, tokenizer and portable PEFT configuration "
            "are included. The base is Qwen/Qwen3-8B at the revision in selection.json. The complete "
            "metric also needs the frozen object/color vocabulary, prompt schema and video backend. "
            "Prompt-only inference source is included under code/object_color/.\n")
        subprocess.run([args.hf, "upload", args.repo_id, str(destination), dimension, "--repo-type", "model",
            "--commit-message", f"model({dimension}): publish paper-final step 300 without dev selection"],
            env=dict(os.environ, HF_ENDPOINT=args.endpoint, HF_HUB_DISABLE_PROGRESS_BARS="1"), check=True)
        selections.append(selection)
        print(json.dumps({"event": "paper_final_model_published", **selection}), flush=True)
    (args.stage / "object-color-models.json").write_text(json.dumps(selections, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--repo-id", default="xju-arlab/vbench-model")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--hf", default="hf")
    run(parser.parse_args())
