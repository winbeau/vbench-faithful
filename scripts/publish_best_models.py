#!/usr/bin/env python3
"""Publish one best available, development-selected LoRA per semantic dimension.

Select among retained checkpoints by logged dev exact match, breaking ties by
the latest step. Record pruned superior checkpoints explicitly. Never use test
scores for selection, and never publish optimizer states or the frozen base.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

SPECS = [("spatial_relationship", "v8-8b", "spatial"),
         ("scene", "v8-8b", "scene"),
         ("human_action", "v9-8b", "action"),
         ("multiple_objects", "v6-8b", "objects")]
BASE_SHA = "b968826d9c46dd6066d109eabc6255188de91218"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(args):
    catalog = []
    for dimension, version, task in SPECS:
        source = args.source / "runs/formal" / version / task
        log = [json.loads(l) for l in (source / "dev_metrics.jsonl").read_text().splitlines() if l.strip()]
        scores = {r["step"]: r for r in log}
        config = json.loads((source / "training_config.json").read_text())
        candidates = []
        for step, row in scores.items():
            p = source / f"checkpoint-{step}" / "adapter_model.safetensors"
            if p.is_file():
                candidates.append((float(row["dev_exact_match"]), step, p))
        final = source / "adapter_model.safetensors"
        if final.is_file() and config["max_steps"] in scores:
            row = scores[config["max_steps"]]
            candidates.append((float(row["dev_exact_match"]), row["step"], final))
        if not candidates:
            raise ValueError("No evaluated retained checkpoint: " + str(source))
        score, step, weight = max(candidates, key=lambda c: (c[0], c[1], c[2] == final))
        best_logged = max(float(r["dev_exact_match"]) for r in log)
        destination = args.stage / dimension
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                         "chat_template.jinja", "merges.txt", "vocab.json", "tokenizer.json",
                         "run_manifest.json", "run_summary.json", "training_config.json", "dev_metrics.jsonl"]:
            if (source / filename).is_file():
                shutil.copy2(source / filename, destination / filename)
        target = destination / "adapter_model.safetensors"
        if not target.exists():
            os.link(weight, target)
        if sha256(target) != sha256(weight):
            raise ValueError("Existing staging weight differs from selected checkpoint")
        raw_config = weight.parent / "adapter_config.json"
        shutil.copy2(raw_config, destination / "source_adapter_config.json")
        adapter = json.loads(raw_config.read_text())
        adapter["base_model_name_or_path"] = "Qwen/Qwen3-8B"
        adapter["revision"] = BASE_SHA
        (destination / "adapter_config.json").write_text(json.dumps(adapter, indent=2) + "\n")
        with target.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if not tensors or not all("lora_" in k for k in tensors):
            raise ValueError("Selected artifact is not a LoRA tensor archive")
        selection = {"dimension": dimension, "version": version, "task": task,
                     "source_checkpoint": str(weight), "selection": "best_retained_dev_exact_match_then_latest_step",
                     "selected_step": step, "dev_exact_match": score,
                     "dev_samples": scores[step]["dev_samples"], "best_logged_dev_exact_match": best_logged,
                     "higher_scoring_checkpoint_pruned": best_logged > score,
                     "test_used_for_selection": False, "sha256": sha256(target),
                     "bytes": target.stat().st_size, "tensor_count": len(tensors),
                     "base_model": "Qwen/Qwen3-8B", "base_revision": BASE_SHA,
                     "source_code_sha": subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()}
        (destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        note = (" A higher-scoring development checkpoint was pruned by the original training run; "
                "this is the best retained checkpoint, not a claim to recover the missing one." if best_logged > score else "")
        (destination / "README.md").write_text(f"# {dimension}\n\n"
            f"{version}, step {step}; best retained checkpoint by the recorded 24-example development "
            f"exact-match probe ({score:.6f}). Ties use the latest step.{note}\n\n"
            "Weights are byte-identical to the selected training artifact. The portable adapter "
            "configuration points to Qwen/Qwen3-8B at the revision in selection.json; the original "
            "adapter configuration is preserved separately. The frozen base model is not duplicated.\n\n"
            "These adapters require the task schemas and deterministic postprocessing from "
            "vbench-prompts-compile. Raw LoRA output alone does not reproduce the complete repaired "
            "metric. Development scores use a small probe and are not an independent test claim.\n")
        catalog.append(selection)
        print(json.dumps({"event": "selected_model", **selection}), flush=True)
        env = dict(os.environ, HF_ENDPOINT=args.endpoint, HF_HUB_DISABLE_PROGRESS_BARS="1")
        subprocess.run([args.hf, "upload", args.repo_id, str(destination), dimension, "--repo-type", "model",
                        "--commit-message", f"model({dimension}): publish best retained {version} step {step}"], env=env, check=True)
    (args.stage / "semantic-models.json").write_text(json.dumps(catalog, indent=2) + "\n")
    subprocess.run([args.hf, "upload", args.repo_id, str(args.stage / "semantic-models.json"), "semantic-models.json",
                    "--repo-type", "model", "--commit-message", "docs: record semantic model selections and hashes"],
                   env=dict(os.environ, HF_ENDPOINT=args.endpoint), check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--repo-id", default="xju-arlab/vbench-model")
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--hf", default="hf")
    run(p.parse_args())
