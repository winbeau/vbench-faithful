#!/usr/bin/env python3
"""Restore pinned nine-dimension assets, optionally including all selected models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_copy(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f"Downloaded content differs: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f"Existing file differs; choose a fresh output directory: {target}")
    else:
        shutil.copy2(source, target)


def extract_verified(archive, destination, entries):
    expected = {entry["path"]: entry for entry in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate expected archive member")
    seen = set()
    destination = destination.resolve()
    with tarfile.open(archive) as tf:
        for member in tf:
            relative = PurePosixPath(member.name)
            target = (destination / member.name).resolve()
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or destination not in target.parents or member.name not in expected or member.name in seen):
                raise ValueError(f"Unsafe or unexpected archive member: {member.name}")
            raw = tf.extractfile(member).read()
            identity = expected[member.name]
            if member.size != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError(f"Archive member differs: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or digest(target) != identity["sha256"]:
                    raise ValueError(f"Existing archive member differs: {target}")
            else:
                target.write_bytes(raw)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive member coverage differs")


def select_endpoint(preferred, allow_fallback, repo):
    import httpx
    try:
        response = httpx.get(f"{preferred}/api/datasets/{repo}", follow_redirects=False, timeout=20)
        response.raise_for_status()
        return preferred
    except httpx.HTTPError as exc:
        if not allow_fallback or preferred.rstrip("/") == "https://huggingface.co":
            raise
        print(json.dumps({"event": "mirror_fallback", "reason": type(exc).__name__,
                          "endpoint": "https://huggingface.co"}), flush=True)
        return "https://huggingface.co"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reproduction/release.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--allow-official-fallback", action="store_true")
    parser.add_argument("--include-models", action="store_true", help="also download six adapters and Dynamic head/backbone, about 2.8 GB")
    parser.add_argument("--base-model", type=Path, help="existing pinned Qwen3-8B directory for generated training configurations")
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = select_endpoint(args.endpoint.rstrip("/"), args.allow_official_fallback, release["dataset_repo"])
    from huggingface_hub import hf_hub_download, snapshot_download

    tasks = [(entry, "dataset", release["dataset_repo"], release["dataset_revision"], root / "assets")
             for entry in release["dataset_files"]]
    tasks += [(entry, "model", release["model_repo"], release["model_revision"], root / "model-assets")
              for entry in release["model_code_files"]]
    tasks += [(entry, "model", release["model_repo"], release["model_revision"], root / "models")
              for entry in release.get("model_metadata_files", [])]
    if args.include_models:
        tasks += [(entry, "model", release["model_repo"], release["model_revision"], root / "models")
                  for entry in release.get("dynamic_weights", [])]

    def fetch(task):
        entry, kind, repo, revision, destination = task
        target = destination / entry["path"]
        if target.is_file() and digest(target) == entry["sha256"]:
            return
        try:
            source = Path(hf_hub_download(repo, entry["path"], repo_type=kind, revision=revision, endpoint=endpoint))
        except Exception as exc:
            if not args.allow_official_fallback or endpoint == "https://huggingface.co":
                raise
            print(json.dumps({"event": "mirror_file_fallback", "path": entry["path"],
                              "reason": type(exc).__name__}), flush=True)
            source = Path(hf_hub_download(repo, entry["path"], repo_type=kind, revision=revision,
                                          endpoint="https://huggingface.co"))
        if source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size differs: {entry['path']}")
        checked_copy(source, target, entry["sha256"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, tasks))
    for archive in release["replay_archives"]:
        extract_verified(root / "assets" / archive["path"], root / "bundle", archive["members"])
    # A writable working copy receives the original relative data layout expected by training.py.
    code = root / "model-code"
    for entry in release["model_code_files"]:
        relative = Path(entry["path"]).relative_to("code/vbench_prompts_compile")
        checked_copy(root / "model-assets" / entry["path"], code / relative, entry["sha256"])
    labels = root / "assets/dimensions/human_action/training/k400-labels.json"
    checked_copy(labels, code / "data/raw/k400/labels.json", digest(labels))
    for job in release["semantic_training"]:
        source = root / "assets" / job["dataset_directory"]
        for name in ["train.jsonl", "dev.jsonl"]:
            checked_copy(source / name, code / job["original_data_directory"] / name, digest(source / name))
        config = json.loads((source / "config.json").read_text())
        config["train_jsonl"] = str(code / job["original_data_directory"] / "train.jsonl")
        config["eval_jsonl"] = str(code / job["original_data_directory"] / "dev.jsonl")
        config["output_dir"] = str(root / "training-runs" / job["dimension"])
        config["model_name_or_path"] = str(args.base_model.resolve() if args.base_model else root / "base/Qwen3-8B")
        target = root / "training-configs" / (job["dimension"] + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(config, indent=2) + "\n")
    shared = root / "assets/provenance/training/object-color-20260920/records.json"
    checked_copy(shared, root / "object-color/semantics/records.json", digest(shared))
    if args.include_models:
        try:
            snapshot_download(release["model_repo"], revision=release["model_revision"], endpoint=endpoint,
                              allow_patterns=[dimension + "/*" for dimension in release["adapter_dimensions"]],
                              local_dir=root / "adapters")
        except Exception as exc:
            if not args.allow_official_fallback or endpoint == "https://huggingface.co":
                raise
            print(json.dumps({"event": "mirror_model_fallback", "reason": type(exc).__name__}), flush=True)
            snapshot_download(release["model_repo"], revision=release["model_revision"], endpoint="https://huggingface.co",
                              allow_patterns=[dimension + "/*" for dimension in release["adapter_dimensions"]],
                              local_dir=root / "adapters")
        for adapter in release["adapters"]:
            weight = root / "adapters" / adapter["path"]
            if weight.stat().st_size != adapter["bytes"] or digest(weight) != adapter["sha256"]:
                raise ValueError(f"Adapter content differs: {adapter['path']}")
    receipt = {"status": "restored_and_hash_verified", "endpoint": endpoint,
               "dataset_revision": release["dataset_revision"], "model_revision": release["model_revision"],
               "files_verified": len(tasks), "replay_members_verified": sum(len(a["members"]) for a in release["replay_archives"]),
               "adapters_downloaded": args.include_models, "training_started": False,
               "dynamic_weights_downloaded": args.include_models and bool(release.get("dynamic_weights")),
               "online_teacher_called": False}
    (root / "restore-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
