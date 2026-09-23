#!/usr/bin/env python3
"""Publish frozen counterfactual files in bounded tar shards from their source host.

No source is modified. Each archive preserves paths relative to the declared
source root. The index records every member, its SHA-256 and its archive.
Only temporary tar shards are removed after a successful upload. Deferred mode
preuploads shards and publishes the complete remaining batch in one commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def collect(spec):
    root = Path(spec["root"])
    files = set()
    excludes = set(spec.get("exclude_components", [])) | {".git", ".cache", "__pycache__", ".venv"}
    for include in spec["include"]:
        p = root / include
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_file():
            files.add(p)
            continue
        for folder, dirs, names in os.walk(p):
            dirs[:] = [d for d in dirs if d not in excludes]
            for name in names:
                f = Path(folder) / name
                rel = f.relative_to(root)
                if any(x in excludes for x in rel.parts):
                    continue
                if f.suffix in spec.get("exclude_suffixes", []):
                    continue
                if "historical-scores" in rel.parts and f.suffix == ".npz":
                    continue
                if f.is_file():
                    files.add(f)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def partition(files, limit):
    batch, size = [], 0
    for p in files:
        n = p.stat().st_size
        if batch and size + n > limit:
            yield batch
            batch, size = [], 0
        batch.append(p)
        size += n
    if batch:
        yield batch


def upload(args, source, destination, message):
    env = dict(os.environ, HF_ENDPOINT=args.endpoint)
    command = [args.hf, "upload", args.repo_id, str(source), destination,
               "--repo-type", "dataset", "--commit-message", message]
    for attempt in range(4):
        r = subprocess.run(command, env=env)
        if r.returncode == 0:
            return
        if attempt == 3:
            raise RuntimeError(f"hf upload failed: {destination}")
        time.sleep(5 * (attempt + 1))


def run(args):
    specs = json.loads(args.spec.read_text())["experiments"]
    args.work.mkdir(parents=True, exist_ok=True)
    deferred = getattr(args, "defer_commit", False)
    operations, pending_states = [], []
    if deferred:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi(endpoint=args.endpoint)
    for spec in specs:
        dimension, version = spec["dimension"], spec["version"]
        root = Path(spec["root"])
        files = collect(spec)
        snapshot = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns} for p in files]
        identity = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        job = args.work / dimension / version
        job.mkdir(parents=True, exist_ok=True)
        state_file = job / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {
            "source_snapshot_sha256": identity, "shard_mib": args.shard_mib, "archives": [], "members": []}
        if state["source_snapshot_sha256"] != identity:
            raise ValueError(f"Source changed during archive: {dimension}/{version}; freeze a new version")
        if state.get("shard_mib", 512) != args.shard_mib:
            raise ValueError("Shard size changed; resume with the original shard-mib setting")
        state.setdefault("pending_archives", [])
        state.setdefault("pending_members", [])
        if not deferred and state["pending_archives"]:
            raise ValueError("Resume pending preuploads with --defer-commit")
        destination = f"dimensions/{dimension}/counterfactual/{version}"
        print(json.dumps({"event": "experiment", "dimension": dimension, "version": version,
                          "files": len(files), "bytes": sum(x['bytes'] for x in snapshot)}), flush=True)
        for index, batch in enumerate(partition(files, args.shard_mib * 1024 ** 2)):
            name = f"data-{index:05d}.tar"
            if any(x["name"] == name for x in state["archives"]):
                continue
            archive = job / name
            members = []
            with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, dereference=True) as tf:
                for p in batch:
                    before = p.stat()
                    rel = p.relative_to(root).as_posix()
                    sha = digest(p)
                    info = tf.gettarinfo(str(p), arcname=rel)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with p.open("rb") as f:
                        tf.addfile(info, f)
                    after = p.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("Source changed during read: " + str(p))
                    members.append({"member": rel, "bytes": before.st_size, "sha256": sha,
                                    "archive": destination + "/" + name})
            entry = {"name": name, "path": destination + "/" + name, "bytes": archive.stat().st_size,
                     "sha256": digest(archive), "files": len(batch)}
            if deferred:
                previous = next((a for a in state["pending_archives"] if a["name"] == name), None)
                if previous is not None and previous != entry:
                    raise ValueError("Rebuilt pending archive identity changed: " + entry["path"])
                operation = CommitOperationAdd(path_in_repo=entry["path"], path_or_fileobj=archive)
                api.preupload_lfs_files(args.repo_id, additions=[operation], repo_type="dataset")
                if operation._upload_mode != "lfs" or not operation._is_uploaded:
                    raise RuntimeError("Archive was not confirmed in binary storage: " + entry["path"])
                operations.append(operation)
                if previous is None:
                    state["pending_archives"].append(entry)
                    state["pending_members"].extend(members)
            else:
                upload(args, archive, entry["path"], f"data({dimension}): archive {version} shard {index}")
                state["archives"].append(entry)
                state["members"].extend(members)
            dump(state_file, state)
            archive.unlink()
            print(json.dumps({"event": "shard_preuploaded" if deferred else "shard_uploaded", **entry}), flush=True)
        metadata = job / "metadata"
        metadata.mkdir(exist_ok=True)
        dump(metadata / "manifest.json", {"schema": "vbench-repair-counterfactual/1",
             "dimension": dimension, "version": version, "source_root": str(root),
             "source_snapshot_sha256": identity, "source_files": len(files),
             "shard_limit_mib": args.shard_mib,
             "source_bytes": sum(x["bytes"] for x in snapshot),
             "archives": state["archives"] + state["pending_archives"],
             "note": spec.get("note", "Frozen research artifacts; read source validation before reuse."),
             "excluded_components": spec.get("exclude_components", []),
             "excluded_suffixes": spec.get("exclude_suffixes", [])})
        (metadata / "files.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
            for x in state["members"] + state["pending_members"]))
        if spec.get("records_file"):
            source_records = root / spec["records_file"]
            records = [json.loads(line) for line in source_records.read_text().splitlines() if line.strip()]
            if spec.get("records_dimension"):
                records = [r for r in records if r.get("dimension") == spec["records_dimension"]]
            (metadata / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        (metadata / "README.md").write_text(f"# {dimension}: {version}\n\n"
            + spec.get("note", "Frozen research artifacts.")
            + "\n\nDownload every data-*.tar shard and extract into one empty directory. "
            "Archive members retain paths relative to the source root, so source-relative "
            "construction records remain interpretable. Absolute historical machine paths "
            "are provenance, not portable paths. `files.jsonl` resolves each member to its "
            "archive and SHA-256. Original inputs, encoding controls, derived videos, frames "
            "and construction parameters retain their original names and byte content.\n")
        if deferred:
            operations.extend(CommitOperationAdd(path_in_repo=destination + "/" + p.name,
                              path_or_fileobj=p) for p in sorted(metadata.iterdir()) if p.is_file())
            pending_states.append((state_file, state))
            print(json.dumps({"event": "experiment_prepared", "dimension": dimension, "version": version,
                              "files": len(files), "shards": len(state['archives']) + len(state['pending_archives'])}), flush=True)
        else:
            upload(args, metadata, destination, f"data({dimension}): publish {version} member hashes and provenance")
            print(json.dumps({"event": "experiment_complete", "dimension": dimension, "version": version,
                              "files": len(files), "shards": len(state['archives'])}), flush=True)
    if deferred and operations:
        api.preupload_lfs_files(args.repo_id, additions=operations, repo_type="dataset")
        not_before = getattr(args, "not_before", 0) or 0
        if time.time() < not_before:
            print(json.dumps({"event": "waiting_for_commit_window", "not_before": not_before}), flush=True)
        while time.time() < not_before:
            time.sleep(min(30, not_before - time.time()))
        result = api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=operations,
            commit_message=f"data: publish {len(specs)} counterfactual experiment archives and verified indexes")
        for state_file, state in pending_states:
            state["archives"].extend(state.pop("pending_archives"))
            state["members"].extend(state.pop("pending_members"))
            state["commit_sha"] = result.oid
            dump(state_file, state)
        print(json.dumps({"event": "counterfactuals_published", "experiments": len(specs),
                          "operations": len(operations), "sha": result.oid}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--repo-id", default="xju-arlab/vbench-repair")
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--hf", default="hf")
    p.add_argument("--shard-mib", type=int, default=512)
    p.add_argument("--defer-commit", action="store_true", help="Preupload all shards, then create one commit (requires huggingface_hub)")
    p.add_argument("--not-before", type=float, default=0, help="Earliest commit time as Unix seconds; preupload can proceed first")
    args = p.parse_args()
    if args.shard_mib < 1:
        p.error("shard-mib must be positive")
    run(args)


if __name__ == "__main__":
    main()
