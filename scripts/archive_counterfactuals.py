#!/usr/bin/env python3
"""Publish frozen counterfactual files in bounded tar shards from their source host.

No source is modified. Each archive preserves paths relative to the declared
source root. The index records every member, its SHA-256 and its archive.
Only temporary tar shards are removed after a successful hf upload.
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
            "source_snapshot_sha256": identity, "archives": [], "members": []}
        if state["source_snapshot_sha256"] != identity:
            raise ValueError(f"Source changed during archive: {dimension}/{version}; freeze a new version")
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
            upload(args, archive, entry["path"], f"data({dimension}): archive {version} shard {index}")
            state["archives"].append(entry)
            state["members"].extend(members)
            dump(state_file, state)
            archive.unlink()
            print(json.dumps({"event": "shard_uploaded", **entry}), flush=True)
        metadata = job / "metadata"
        metadata.mkdir(exist_ok=True)
        dump(metadata / "manifest.json", {"schema": "vbench-repair-counterfactual/1",
             "dimension": dimension, "version": version, "source_root": str(root),
             "source_snapshot_sha256": identity, "source_files": len(files),
             "source_bytes": sum(x["bytes"] for x in snapshot), "archives": state["archives"],
             "note": spec.get("note", "Frozen research artifacts; read source validation before reuse."),
             "excluded_components": spec.get("exclude_components", []),
             "excluded_suffixes": spec.get("exclude_suffixes", [])})
        (metadata / "files.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in state["members"]))
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
        upload(args, metadata, destination, f"data({dimension}): publish {version} member hashes and provenance")
        print(json.dumps({"event": "experiment_complete", "dimension": dimension, "version": version,
                          "files": len(files), "shards": len(state['archives'])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--repo-id", default="xju-arlab/vbench-repair")
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--hf", default="hf")
    p.add_argument("--shard-mib", type=int, default=512)
    args = p.parse_args()
    if args.shard_mib < 1:
        p.error("shard-mib must be positive")
    run(args)


if __name__ == "__main__":
    main()
