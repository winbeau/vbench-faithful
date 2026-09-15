#!/usr/bin/env python3
"""Download the frozen VBench sampled-video packs one dimension at a time.

The remote path is always ``<model-pack>/<dimension>``.  This script refuses
to substitute another dimension when a requested directory is absent, uses
hf-mirror.com, resumes partial files, and writes a hash manifest after each
dimension.  It does not require torch or a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "dynamic_degree",
    "subject_consistency",
    "human_action",
    "spatial_relationship",
    "scene",
    "multiple_objects",
    "overall_consistency",
    "motion_smoothness",
)
DEFAULT_PACKS = ("cogvideo", "lavie", "modelscope", "videocrafter")


def api_json(url: str, token: str | None) -> Any:
    headers = {"User-Agent": "vbench-audit-official-video-fetch/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def list_files(endpoint: str, repo_id: str, model_pack: str, dimension: str, token: str | None) -> list[str]:
    path = f"{model_pack}/{dimension}"
    encoded = urllib.parse.quote(path, safe="/")
    query = urllib.parse.urlencode({"recursive": "true", "limit": "1000"})
    url = f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/tree/main/{encoded}?{query}"
    payload = api_json(url, token)
    if not isinstance(payload, list):
        raise RuntimeError(f"HF tree API failed for {path}: {payload!r}")
    files = [str(item["path"]) for item in payload if item.get("type") == "file" and str(item.get("path", "")).lower().endswith((".mp4", ".gif"))]
    return sorted(files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(endpoint: str, repo_id: str, remote_path: str, destination: Path, token: str | None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    encoded = urllib.parse.quote(remote_path, safe="/")
    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{encoded}?download=true"
    command = [
        "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "2",
        "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url,
    ]
    if token:
        command[1:1] = ["--header", f"Authorization: Bearer {token}"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {"path": remote_path, "status": "failed", "returncode": result.returncode, "stderr": result.stderr[-1000:]}
    if not partial.is_file() or partial.stat().st_size == 0:
        return {"path": remote_path, "status": "failed", "error": "empty_download"}
    partial.replace(destination)
    return {"path": remote_path, "status": "downloaded", "size_bytes": destination.stat().st_size, "sha256": sha256(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=DIMENSIONS, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--repo-id", default="Vchitect/VBench_sampled_video")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--model-pack", action="append", dest="model_packs", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    packs = tuple(args.model_packs or DEFAULT_PACKS)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_candidates = []
    if os.environ.get("HF_TOKEN_PATH"):
        token_candidates.append(Path(os.environ["HF_TOKEN_PATH"]).expanduser())
    if os.environ.get("HF_HOME"):
        token_candidates.append(Path(os.environ["HF_HOME"]).expanduser() / "token")
    token_candidates.append(Path.home() / ".cache/huggingface/token")
    if not token:
        for candidate in token_candidates:
            try:
                if candidate.is_file():
                    token = candidate.read_text(encoding="utf-8").strip()
                    if token:
                        break
            except OSError:
                continue
    if not token and not args.dry_run:
        raise RuntimeError("gated VBench dataset requires HF_TOKEN or a readable Hugging Face token cache")
    plan: list[tuple[str, str, Path]] = []
    for pack in packs:
        for remote_path in list_files(args.endpoint, args.repo_id, pack, args.dimension, token):
            relative = Path(remote_path)
            plan.append((pack, remote_path, args.dest / relative))
    if not plan:
        raise RuntimeError(f"no video files found for the exact remote dimension directory {args.dimension!r}; no substitute directory was used")
    manifest_dir = args.dest / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.dimension}.jsonl"
    print(json.dumps({"event": "plan", "dimension": args.dimension, "packs": packs, "files": len(plan), "dest": str(args.dest)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, args.endpoint, args.repo_id, remote, destination, token): (pack, remote, destination)
            for pack, remote, destination in plan
        }
        for future in as_completed(futures):
            pack, remote, destination = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # keep the dimension resumable
                record = {"path": remote, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            record.update({"dimension": args.dimension, "model_pack": pack, "local_path": str(destination)})
            records.append(record)
            print(json.dumps({"event": "file", **record}, ensure_ascii=False), flush=True)
    records.sort(key=lambda row: row["path"])
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    failures = [record for record in records if record.get("status") != "downloaded"]
    print(json.dumps({"event": "complete", "dimension": args.dimension, "files": len(records), "failures": len(failures), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
