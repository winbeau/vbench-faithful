"""Content-addressed evidence and shared metadata envelopes; no metric logic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_json_artifact(directory: str | Path, kind: str, value) -> dict:
    body = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{kind}-{digest}.json"
    try:
        with destination.open("xb") as handle:
            handle.write(body)
    except FileExistsError:
        if destination.read_bytes() != body:
            raise ValueError(f"artifact hash collision or corrupt evidence: {destination}")
    return {"path": str(destination.resolve()), "sha256": digest}


def dimension_metadata(entry: dict, dimension: str) -> dict:
    value = entry.get("dimension_metadata", {}).get(dimension)
    if value is None:
        value = entry.get("auxiliary_info", {}).get(dimension, {})
    if not isinstance(value, dict):
        raise ValueError(f"{dimension} metadata must be an object")
    return value


def result_identity(video: Path, entry: dict) -> dict:
    return {"video_uid": entry.get("video_uid", hashlib.sha256(str(video.resolve()).encode()).hexdigest()),
            "query_uid": entry.get("query_uid"), "base_id": entry.get("base_id"),
            "prompt": entry.get("prompt")}
