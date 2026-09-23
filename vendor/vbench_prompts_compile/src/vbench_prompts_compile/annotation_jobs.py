"""Durable annotation requests and identity checks; no model dependencies.

Successful HTTP responses are saved before the annotator parses them. Restarting
after either pass reuses that exact response, including rejected model answers.
Transport errors remain retryable; 401/403 stop the batch immediately.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .llm import ChatResponse, LLMError, RequestBudget, utc_now
from .records import canonical_json, normalize_space, sha256_text


class ProviderUnavailable(RuntimeError):
    """A provider rejected credentials/balance; do not repeat for every item."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def bind_job(path: Path, identity: dict) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != identity:
        raise ValueError("job identity changed; preserve the old output and use a new path")
    atomic_json(path, identity)


@contextmanager
def job_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another writer holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def observation_id(row: dict) -> str:
    """Do not use legacy sample IDs: many omit the evidence video/frame."""
    meta = row.get("meta", {})
    identity = {
        "prompt": normalize_space(row["input"]["prompt"]),
        "caption": normalize_space(row["input"]["caption"]),
        "video": meta.get("frame_video"),
        "frame_index": meta.get("frame_index"),
        "image": meta.get("image"),
        "frame_source_prompt": meta.get("frame_source_prompt"),
    }
    return "scene-observation-" + sha256_text(canonical_json(identity))[:24]


def merge_observations(rows: list[dict]) -> tuple[list[dict], dict]:
    """Merge exact observations, retaining every old ID and conflicting label.

    Conflicting visual labels become unknown; evidence supervision can still be
    independently obtained from the text. Never choose whichever label came first.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(observation_id(row), []).append(row)
    merged = []
    conflicts = 0
    for identity, originals in grouped.items():
        labels = sorted({r.get("meta", {}).get("visual_truth", r.get("target")) for r in originals} - {None})
        conflicts += len(labels) > 1
        row = {**originals[0], "sample_id": identity}
        row["meta"] = {
            **row.get("meta", {}),
            "legacy_sample_ids": sorted({r["sample_id"] for r in originals}),
            "merged_rows": len(originals),
            "visual_labels_observed": labels,
            "visual_truth": labels[0] if len(labels) == 1 else None,
            "visual_truth_conflict": len(labels) > 1,
        }
        merged.append(row)
    return merged, {"input_rows": len(rows), "observations": len(merged),
                    "duplicates_merged": len(rows) - len(merged), "visual_conflicts": conflicts}


class ResumableChatClient:
    def __init__(self, client, budget: RequestBudget, journal: Path):
        self.client = client
        self.provider, self.model = client.provider, client.model
        self.budget, self.journal = budget, journal
        self.cache: dict[str, dict] = {}
        self.cache_hits = 0
        for entry in read_jsonl(journal):
            if entry.get("response") is not None:
                self.cache[entry["request_id"]] = entry["response"]

    def chat(self, **kwargs) -> ChatResponse:
        identity = dict(kwargs)
        identity["images"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in kwargs.get("images", [])]
        request_id = sha256_text(canonical_json({"provider": self.provider, "model": self.model,
            "base_url": getattr(self.client, "base_url", None),
            "extra_payload": getattr(self.client, "extra_payload", {}),
            "json_mode": getattr(self.client, "json_mode", True), **identity}))
        if request_id in self.cache:
            self.cache_hits += 1
            return ChatResponse(**self.cache[request_id])
        self.budget.charge({"request_id": request_id, "model": self.model}, provider=self.provider)
        try:
            response = self.client.chat(**kwargs)
        except LLMError as error:
            append_jsonl(self.journal, {"request_id": request_id, "utc": utc_now(), "error": str(error)})
            if "HTTP 401" in str(error) or "HTTP 403" in str(error):
                raise ProviderUnavailable(str(error)) from error
            raise
        metadata = asdict(response)
        append_jsonl(self.journal, {"request_id": request_id, "utc": utc_now(), "response": metadata})
        self.cache[request_id] = metadata
        self.budget.append({"kind": "response", "request_id": request_id, "provider": self.provider,
                            **response.as_metadata(), "utc": utc_now()})
        return response

    def usage(self) -> dict:
        return {"successful_requests": len(self.cache), "cache_hits_this_run": self.cache_hits,
                "prompt_tokens": sum(r["prompt_tokens"] for r in self.cache.values()),
                "completion_tokens": sum(r["completion_tokens"] for r in self.cache.values()),
                "currency_cost": None, "cost_note": "provider billing amount not supplied; token usage recorded"}
