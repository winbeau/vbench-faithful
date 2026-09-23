"""Deterministic resolution of a teacher pilot config into concrete requests.

The config is reviewable and small: it stores *selection rules* (task, length
bucket, pool, cue regex) plus the authorised limits. Prompt text is resolved from
the pinned raw snapshots at run time, so no upstream text has to be committed.
Resolution is reproducible: candidates are filtered by rule and sorted by
``(word_count, sha256)``, and a prompt is used at most once per pilot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .records import length_bucket, normalize_space, sha256_text, word_count
from .sources import REPO_ROOT, iter_flickr_captions, load_moviegen_prompts


class PilotError(RuntimeError):
    """Raised when a pilot config cannot be resolved without guessing."""


@dataclass(frozen=True)
class PoolEntry:
    text: str
    source: str
    source_file: str
    source_id: str
    group_id: str
    caption: str | None = None
    label: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def word_count(self) -> int:
        return word_count(self.text)


@dataclass(frozen=True)
class ResolvedItem:
    item_id: str
    group: str
    task: str
    bucket: str
    entry: PoolEntry
    note: str = ""

    @property
    def prompt(self) -> str:
        return self.entry.text

    @property
    def caption(self) -> str | None:
        return self.entry.caption

    def as_plan_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group": self.group,
            "task": self.task,
            "bucket_requested": self.bucket,
            "bucket_actual": length_bucket(self.entry.word_count),
            "word_count": self.entry.word_count,
            "char_count": len(self.entry.text),
            "source": self.entry.source,
            "source_file": self.entry.source_file,
            "source_id": self.entry.source_id,
            "group_id": self.entry.group_id,
            "prompt_sha256": self.entry.sha256,
            "prompt": self.entry.text,
            "caption": self.entry.caption,
            "caption_sha256": sha256_text(self.entry.caption) if self.entry.caption else None,
            "note": self.note,
        }


def load_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("pilot_id", "model", "limits", "items", "pools"):
        if key not in config:
            raise PilotError(f"pilot config missing {key!r}")
    return config


def _moviegen_pool(_: Mapping[str, Any]) -> list[PoolEntry]:
    entries = []
    for ref in load_moviegen_prompts():
        entries.append(
            PoolEntry(
                text=ref.text,
                source="moviegen",
                source_file=ref.source_file,
                source_id=f"line={ref.line}",
                group_id=f"moviegen:{ref.sha256[:12]}",
            )
        )
    return entries


def _flickr_pool(spec: Mapping[str, Any]) -> list[PoolEntry]:
    split = str(spec.get("split", "train"))
    entries = []
    for caption in iter_flickr_captions(split):
        entries.append(
            PoolEntry(
                text=caption.text,
                source="flickr30k_entities",
                source_file=f"Sentences/{caption.image_id}.txt",
                source_id=f"image={caption.image_id}#{caption.index}",
                group_id=f"flickr30k:{caption.image_id}",
                extra={"split": split},
            )
        )
    return entries


def _scene_fixture_pool(spec: Mapping[str, Any]) -> list[PoolEntry]:
    path = REPO_ROOT / str(spec["path"])
    entries: list[PoolEntry] = []
    families: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        family = str(row["family_id"])
        families.add(family)
        entries.append(
            PoolEntry(
                text=normalize_space(row["prompt"]),
                source="scene_fixture",
                source_file=str(spec["path"]),
                source_id=f"{family}#{row['label']}",
                group_id=f"scene-fixture:{family}",
                caption=normalize_space(row["caption"]),
                label=str(row["label"]),
                extra={"line": line_no},
            )
        )
    if len(families) < 3:
        raise PilotError(f"scene fixture pool too small: {len(families)} families")
    return entries


POOL_BUILDERS = {
    "moviegen": _moviegen_pool,
    "flickr": _flickr_pool,
    "scene_fixtures": _scene_fixture_pool,
}


def build_pools(config: Mapping[str, Any]) -> dict[str, list[PoolEntry]]:
    pools: dict[str, list[PoolEntry]] = {}
    for name, spec in config["pools"].items():
        kind = spec.get("kind")
        builder = POOL_BUILDERS.get(str(kind))
        if builder is None:
            raise PilotError(f"unknown pool kind {kind!r} for pool {name!r}")
        pools[name] = builder(spec)
    return pools


def resolve_items(config: Mapping[str, Any]) -> list[ResolvedItem]:
    pools = build_pools(config)
    cue_regex = {k: re.compile(v, re.IGNORECASE) for k, v in (config.get("cue_regex") or {}).items()}
    used_prompts: set[str] = set()
    resolved: list[ResolvedItem] = []
    for spec in config["items"]:
        item_id = str(spec["item_id"])
        pool_name = str(spec["pool"])
        if pool_name not in pools:
            raise PilotError(f"{item_id}: unknown pool {pool_name!r}")
        candidates: list[PoolEntry] = []
        for entry in pools[pool_name]:
            if spec.get("bucket") not in (None, "any") and length_bucket(entry.word_count) != spec["bucket"]:
                continue
            if spec.get("label_filter") is not None and entry.label != spec["label_filter"]:
                continue
            require_key = spec.get("require_regex_key")
            if require_key is not None:
                regex = cue_regex.get(str(require_key))
                if regex is None:
                    raise PilotError(f"{item_id}: unknown cue regex {require_key!r}")
                if not regex.search(entry.text):
                    continue
            exclude_key = spec.get("exclude_regex_key")
            if exclude_key is not None:
                regex = cue_regex.get(str(exclude_key))
                if regex is None:
                    raise PilotError(f"{item_id}: unknown cue regex {exclude_key!r}")
                if regex.search(entry.text):
                    continue
            candidates.append(entry)
        candidates.sort(key=lambda e: (e.word_count, e.sha256))
        chosen: list[PoolEntry] = []
        for entry in candidates:
            if entry.sha256 in used_prompts:
                continue
            chosen.append(entry)
            if len(chosen) >= int(spec.get("count", 1)):
                break
        if len(chosen) < int(spec.get("count", 1)):
            raise PilotError(
                f"{item_id}: only {len(chosen)} of {spec.get('count', 1)} candidates after filtering "
                f"(pool={pool_name}, bucket={spec.get('bucket')})"
            )
        for entry in chosen:
            used_prompts.add(entry.sha256)
            resolved.append(
                ResolvedItem(
                    item_id=item_id,
                    group=str(spec.get("group", "parse")),
                    task=str(spec["task"]),
                    bucket=str(spec.get("bucket", "any")),
                    entry=entry,
                    note=str(spec.get("note", "")),
                )
            )
    declared = int(config["limits"]["max_requests"])
    if len(resolved) > declared:
        raise PilotError(f"resolved {len(resolved)} items > declared max_requests {declared}")
    return resolved
