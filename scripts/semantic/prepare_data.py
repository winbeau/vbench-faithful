#!/usr/bin/env python3
"""Clean the pinned raw snapshots into a versioned, ignored build.

Design rules:

* ``data/raw`` is read-only; nothing here downloads, extracts media or trains.
* every record passes the shared contract in ``records.py`` before it is written;
  rejects go to ``quarantine`` with a reason, never silently dropped;
* counts must close: scanned = retained + quarantined + rule_filtered;
* engineering fixtures are written to a *separate* tree (``data/smoke``) so they
  can never be mistaken for research data.

Usage examples::

    uv run --no-sync python scripts/semantic/prepare_data.py --source k400
    uv run --no-sync python scripts/semantic/prepare_data.py --source visual_genome --limit 400
    uv run --no-sync python scripts/semantic/prepare_data.py --source all --build-id local-0001
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/prompt-compiler/src"))

sys.path.insert(0, str(ROOT))
from scripts.semantic.paths import DATA_ROOT, RAW_ROOT, FIXTURES_ROOT
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import sources as S  # noqa: E402

DEFAULT_BUILD_ID = "local-0001"
FIXTURES = FIXTURES_ROOT
SCENE_LABELS = ("supported", "contradicted", "insufficient")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


class Build:
    """Accumulates records/quarantine/inventory and writes one build directory."""

    def __init__(self, build_id: str, *, limit: int, seed: int, root: Path | None = None) -> None:
        self.build_id = build_id
        self.limit = limit
        self.seed = seed
        self.root = Path(root) if root else ROOT / 'output/paper-data'
        self.processed = self.root / "data" / "processed" / build_id
        self.smoke = self.root / "data" / "smoke" / build_id
        self.candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.quarantine: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.counts: dict[str, Any] = {}
        self.sources_used: dict[str, Any] = {}
        self._action_resolver: Any = None

    # -- helpers ---------------------------------------------------------
    def add_record(self, record: dict[str, Any]) -> bool:
        """Validate and store; invalid records are quarantined with reasons."""
        validation = R.validate_record(record, action_resolver=self._action_resolver)
        if not validation.ok:
            self.add_quarantine(
                record["task"],
                reason=";".join(validation.errors),
                source=record["source"],
                source_id=record["source_id"],
                payload={"input": record["input"], "target": record["target"]},
            )
            return False
        stored = R.add_length_meta(dict(record))
        stored.setdefault("meta", {})
        if validation.warnings:
            stored["meta"]["validation_warnings"] = sorted(set(validation.warnings))
        self.candidates[record["task"]].append(stored)
        return True

    def add_quarantine(self, task: str, *, reason: str, source: str, source_id: str, payload: Mapping[str, Any]) -> None:
        self.quarantine[task].append(
            {"task": task, "reason": reason, "source": source, "source_id": str(source_id), **dict(payload)}
        )

    def add_inventory(self, name: str, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        self.inventory[name].extend(rows)
        return len(rows)

    # -- writing ---------------------------------------------------------
    def write(self, *, parameters: Mapping[str, Any]) -> dict[str, Any]:
        self.processed.mkdir(parents=True, exist_ok=True)
        self.smoke.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, Any]] = {}
        for task, rows in sorted(self.candidates.items()):
            rows.sort(key=lambda r: r["sample_id"])
            path = self.processed / "candidates" / f"{task}.jsonl"
            R.write_jsonl(path, rows)
            files[str(path.relative_to(self.processed))] = {"count": len(rows), "sha256": sha256_file(path)}
        for task, rows in sorted(self.quarantine.items()):
            rows.sort(key=lambda r: (r["reason"], r["source_id"]))
            path = self.processed / "quarantine" / f"{task}.jsonl"
            R.write_jsonl(path, rows)
            files[str(path.relative_to(self.processed))] = {"count": len(rows), "sha256": sha256_file(path)}
        for name, rows in sorted(self.inventory.items()):
            rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
            path = self.processed / "inventory" / f"{name}.jsonl"
            R.write_jsonl(path, rows)
            files[str(path.relative_to(self.processed))] = {"count": len(rows), "sha256": sha256_file(path)}
        counts = {
            "build_id": self.build_id,
            "candidates": {task: len(rows) for task, rows in sorted(self.candidates.items())},
            "quarantine": {task: len(rows) for task, rows in sorted(self.quarantine.items())},
            "inventory": {name: len(rows) for name, rows in sorted(self.inventory.items())},
            "per_source": self.counts,
        }
        (self.processed / "counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        smoke_files = {
            str(path.relative_to(self.smoke)): {"count": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "sha256": sha256_file(path)}
            for path in sorted(self.smoke.glob("*.jsonl"))
        }
        manifest = {
            "build_id": self.build_id,
            "parameters": dict(parameters),
            "git_sha": git_sha(),
            "sources": self.sources_used,
            "outputs": files,
            "smoke_outputs": smoke_files,
            "note": "candidates/quarantine/inventory are ignored build artifacts; no research train split is produced here",
        }
        (self.processed / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return counts


class _Resolver:
    """K400 resolver wrapper so records.validate can normalise actions."""

    def __init__(self, vocab: S.K400Vocabulary | None) -> None:
        self.vocab = vocab

    def __call__(self, name: str) -> str | None:
        return None if self.vocab is None else self.vocab.resolve(name)


# ---------------------------------------------------------------------------
# K400 -> action weak supervision
# ---------------------------------------------------------------------------

def gerund_label(label: str) -> bool:
    first = R.normalize_phrase(label).split(" ")[0]
    return bool(first) and first.endswith("ing")


def prepare_k400(build: Build, vocab: S.K400Vocabulary) -> None:
    scanned = retained = quarantined = awkward = 0
    for entry in vocab.entries:
        scanned += 1
        label = entry["label"]
        if gerund_label(label):
            prompt = f"A person is {label}."
            template = "person_is_gerund_v1"
        else:
            prompt = f"The video shows {label}."
            template = "video_shows_noun_v1"
        try:
            record = R.make_record(
                task="action",
                sample_input={"prompt": prompt},
                target={"actions": [label]},
                source="k400_template",
                source_id=f"k400-{entry['id']}",
                group_id=f"k400-class-{entry['id']}",
                quality="synthetic_weak",
                meta={"k400_id": entry["id"], "template": template, "grammar_review": "ok" if template.endswith("gerund_v1") else "required"},
                action_resolver=build._action_resolver,
            )
        except R.RecordError as error:
            quarantined += 1
            build.add_quarantine("action", reason=";".join(error.errors), source="k400_template", source_id=f"k400-{entry['id']}", payload={"target": {"actions": [label]}})
            continue
        if record["meta"]["grammar_review"] != "ok":
            awkward += 1
        build.candidates["action"].append(R.add_length_meta(record))
        retained += 1
    build.counts["k400_template"] = {
        "scanned": scanned,
        "retained": retained,
        "quarantined": quarantined,
        "non_gerund_templates": awkward,
        "closure": scanned == retained + quarantined,
        "natural_prompt_pairs": 0,
    }


# ---------------------------------------------------------------------------
# Visual Genome -> spatial weak supervision
# ---------------------------------------------------------------------------

VG_TEMPLATES = {
    "left": "The {subject} is to the left of the {object}.",
    "right": "The {subject} is to the right of the {object}.",
    "above": "The {subject} is above the {object}.",
    "below": "The {subject} is below the {object}.",
}


def vg_prompt(subject: str, relation: str, obj: str) -> str:
    return VG_TEMPLATES[relation].format(subject=subject.lower(), object=obj.lower())


def prepare_visual_genome(build: Build, limit: int, seed: int) -> None:
    per_direction = max(1, limit // 4)
    rng = random.Random(seed)
    by_direction: dict[str, list[S.VGEdge]] = defaultdict(list)
    scanned = 0
    for edge in S.iter_vg_edges(limit=limit * 40):
        scanned += 1
        if len(by_direction[edge.direction]) < per_direction * 4:
            by_direction[edge.direction].append(edge)
    chosen: list[S.VGEdge] = []
    for direction, edges in by_direction.items():
        rng.shuffle(edges)
        chosen.extend(edges[:per_direction])
    chosen.sort(key=lambda e: (e.direction, e.image_id, str(e.relationship_id)))
    retained = quarantined = 0
    multi_by_image: dict[int, list[S.VGEdge]] = defaultdict(list)
    for edge in chosen:
        prompt = vg_prompt(edge.subject, edge.direction, edge.object)
        try:
            record = R.make_record(
                task="spatial",
                sample_input={"prompt": prompt},
                target={"relationships": [{"subject": edge.subject, "relation": edge.direction, "object": edge.object}]},
                source="visual_genome",
                source_id=f"{edge.image_id}:{edge.relationship_id}",
                group_id=f"vg-image-{edge.image_id}",
                quality="synthetic_weak",
                meta={
                    "raw_predicate": edge.predicate,
                    "template": "vg_direction_v1",
                    "direction": edge.direction,
                    "subject_object_id": edge.subject_id,
                    "object_object_id": edge.object_id,
                },
            )
        except R.RecordError as error:
            quarantined += 1
            build.add_quarantine("spatial", reason=";".join(error.errors), source="visual_genome", source_id=f"{edge.image_id}:{edge.relationship_id}", payload={"edge": {"subject": edge.subject, "relation": edge.direction, "object": edge.object}})
            continue
        build.candidates["spatial"].append(R.add_length_meta(record))
        retained += 1
        multi_by_image[edge.image_id].append(edge)
    # Multi-relation candidates: only edges from the same image, capped and flagged.
    multi_limit = max(0, min(50, limit // 8))
    multi_kept = 0
    for image_id in sorted(multi_by_image):
        edges = multi_by_image[image_id]
        if len(edges) < 2 or multi_kept >= multi_limit:
            continue
        first, second = edges[0], edges[1]
        if first.direction == second.direction:
            continue
        prompt = f"{vg_prompt(first.subject, first.direction, first.object)} {vg_prompt(second.subject, second.direction, second.object)}"
        try:
            record = R.make_record(
                task="spatial",
                sample_input={"prompt": prompt},
                target={
                    "relationships": [
                        {"subject": first.subject, "relation": first.direction, "object": first.object},
                        {"subject": second.subject, "relation": second.direction, "object": second.object},
                    ]
                },
                source="visual_genome",
                source_id=f"{image_id}:multi:{first.relationship_id}:{second.relationship_id}",
                group_id=f"vg-image-{image_id}",
                quality="synthetic_weak",
                meta={"template": "vg_direction_v1_multi", "review": "required", "edges": [first.predicate, second.predicate]},
            )
        except R.RecordError as error:
            quarantined += 1
            build.add_quarantine("spatial", reason=";".join(error.errors), source="visual_genome", source_id=f"{image_id}:multi", payload={"edges": [first.predicate, second.predicate]})
            continue
        build.candidates["spatial"].append(R.add_length_meta(record))
        multi_kept += 1
    direction_counts = Counter(edge.direction for edge in chosen)
    build.counts["visual_genome"] = {
        "scanned_whitelisted_edges": scanned,
        "sampled_edges": len(chosen),
        "retained_single": retained,
        "retained_multi": multi_kept,
        "quarantined": quarantined,
        "closure": len(chosen) == retained + quarantined,
        "sampled_direction_counts": dict(direction_counts),
        "left_right_scarce": True,
    }


# ---------------------------------------------------------------------------
# Visual Genome -> objects weak supervision (licence-clean annotation source)
# ---------------------------------------------------------------------------

def prepare_vg_objects(build: Build, limit: int, seed: int) -> None:
    """Entity extraction from the same whitelisted edges used for spatial.

    The prompt is a template sentence and both endpoints are, by construction,
    mentioned in it; this is weak supervision, not human object annotation.
    """
    rng = random.Random(seed + 1)
    edges = list(S.iter_vg_edges(limit=limit))
    rng.shuffle(edges)
    scanned = retained = quarantined = 0
    by_image: dict[int, list[S.VGEdge]] = defaultdict(list)
    for edge in edges:
        scanned += 1
        by_image[edge.image_id].append(edge)
        entities = [edge.subject, edge.object]
        prompt = vg_prompt(edge.subject, edge.direction, edge.object)
        try:
            record = R.make_record(
                task="objects",
                sample_input={"prompt": prompt},
                target={"entities": entities},
                source="visual_genome_objects",
                source_id=f"{edge.image_id}:{edge.relationship_id}",
                group_id=f"vg-image-{edge.image_id}",
                quality="synthetic_weak",
                meta={"derivation": "vg_relation_endpoints", "raw_predicate": edge.predicate, "template": "vg_direction_v1"},
            )
        except R.RecordError as error:
            quarantined += 1
            build.add_quarantine("objects", reason=";".join(error.errors), source="visual_genome_objects", source_id=f"{edge.image_id}:{edge.relationship_id}", payload={"target": {"entities": entities}})
            continue
        build.candidates["objects"].append(R.add_length_meta(record))
        retained += 1
    # A few four-entity prompts built from two edges of one image, same group.
    multi = 0
    for image_id in sorted(by_image):
        group = by_image[image_id]
        if len(group) < 2 or multi >= max(1, limit // 20):
            continue
        first, second = group[0], group[1]
        prompt = f"{vg_prompt(first.subject, first.direction, first.object)} {vg_prompt(second.subject, second.direction, second.object)}"
        entities, _ = R.dedup_preserving_order([first.subject, first.object, second.subject, second.object])
        if len(entities) < 3:
            continue
        try:
            record = R.make_record(
                task="objects",
                sample_input={"prompt": prompt},
                target={"entities": entities},
                source="visual_genome_objects",
                source_id=f"{image_id}:multi:{first.relationship_id}:{second.relationship_id}",
                group_id=f"vg-image-{image_id}",
                quality="synthetic_weak",
                meta={"derivation": "vg_relation_endpoints_multi", "review": "required"},
            )
        except R.RecordError:
            continue
        build.candidates["objects"].append(R.add_length_meta(record))
        multi += 1
    build.counts["visual_genome_objects"] = {
        "scanned_edges": scanned,
        "retained": retained,
        "retained_multi": multi,
        "quarantined": quarantined,
        "closure": scanned == retained + quarantined,
    }


# ---------------------------------------------------------------------------
# Flickr30k Entities -> objects candidates (licence pending)
# ---------------------------------------------------------------------------

def prepare_flickr(build: Build, limit: int) -> None:
    scanned = retained = quarantined = rule_filtered = 0
    for caption in S.iter_flickr_captions("train", limit=None):
        if retained >= limit:
            break
        scanned += 1
        entities: list[str] = []
        flagged: list[str] = []
        excluded: list[str] = []
        for mention in caption.mentions:
            if set(mention.types) & S.FLICKR_NON_OBJECT_TYPES or not mention.positioned:
                excluded.append(mention.phrase)
                continue
            if set(mention.types) & S.FLICKR_FLAGGED_TYPES:
                flagged.append(mention.phrase)
            entities.append(mention.phrase)
        rule_filtered += len(excluded)
        if not entities:
            quarantined += 1
            build.add_quarantine("objects", reason="no_entities_after_rules", source="flickr30k_entities", source_id=f"{caption.image_id}#{caption.index}", payload={"input": {"prompt": caption.text}})
            continue
        try:
            record = R.make_record(
                task="objects",
                sample_input={"prompt": caption.text},
                target={"entities": entities},
                source="flickr30k_entities",
                source_id=f"{caption.image_id}#{caption.index}",
                group_id=f"flickr30k-image-{caption.image_id}",
                quality="synthetic_weak",
                meta={
                    "split": caption.split,
                    "licence_status": "annotation_licence_unclear",
                    "train_ready": False,
                    "flagged_mentions": flagged,
                    "excluded_mentions": excluded,
                    "chain_ids": [m.chain for m in caption.mentions],
                },
            )
        except R.RecordError as error:
            quarantined += 1
            build.add_quarantine("objects", reason=";".join(error.errors), source="flickr30k_entities", source_id=f"{caption.image_id}#{caption.index}", payload={"input": {"prompt": caption.text}, "target": {"entities": entities}})
            continue
        build.candidates["objects"].append(R.add_length_meta(record))
        retained += 1
    build.counts["flickr30k_entities"] = {
        "scanned_captions": scanned,
        "retained": retained,
        "quarantined": quarantined,
        "mentions_rule_filtered": rule_filtered,
        "closure": scanned == retained + quarantined,
        "train_ready": False,
        "reason": "annotation licence not cleared; candidates only",
    }


# ---------------------------------------------------------------------------
# Visual Genome region descriptions -> objects weak supervision (licence-clean)
# ---------------------------------------------------------------------------

REGION_PROMPT_TEMPLATES = {
    1: "A scene showing {p1}.",
    2: "A scene showing {p1} and {p2}.",
    3: "A scene showing {p1}, {p2} and {p3}.",
}


def prepare_vg_regions(build: Build, limit: int, seed: int) -> None:
    """Objects targets from human-written region phrases (replaces Flickr, B4).

    The phrases are human annotations (CC-BY-4.0); only the lead-in sentence is a
    template, so these records stay ``synthetic_weak`` but carry natural object
    wording instead of an artificial relation sentence.
    """
    rng = random.Random(seed + 2)
    pool = list(S.iter_vg_regions(limit=max(limit * 2, 200)))
    rng.shuffle(pool)
    scanned = retained = quarantined = 0
    for region_set in pool:
        if retained >= limit:
            break
        scanned += 1
        phrases = region_set.phrases[:3]
        if not phrases:
            continue
        key = "_".join(str(index) for index, _ in enumerate(phrases, start=1))
        prompt = REGION_PROMPT_TEMPLATES[len(phrases)].format(**{f"p{i}": phrase for i, phrase in enumerate(phrases, start=1)})
        entities, _ = R.dedup_preserving_order([R.head_noun(phrase) for phrase in phrases])
        entities = [R.canonical_entity_name(name) for name in entities]
        try:
            record = R.make_record(
                task="objects",
                sample_input={"prompt": prompt},
                target={"entities": entities},
                source="visual_genome_regions",
                source_id=f"{region_set.image_id}:{key}",
                group_id=f"vg-image-{region_set.image_id}",
                quality="synthetic_weak",
                meta={
                    "construction": "region_phrases_head_nouns",
                    "region_phrases": list(phrases),
                    "licence": "CC-BY-4.0",
                },
            )
        except R.RecordError as error:
            quarantined += 1
            build.add_quarantine("objects", reason=";".join(error.errors), source="visual_genome_regions", source_id=str(region_set.image_id), payload={"input": {"prompt": prompt}})
            continue
        build.candidates["objects"].append(R.add_length_meta(record))
        retained += 1
    build.counts["visual_genome_regions"] = {
        "scanned_images": scanned,
        "retained": retained,
        "quarantined": quarantined,
        "closure": scanned == retained + quarantined,
        "licence": "CC-BY-4.0",
    }


# ---------------------------------------------------------------------------
# MovieGen -> unlabeled prompt inventory
# ---------------------------------------------------------------------------

def prepare_moviegen(build: Build) -> None:
    refs = S.load_moviegen_prompts()
    rows = [
        {
            "source": "moviegen",
            "source_file": ref.source_file,
            "line": ref.line,
            "sha256": ref.sha256,
            "word_count": ref.word_count,
            "length_bucket": R.length_bucket(ref.word_count),
            "text": ref.text,
            "training_status": "unlabeled_candidate_no_target",
            "licence": "CC-BY-NC-4.0",
        }
        for ref in refs
    ]
    build.add_inventory("moviegen", rows)
    build.counts["moviegen"] = {
        "unique_prompts": len(refs),
        "buckets": dict(Counter(row["length_bucket"] for row in rows)),
        "labels_generated": 0,
        "note": "prompt pool only; concept tags are not targets",
    }


# ---------------------------------------------------------------------------
# SNLI -> auxiliary NLI material (never scene targets)
# ---------------------------------------------------------------------------

def prepare_snli(build: Build, limit: int | None = None) -> None:
    rows, status = S.load_snli_split("train")
    if status != "ok" or rows is None:
        build.counts["snli"] = {"status": status, "note": "pyarrow unavailable in this interpreter; run in the training environment"}
        return
    seen: set[tuple[str, str]] = set()
    conflicts: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        key = (row["premise"], row["hypothesis"])
        conflicts[key].add(int(row["label"]))
    conflict_keys = {key for key, labels in conflicts.items() if len(labels) > 1}
    kept: list[dict[str, Any]] = []
    dropped_unlabeled = dropped_duplicate = dropped_conflict = 0
    for row in rows:
        premise, hypothesis, label = row["premise"], row["hypothesis"], int(row["label"])
        if label < 0 or not premise.strip() or not hypothesis.strip():
            dropped_unlabeled += 1
            continue
        key = (premise, hypothesis)
        if key in conflict_keys:
            dropped_conflict += 1
            continue
        if key in seen:
            dropped_duplicate += 1
            continue
        seen.add(key)
        kept.append(
            {
                "source": "snli",
                "split": "train",
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "usage": "auxiliary_nli_not_scene_target",
            }
        )
        if limit is not None and len(kept) >= limit:
            break
    build.add_inventory("snli_auxiliary", kept)
    build.counts["snli"] = {
        "status": "ok",
        "scanned": len(rows),
        "kept": len(kept),
        "dropped_unlabeled_or_empty": dropped_unlabeled,
        "dropped_duplicate": dropped_duplicate,
        "dropped_label_conflict": dropped_conflict,
        "conflict_text_pairs": len(conflict_keys),
        "target_usage": "none; auxiliary NLI only",
    }


# ---------------------------------------------------------------------------
# Engineering fixtures -> data/smoke (never research data)
# ---------------------------------------------------------------------------

def prepare_engineering_fixtures(build: Build) -> None:
    summary: dict[str, Any] = {}
    tasks = ("spatial", "action", "objects")
    for task in tasks:
        path = FIXTURES / f"{task}_engineering.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out: list[dict[str, Any]] = []
        for row in rows:
            record = R.make_record(
                task=task,
                sample_input={"prompt": row["prompt"]},
                target=row["target"],
                source="engineering_fixture",
                source_id=row["fixture_id"],
                group_id=f"fixture-{row['fixture_id']}",
                quality="engineering_only",
                meta={"note": row.get("note", "")},
                action_resolver=build._action_resolver,
            )
            out.append(R.add_length_meta(record))
        target_path = build.smoke / f"{task}.jsonl"
        R.write_jsonl(target_path, out)
        summary[task] = {"count": len(out), "sha256": sha256_file(target_path)}

    scene_path = FIXTURES / "scene_engineering.jsonl"
    scene_rows = [json.loads(line) for line in scene_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scene_out: list[dict[str, Any]] = []
    families: set[str] = set()
    labels: Counter[str] = Counter()
    for row in scene_rows:
        families.add(row["family_id"])
        labels[row["label"]] += 1
        record = R.make_record(
            task="scene",
            sample_input={"prompt": row["prompt"], "caption": row["caption"]},
            target=row["label"],
            source="engineering_fixture",
            source_id=f"{row['family_id']}#{row['label']}",
            group_id=f"scene-fixture:{row['family_id']}",
            quality="engineering_only",
            meta={"fixture_label": row["label"], "family_id": row["family_id"]},
        )
        scene_out.append(R.add_length_meta(record))
    target_path = build.smoke / "scene.jsonl"
    R.write_jsonl(target_path, scene_out)
    summary["scene"] = {
        "count": len(scene_out),
        "families": len(families),
        "labels": dict(labels),
        "sha256": sha256_file(target_path),
        "research_data_missing": True,
    }
    build.counts["engineering_fixtures"] = summary


# ---------------------------------------------------------------------------

def source_hashes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("k400", "moviegen", "flickr30k", "visual_genome", "snli"):
        manifest = S.RAW_ROOT / name / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            files = {f.get("path") or f.get("file"): f.get("sha256") for f in data.get("files", [])}
            if not files and data.get("sha256"):
                files = {"<archive>": data["sha256"]}
            out[name] = {
                "manifest_sha256": sha256_file(manifest),
                "revision": data.get("revision") or data.get("repository"),
                "files": files,
            }
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="all", choices=["all", "clean", "k400", "visual_genome", "vg_objects", "vg_regions", "flickr", "moviegen", "snli", "fixtures"])
    parser.add_argument("--build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--limit", type=int, default=400, help="per-source small-batch cap")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--snli-limit", type=int, default=2000)
    parser.add_argument("--root", default=str(ROOT / "output/paper-data"), help="new construction output; raw inputs use VBENCH_TRAINING_RAW")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build = Build(args.build_id, limit=args.limit, seed=args.seed, root=Path(args.root))
    vocab = S.load_k400()
    build._action_resolver = _Resolver(vocab)
    build.sources_used = source_hashes()

    if args.source == "all":
        selected = ["k400", "visual_genome", "vg_objects", "flickr", "moviegen", "snli", "fixtures"]
    elif args.source == "clean":
        # Flickr-free rebuild used for the Objects adapter (decision B4).
        selected = ["k400", "visual_genome", "vg_objects", "vg_regions", "fixtures"]
    else:
        selected = [args.source]
    build.counts["k400_vocabulary"] = {"labels": len(vocab), "aliases": len(vocab.aliases)}
    if "k400" in selected:
        prepare_k400(build, vocab)
    if "visual_genome" in selected:
        prepare_visual_genome(build, args.limit, args.seed)
    if "vg_objects" in selected:
        prepare_vg_objects(build, args.limit, args.seed)
    if "vg_regions" in selected:
        prepare_vg_regions(build, args.limit, args.seed)
    if "flickr" in selected:
        prepare_flickr(build, args.limit)
    if "moviegen" in selected:
        prepare_moviegen(build)
    if "snli" in selected:
        prepare_snli(build, args.snli_limit)
    if "fixtures" in selected:
        prepare_engineering_fixtures(build)

    counts = build.write(parameters={"source": args.source, "limit": args.limit, "seed": args.seed, "build_id": args.build_id})
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
