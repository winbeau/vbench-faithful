#!/usr/bin/env python3
"""Annotation-only audit. Network is opt-in; pyarrow is optional for SNLI.

Downloads are hard bounded to 150,000,000 bytes per source including metadata.
Never executes upstream dataset scripts, fetches media, or generates labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

VG_REV = "65bc9e7e7353fff750326c9523e384701934e530"
SNLI_REV = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
LIMIT = 150_000_000
SOURCES = {
    "visual_genome": {
        "visual_genome.py": f"https://huggingface.co/datasets/ranjaykrishna/visual_genome/raw/{VG_REV}/visual_genome.py",
        "relationships.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    },
    "snli": {
        "README.md": f"https://huggingface.co/datasets/stanfordnlp/snli/raw/{SNLI_REV}/README.md",
        **{f"{s}-00000-of-00001.parquet": f"https://huggingface.co/datasets/stanfordnlp/snli/resolve/{SNLI_REV}/plain_text/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")},
    },
}
# Deliberately conservative exact whitelist, not substring matching.
PREDICATES = {
    "left": {"left of", "to the left of", "on the left of"},
    "right": {"right of", "to the right of", "on the right of"},
    "above": {"above"},
    "below": {"below", "under", "underneath", "beneath"},
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname or ""
        if host == "visualgenome.org" or host.endswith(".visualgenome.org"):
            raise ValueError("Refusing unrelated visualgenome.org domain")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(source, root):
    directory = root / source
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_files = {x["file"]: x for x in previous.get("files", [])}
    total = 0
    records = []
    opener = urllib.request.build_opener(SafeRedirect())
    for name, url in SOURCES[source].items():
        path = directory / name
        if path.exists():
            old = previous_files.get(name)
            if not old or old["url"] != url or old["sha256"] != sha256(path):
                raise ValueError(f"Unverified existing file: {path}")
            record = old
        else:
            temporary = path.with_suffix(path.suffix + ".partial")
            try:
                with opener.open(url, timeout=90) as response, temporary.open("wb") as out:
                    declared = response.headers.get("Content-Length")
                    if declared and total + int(declared) > LIMIT:
                        raise ValueError("Source exceeds compressed byte budget")
                    size = 0
                    while True:
                        chunk = response.read(min(1024 * 1024, LIMIT - total - size + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if total + size > LIMIT:
                            raise ValueError("Source exceeds compressed byte budget")
                        out.write(chunk)
                    if declared and size != int(declared):
                        raise ValueError("Incomplete response")
                    # Do not persist temporary signed CDN redirect URLs.
                    headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length", "X-Repo-Commit")}
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            record = {"file": name, "url": url, "bytes": path.stat().st_size,
                      "sha256": sha256(path), "retrieved_utc": datetime.now(timezone.utc).isoformat(), "headers": headers}
        total += record["bytes"]
        if total > LIMIT:
            raise ValueError("Cached source exceeds compressed byte budget")
        records.append(record)
        manifest_path.write_text(json.dumps({"source": source, "budget_bytes": LIMIT, "files": records}, indent=2) + "\n")
    return records


def json_array(stream):
    """Incrementally decode one image record at a time from a JSON array."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    ended = False
    eof = False
    while not ended:
        if not eof:
            chunk = stream.read(1024 * 1024)
            eof = not chunk
            buffer += chunk
        if not started:
            buffer = buffer.lstrip()
            if not buffer.startswith("["):
                raise ValueError("Expected JSON array")
            buffer = buffer[1:]
            started = True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                ended = True
                break
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Truncated JSON array") from None
                break
            yield value
            buffer = buffer[end:]


def entity_name(entity):
    names = entity.get("names", [])
    return str(entity.get("name") or (names[0] if names else "")).strip()


def inspect_vg(directory):
    counts = Counter()
    matched = Counter()
    lexical = Counter()
    exact_predicates = Counter()
    seen = set()
    image_ids = set()
    examples = []
    with zipfile.ZipFile(directory / "relationships.json.zip") as archive:
        member = archive.getinfo("relationships.json")
        if member.file_size > 3_000_000_000:
            raise ValueError("Unexpected uncompressed size")
        with io.TextIOWrapper(archive.open(member), encoding="utf-8") as stream:
            for image in json_array(stream):
                counts["image_records"] += 1
                image_id = image.get("image_id", image.get("id"))
                image_ids.add(image_id)
                for rel in image["relationships"]:
                    counts["raw_relationships"] += 1
                    predicate = " ".join(rel["predicate"].lower().split())
                    exact_predicates[predicate] += 1
                    for word in ("left", "right", "above", "below"):
                        if re.search(r"\b" + word + r"\b", predicate):
                            lexical[word] += 1
                    direction = next((k for k, values in PREDICATES.items() if predicate in values), None)
                    if direction is None:
                        continue
                    matched[direction] += 1
                    s, o = rel["subject"], rel["object"]
                    sn, on = entity_name(s), entity_name(o)
                    sid, oid = s.get("object_id"), o.get("object_id")
                    if not sn or not on or sid is None or oid is None or sid == oid:
                        counts["rejected_missing_name_id_or_self"] += 1
                        continue
                    key = (image_id, sid, direction, oid)
                    if key in seen:
                        counts["duplicate_supported_triples"] += 1
                        continue
                    seen.add(key)
                    counts["unique_structural_candidates"] += 1
                    counts[f"unique_{direction}"] += 1
                    if len(examples) < 6:
                        examples.append({"image_id": image_id, "relationship": rel})
    return {"counts": dict(counts), "unique_image_ids": len(image_ids),
            "uncompressed_bytes": member.file_size, "whitelist": {k: sorted(v) for k, v in PREDICATES.items()},
            "matched_predicate_counts": dict(matched), "lexical_word_counts_not_labels": dict(lexical),
            "whitelist_raw_predicates": {k: exact_predicates[k] for values in PREDICATES.values() for k in sorted(values)},
            "generic_on_excluded": exact_predicates["on"], "examples": examples,
            "approved_natural_prompt_label_pairs": 0}


def inspect_snli(directory):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Optional pyarrow unavailable; use the existing rtx4090 project .venv for --source snli; previous report preserved") from exc
    results = {}
    split_keys = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}-00000-of-00001.parquet"
        parquet = pq.ParquetFile(path)
        labels = Counter()
        seen = set()
        pair_labels = {}
        counts = Counter()
        examples = []
        for batch in parquet.iter_batches(batch_size=8192):
            for row in batch.to_pylist():
                counts["raw_rows"] += 1
                label = row["label"]
                labels[str(label)] += 1
                premise, hypothesis = row["premise"].strip(), row["hypothesis"].strip()
                if label not in (0, 1, 2):
                    counts["rejected_unlabeled"] += 1
                    continue
                if not premise or not hypothesis:
                    counts["rejected_empty"] += 1
                    continue
                counts["valid_labeled_rows"] += 1
                key = (premise, hypothesis, label)
                if key in seen:
                    counts["duplicate_exact_labeled_rows"] += 1
                    continue
                seen.add(key)
                pair_labels.setdefault((premise, hypothesis), set()).add(label)
                if len(examples) < 3:
                    examples.append(row)
        split_keys[split] = set(pair_labels)
        counts["unique_labeled_rows"] = len(seen)
        counts["conflicting_text_pairs"] = sum(len(v) > 1 for v in pair_labels.values())
        counts["unambiguous_unique_text_pairs"] = sum(len(v) == 1 for v in pair_labels.values())
        results[split] = {"counts": dict(counts), "labels": dict(labels), "schema": str(parquet.schema_arrow),
                          "examples": examples, "file_sha256": sha256(path)}
    results["cross_split_exact_pair_overlap"] = {f"{a}/{b}": len(split_keys[a] & split_keys[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    results["approved_scene_pipeline_pairs"] = 0
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", choices=("visual_genome", "snli", "all"), default="all")
    parser.add_argument("--download", action="store_true", help="Opt into bounded annotation downloads")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    for source in SOURCES if args.source == "all" else [args.source]:
        directory = args.root / source
        if args.download:
            download(source, args.root)
        if not args.download_only:
            manifest = json.loads((directory / "manifest.json").read_text())
            records = {record["file"]: record for record in manifest["files"]}
            required = ["relationships.json.zip"] if source == "visual_genome" else [f"{split}-00000-of-00001.parquet" for split in ("train", "validation", "test")]
            for name in required:
                record = records[name]
                path = directory / name
                if record["url"] != SOURCES[source][name] or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise ValueError(f"Provenance verification failed: {path}")
            result = inspect_vg(directory) if source == "visual_genome" else inspect_snli(directory)
            result["inspected_utc"] = datetime.now(timezone.utc).isoformat()
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "inspection.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"source": source, "report": str(directory / "inspection.json"), "result": result}, indent=2))


if __name__ == "__main__":
    main()
