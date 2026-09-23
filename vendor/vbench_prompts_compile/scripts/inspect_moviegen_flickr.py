#!/usr/bin/env python3
"""Fetch pinned annotation-only inputs (<=100 MB/source); audit, never split/train.

Standard library only. Default is offline statistics; --fetch permits downloads.
Raw inputs, manifest and statistics stay under ignored data/raw/{source}.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CAP = 100_000_000
SOURCES = {
    "moviegen": ("facebookresearch/MovieGenBench", "bab7753fce1108def167a1efb4c1ae1d69b8f03a", {
        "README.md": 3801, "LICENSE": 19334,
        "benchmark/MovieGenVideoBench.txt": 106686,
        "benchmark/MovieGenVideoBenchWithTag.csv": 132624,
        "benchmark/MovieGenAudioBenchSfx.jsonl": 121743,
        "benchmark/MovieGenAudioBenchSfxMusic.jsonl": 216325,
    }),
    "flickr30k": ("BryanPlummer/flickr30k_entities", "68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b", {
        "README.md": 4493, "UNRELATED_CAPTIONS": 303,
        "annotations.zip": 29284070, "train.txt": 320344,
        "test.txt": 10733, "val.txt": 10758,
    }),
}


def collect(name: str, fetch: bool) -> Path:
    repo, revision, files = SOURCES[name]
    assert sum(files.values()) <= CAP
    base = ROOT / "data" / "raw" / name
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    prior = {x["path"]: x for x in previous["files"]} if previous else {}
    manifest = {"repository": repo, "revision": revision, "byte_cap": CAP,
                "training_status": ("training_candidate_unlabeled_pending_compliance_and_leakage_protocol"
                                    if name == "moviegen" else "candidate_pending_annotation_license_review"),
                "files": []}
    transferred = 0
    for relative, expected in files.items():
        path = base / relative
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{relative}"
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(f"{path}; use --fetch explicitly")
            request = urllib.request.Request(url, headers={"User-Agent": "annotation-audit/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                # Expected Git blob size is a tighter bound than the source cap.
                data = response.read(min(expected + 1, CAP - transferred))
            transferred += len(data)
            if len(data) != expected:
                raise ValueError(f"Unexpected size for {url}: {len(data)} != {expected}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != expected:
            raise ValueError(f"Cached size mismatch: {path}")
        if relative in prior and digest != prior[relative]["sha256"]:
            raise ValueError(f"Cached SHA256 mismatch: {path}")
        manifest["files"].append({"path": relative, "url": url, "bytes": len(data), "sha256": digest})
    manifest["total_bytes"] = sum(x["bytes"] for x in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return base


def moviegen_stats(base: Path) -> dict:
    benchmark = base / "benchmark"
    prompts = (benchmark / "MovieGenVideoBench.txt").read_text().splitlines()
    with (benchmark / "MovieGenVideoBenchWithTag.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames
    audio = {}
    audio_video_sets = []
    for name in ("MovieGenAudioBenchSfx.jsonl", "MovieGenAudioBenchSfxMusic.jsonl"):
        entries = [json.loads(line) for line in (benchmark / name).read_text().splitlines() if line.strip()]
        audio[name] = {"records": len(entries), "keys": sorted(entries[0]), "example": entries[0]}
        audio_video_sets.append({row["video_prompt"] for row in entries})
    cross = {
        "audio_variants_video_sets_identical": audio_video_sets[0] == audio_video_sets[1],
        "video_audio_exact_overlap": len(set(prompts) & set.union(*audio_video_sets)),
        "union_unique_video_prompts": len(set(prompts).union(*audio_video_sets)),
    }
    return {"cross_file_exact_dedup": cross, "video_lines": len(prompts), "video_nonempty": sum(bool(p.strip()) for p in prompts),
            "video_unique_exact": len(set(prompts)), "tag_rows": len(rows), "tag_columns": columns,
            "tag_example": rows[0], "prompt_example": prompts[0], "audio": audio}


def flickr_stats(base: Path) -> dict:
    splits = {s: (base / f"{s}.txt").read_text().splitlines() for s in ("train", "val", "test")}
    out = {"split_counts": {s: len(ids) for s, ids in splits.items()},
           "split_unique": {s: len(set(ids)) for s, ids in splits.items()},
           "split_pairwise_overlap": {f"{a}/{b}": len(set(splits[a]) & set(splits[b]))
                                      for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}}
    types = Counter()
    captions = mentions = boxes = scene = nobox = zero = 0
    unique_captions, chains = set(), set()
    pattern = re.compile(r"\[/EN#([^/\s]+)/([^\s]+) ([^\]]+)\]")
    with zipfile.ZipFile(base / "annotations.zip") as archive:
        sentences = sorted(n for n in archive.namelist() if n.startswith("Sentences/") and n.endswith(".txt"))
        annotations = sorted(n for n in archive.namelist() if n.startswith("Annotations/") and n.endswith(".xml"))
        out["archive_members"] = len(archive.infolist())
        out["sentence_files"] = len(sentences)
        out["annotation_files"] = len(annotations)
        out["annotation_text_uncompressed_bytes"] = sum(archive.getinfo(n).file_size for n in sentences + annotations)
        # Read only known text entries in memory; never extract media or arbitrary paths.
        for name in sentences:
            if archive.getinfo(name).file_size > 1_000_000:
                raise ValueError(f"Oversized sentence member: {name}")
            lines = archive.read(name).decode("utf-8").splitlines()
            for line in lines:
                captions += 1
                unique_captions.add(pattern.sub(lambda m: m[3], line))
                matches = list(pattern.finditer(line))
                mentions += len(matches)
                for match in matches:
                    types.update(match[2].split("/"))
                    if match[1] == "0":
                        zero += 1
                    else:
                        chains.add((Path(name).stem, match[1]))
        for name in annotations:
            if archive.getinfo(name).file_size > 1_000_000:
                raise ValueError(f"Oversized XML member: {name}")
            tree = ET.fromstring(archive.read(name))
            for obj in tree.findall("object"):
                boxes += len(obj.findall("bndbox"))
                scene += obj.findtext("scene") == "1"
                nobox += obj.findtext("nobndbox") == "1"
        ids = {Path(n).stem for n in sentences}
        out["sentence_annotation_id_match"] = ids == {Path(n).stem for n in annotations}
        out["split_union_matches_sentence_ids"] = ids == set().union(*(set(v) for v in splits.values()))
        out["sentence_example_path"] = sentences[0]
        out["sentence_example"] = archive.read(sentences[0]).decode().splitlines()[0]
        out["xml_example_path"] = annotations[0]
        out["xml_example"] = archive.read(annotations[0]).decode()
    out.update(captions=captions, unique_plain_captions_exact=len(unique_captions), phrase_mentions=mentions,
               nonzero_image_chain_pairs=len(chains), zero_id_mentions=zero, phrase_type_assignments=dict(types),
               bounding_box_elements=boxes, scene_object_entries=scene, nobndbox_object_entries=nobox,
               unrelated_caption_lines=len((base / "UNRELATED_CAPTIONS").read_text().splitlines()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Allow pinned annotation-only downloads")
    parser.add_argument("--source", choices=["all", *SOURCES], default="all")
    args = parser.parse_args()
    for name in SOURCES if args.source == "all" else [args.source]:
        base = collect(name, args.fetch)
        stats = moviegen_stats(base) if name == "moviegen" else flickr_stats(base)
        (base / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
        print(name, json.dumps({k: v for k, v in stats.items() if "example" not in k}, ensure_ascii=False))


if __name__ == "__main__":
    main()
