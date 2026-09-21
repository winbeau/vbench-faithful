"""Resolve background videos via shared scene MEDIA identities, not its labels.

The frozen E0 background paths contain articles, wrong suffixes, and some
generator directory mismatches. Keep its UIDs and splits for label linkage,
but resolve each (prompt, seed, generator) using the verified Scene media map.
Scene human_anno is deliberately never read by the mapping function.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .common import ROOT, sha256_file
from .subject_artifacts import write_json, write_jsonl


def resolve_media(rows, scene_rows):
    mapping = {}
    for row in scene_rows:
        for generator, relative in row["videos"].items():
            seed = int(Path(relative).stem.rsplit("-", 1)[1])
            key = (row["prompt_en"], seed, generator)
            if key in mapping:
                raise ValueError("ambiguous scene media identity")
            mapping[key] = relative
    result = []
    for row in rows:
        seed = int(Path(row["relative_video_path"]).stem.rsplit("-", 1)[1])
        key = (row["prompt_id"], seed, row["generator"])
        relative = mapping[key]
        actual_generator = Path(relative).parts[0]
        if actual_generator != {"videocraft": "videocrafter"}.get(row["generator"], row["generator"]):
            raise ValueError("scene media map has a conflicting generator identity")
        result.append({**row, "manifest_video_path": row["relative_video_path"],
                       "relative_video_path": relative, "seed": seed})
    if len({r["relative_video_path"] for r in result}) != len(result):
        raise ValueError("two official identities resolve to one video")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-media-json", type=Path, required=True)
    args = parser.parse_args()
    config = ROOT / "configs/background-repair"
    source = ROOT / "data/processed/e0_scoring_manifest.csv"
    rows = [r for r in csv.DictReader(source.open()) if r["dimension"] == "background_consistency"]
    resolved = resolve_media(rows, json.loads(args.scene_media_json.read_text()))
    manifest = config / "natural1720_manifest_v2.jsonl"
    protocol_path = config / "development_protocol_v2.json"
    if manifest.exists() or protocol_path.exists():
        raise FileExistsError("refusing to overwrite a frozen input resolution")
    assert len(resolved) == 1720 and Counter(r["split"] for r in resolved) == {"dev": 680, "test": 1040}
    write_jsonl(manifest, resolved)
    protocol = json.loads((config / "development_protocol.json").read_text())
    protocol.update(revision=2, manifest_file=manifest.name, natural_manifest_sha256=sha256_file(manifest),
        media_resolution={"source": str(args.scene_media_json), "sha256": sha256_file(args.scene_media_json),
            "key": ["prompt_en", "seed", "generator"], "scene_labels_used": False,
            "corrected_paths": sum(a["relative_video_path"] != b["relative_video_path"] for a, b in zip(rows, resolved))},
        supersedes="v1 run exposed 665/680 missing paths from the frozen E0 mapping; resolve shared media only. Labels, splits, methods and thresholds are unchanged. The 15 completed dev rows and every failed row remain archived.")
    write_json(protocol_path, protocol)
    print(json.dumps({"videos": len(resolved), "resolution": protocol["media_resolution"], "protocol_sha256": sha256_file(protocol_path)}))


if __name__ == "__main__":
    main()
