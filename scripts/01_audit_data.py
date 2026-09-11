#!/usr/bin/env python3
"""Audit VBench human-preference JSON files without touching videos or models."""
import argparse, csv, json
from pathlib import Path

DIMENSIONS = ["subject_consistency", "background_consistency", "temporal_flickering",
              "motion_smoothness", "dynamic_degree", "aesthetic_quality",
              "imaging_quality", "object_class", "multiple_objects", "human_action",
              "color", "spatial_relationship", "scene", "temporal_style",
              "appearance_style", "overall_consistency"]

def audit(path, dimension):
    data = json.loads(path.read_text())
    if not isinstance(data, list): raise ValueError("top-level JSON must be a list")
    prompts, groups, videos, model_sets, pairs, labels, ties, bad = set(), set(), set(), set(), 0, 0, 0, 0
    four_model_records = 0
    missing_videos = missing_anno = 0
    for row in data:
        prompts.add(row.get("prompt_en"))
        groups.add(row.get("group_id", row.get("style_en", row.get("category_en", ""))))
        v = row.get("videos", {}) or {}
        videos.update(v.keys())
        model_sets.add(tuple(sorted(v)))
        if len(v) != 4: missing_videos += 1
        else: four_model_records += 1
        anno = row.get("human_anno", {})
        if not anno: missing_anno += 1
        for a, inner in anno.items():
            for b, value in inner.items():
                if a >= b: continue
                pairs += 1; labels += 1
                if value == 0.5: ties += 1
                if value not in (0, 0.5, 1): bad += 1
    return {"dimension": dimension, "instances": len(data), "prompts": len(prompts),
            "groups": len(groups), "four_model_groups": four_model_records,
            "video_model_ids": len(videos), "pairs": pairs, "released_labels": labels,
            "ties": ties, "invalid_labels": bad, "missing_video_records": missing_videos,
            "missing_annotation_records": missing_anno,
            "model_video_ids": "|".join(sorted(videos)),
            "status": "ok" if not bad else "invalid_labels"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--dimension", required=True); ap.add_argument("--out", type=Path, required=True)
    a = audit(ap.parse_args().input, ap.parse_args().dimension) if False else None
    ns = ap.parse_args(); a = audit(ns.input, ns.dimension); ns.out.parent.mkdir(parents=True, exist_ok=True)
    with ns.out.open("w", newline="") as f: csv.DictWriter(f, fieldnames=a).writeheader(); csv.DictWriter(f, fieldnames=a).writerow(a)
    print(json.dumps(a, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
