#!/usr/bin/env python3
"""Origin vs Repair on the original dataset: same visual cache, two text interfaces (Table B).

The visual evidence is computed once per video (``scripts/cache_backend_outputs.py``) and
never recomputed here, so a score difference can only come from how a prompt is turned
into targets:

* **Origin interface** -- deterministic prompt parsing in the shape the official
  dimension uses (spatial: ``a X on the left of a Y``; objects: ``X and Y``; scene: the
  prompt itself is the scene key);
* **Repair interface** -- the fine-tuned adapters' parsed targets
  (``scripts/batch_predict.py``), scored over the *same* cached detections/captions.

Official rule parity, taken from the locked VBench checkout:

* spatial: GRiT boxes + ``get_position_score`` (max over matching detection pairs, mean
  over 16 frames);
* objects: GRiT labels, a frame counts only if **every** required object is detected;
* scene: Tag2Text caption must contain every word of the scene key; a Repair
  ``insufficient`` answer is reported as abstention, not as a success.

Runs on the scoring host (needs the VBench checkout for ``get_position_score``); no GPU.

Usage::

    python scripts/score_origin_vs_repair.py --dimension spatial \
        --cache data/backend-cache/spatial-200.jsonl --repair data/repair/spatial.jsonl \
        --out data/backend-cache/origin-vs-repair-spatial.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Sequence

RELATION_TO_OFFICIAL = {
    "left": "on the left of",
    "right": "on the right of",
    "above": "on the top of",
    "below": "on the bottom of",
}
SPATIAL_PATTERNS = [
    ("on the left of", re.compile(r"^(?:a|an|the)\s+(?P<a>.+?)\s+on the left of\s+(?:a|an|the)\s+(?P<b>.+?)(?:,|$)", re.I)),
    ("on the right of", re.compile(r"^(?:a|an|the)\s+(?P<a>.+?)\s+on the right of\s+(?:a|an|the)\s+(?P<b>.+?)(?:,|$)", re.I)),
    ("on the top of", re.compile(r"^(?:a|an|the)\s+(?P<a>.+?)\s+on the top of\s+(?:a|an|the)\s+(?P<b>.+?)(?:,|$)", re.I)),
    ("on the bottom of", re.compile(r"^(?:a|an|the)\s+(?P<a>.+?)\s+on the bottom of\s+(?:a|an|the)\s+(?P<b>.+?)(?:,|$)", re.I)),
]
OBJECTS_PATTERN = re.compile(r"^(?P<a>.+?)\s+and\s+(?P<b>.+?)(?:,|$)", re.I)
ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)


def strip_article(name: str) -> str:
    """The official auxiliary_info stores bare object names ("car", not "a car")."""
    return ARTICLE.sub("", name.strip()).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rule_target(dimension: str, prompt: str) -> Any | None:
    prompt = prompt.strip()
    if dimension == "spatial":
        for relation, pattern in SPATIAL_PATTERNS:
            match = pattern.match(prompt)
            if match:
                return {"object_a": match.group("a").strip(), "object_b": match.group("b").strip(), "relationship": relation}
        return None
    if dimension == "objects":
        match = OBJECTS_PATTERN.match(prompt)
        if match:
            return [strip_article(match.group("a")), strip_article(match.group("b"))]
        return None
    if dimension == "scene":
        return prompt
    return None


def spatial_frame_score(relation: str, boxes_a: list[list[float]], boxes_b: list[list[float]], position_score) -> float:
    best = 0.0
    for box_a in boxes_a:
        for box_b in boxes_b:
            best = max(best, float(position_score(relation, box_a, box_b)))
    return best


def score_spatial(entries: list[dict[str, Any]], repair_by_prompt: dict[str, Any], position_score) -> dict[str, Any]:
    origin_frames: list[float] = []
    repair_frames: list[float] = []
    repair_missing = 0
    per_prompt: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"origin": [], "repair": []})
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        prompt = entry["prompt"]
        detections = entry["frame_detections"]
        origin_target = rule_target("spatial", prompt)
        repair_value = repair_by_prompt.get(prompt) or {}
        repair_triples = repair_value.get("relationships") if isinstance(repair_value, dict) else None
        for frame in detections:
            labels = [str(item["label"]).lower() for item in frame]
            boxes = [item["box"] for item in frame]
            if origin_target:
                boxes_a = [box for label, box in zip(labels, boxes) if label == origin_target["object_a"].lower()]
                boxes_b = [box for label, box in zip(labels, boxes) if label == origin_target["object_b"].lower()]
                frame_score = spatial_frame_score(origin_target["relationship"], boxes_a, boxes_b, position_score) if boxes_a and boxes_b else 0.0
                origin_frames.append(frame_score)
                per_prompt[prompt]["origin"].append(frame_score)
            # Every frame of every cached video stays in the Repair denominator: a missing
            # or unmappable parse is a failure, not a dropped sample.
            scores = []
            for triple in repair_triples or []:
                relation = RELATION_TO_OFFICIAL.get(str(triple.get("relation", "")).lower())
                if relation is None:
                    continue
                boxes_a = [box for label, box in zip(labels, boxes) if label == str(triple["subject"]).lower()]
                boxes_b = [box for label, box in zip(labels, boxes) if label == str(triple["object"]).lower()]
                scores.append(spatial_frame_score(relation, boxes_a, boxes_b, position_score) if boxes_a and boxes_b else 0.0)
            frame_score = (sum(scores) / len(scores)) if scores else 0.0
            if not scores:
                repair_missing += 1
            repair_frames.append(frame_score)
            per_prompt[prompt]["repair"].append(frame_score)
    return {
        "origin_frame_mean": statistics.mean(origin_frames) if origin_frames else None,
        "repair_frame_mean": statistics.mean(repair_frames) if repair_frames else None,
        "origin_frames": len(origin_frames),
        "repair_frames": len(repair_frames),
        "repair_frames_without_mappable_target": repair_missing,
        "per_prompt": {prompt: {k: (statistics.mean(v) if v else None) for k, v in values.items()} for prompt, values in per_prompt.items()},
    }


def score_objects(entries: list[dict[str, Any]], repair_by_prompt: dict[str, Any]) -> dict[str, Any]:
    origin_frames = repair_frames = origin_success = repair_success = 0
    per_prompt: dict[str, dict[str, float | None]] = {}
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        prompt = entry["prompt"]
        origin_target = rule_target("objects", prompt)
        repair_value = repair_by_prompt.get(prompt) or {}
        repair_entities = repair_value.get("entities") if isinstance(repair_value, dict) else None
        for frame in entry["frame_detections"]:
            labels = " ".join(str(item["label"]).lower() for item in frame)
            if origin_target:
                origin_frames += 1
                origin_success += int(all(str(name).lower() in labels for name in origin_target))
            # same rule: a missing parse is a failure, the frame stays in the denominator
            repair_frames += 1
            repair_success += int(bool(repair_entities) and all(str(name).lower() in labels for name in repair_entities))
        per_prompt[prompt] = {
            "origin": (origin_success / origin_frames) if origin_frames else None,
            "repair": (repair_success / repair_frames) if repair_frames else None,
        }
    return {
        "origin_frame_rate": (origin_success / origin_frames) if origin_frames else None,
        "repair_frame_rate": (repair_success / repair_frames) if repair_frames else None,
        "origin_frames": origin_frames,
        "repair_frames": repair_frames,
        "per_prompt": per_prompt,
    }


def score_scene(entries: list[dict[str, Any]], repair_by_prompt: dict[str, Any], repair_labels: dict[str, str]) -> dict[str, Any]:
    """``repair_labels`` is keyed by video path (the model sees prompt + caption)."""
    origin_frames = origin_success = 0
    repair_supported = repair_contradicted = repair_abstain = 0
    origin_video_rates: list[float] = []
    repair_video_scores: list[float] = []
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        prompt = entry["prompt"]
        key_words = [word for word in prompt.lower().split() if word]
        frame_hits = []
        for caption in entry.get("frame_captions", []):
            hit = int(all(word in caption.lower() for word in key_words))
            origin_frames += 1
            origin_success += hit
            frame_hits.append(hit)
        if frame_hits:
            origin_video_rates.append(sum(frame_hits) / len(frame_hits))
        label = repair_labels.get(str(entry.get("video_path"))) or repair_labels.get(prompt)
        if label == "supported":
            repair_supported += 1
            repair_video_scores.append(1.0)
        elif label == "contradicted":
            repair_contradicted += 1
            repair_video_scores.append(0.0)
        else:
            repair_abstain += 1
    decided = repair_supported + repair_contradicted
    return {
        "origin_frame_rate": (origin_success / origin_frames) if origin_frames else None,
        "origin_video_rate": (sum(origin_video_rates) / len(origin_video_rates)) if origin_video_rates else None,
        "repair_video_rate": (sum(repair_video_scores) / len(repair_video_scores)) if repair_video_scores else None,
        "origin_frames": origin_frames,
        "repair_decided_videos": decided,
        "repair_abstain_videos": repair_abstain,
        "repair_supported_videos": repair_supported,
        "repair_contradicted_videos": repair_contradicted,
        "repair_support_rate_decided": (repair_supported / decided) if decided else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dimension", required=True, choices=["spatial", "objects", "scene"])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--repair", required=True, help="batch_predict.py output (prompt -> target)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vbench-root", default="/root/wenbiao_zhao/VBench")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_jsonl(Path(args.cache))
    repair_rows = load_jsonl(Path(args.repair))
    repair_by_prompt = {row["prompt"]: row.get("target") for row in repair_rows}
    repair_labels: dict[str, str] = {}
    for row in repair_rows:
        target = row.get("target")
        if not isinstance(target, str):
            continue
        repair_labels[str(row.get("video_path") or row["prompt"])] = target
        repair_labels.setdefault(row["prompt"], target)

    sys.path.insert(0, str(Path(args.vbench_root).resolve()))
    result: dict[str, Any]
    if args.dimension == "spatial":
        from vbench.spatial_relationship import get_position_score  # noqa: PLC0415

        result = score_spatial(entries, repair_by_prompt, get_position_score)
    elif args.dimension == "objects":
        result = score_objects(entries, repair_by_prompt)
    else:
        result = score_scene(entries, repair_by_prompt, repair_labels)

    payload = {
        "dimension": args.dimension,
        "cache": args.cache,
        "repair": args.repair,
        "videos_in_cache": sum(1 for entry in entries if entry.get("status") == "ok"),
        "prompts_in_cache": len({entry["prompt"] for entry in entries if entry.get("status") == "ok"}),
        "repair_prompts": len(repair_by_prompt),
        "result": result,
        "caveats": [
            "the Origin interface is a reconstruction of the official prompt parsing, not the official auxiliary_info file",
            "the visual backend is identical on both sides; only the text interface differs",
            "insufficient answers are reported as abstention rather than counted as successes",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "result"}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
