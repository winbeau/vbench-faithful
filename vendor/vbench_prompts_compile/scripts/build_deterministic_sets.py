#!/usr/bin/env python3
"""Deterministic (metamorphic) experiment sets for the four dimensions.

Transformations are grouped by the relation they test:

* ``scene-synonym``   -- invariance: replace a scene word with a same-scene synonym
  (ocean -> sea). The official pipeline matches prompt keywords inside the Tag2Text
  caption, so this is exactly where it is expected to be brittle.
* ``action-swap``     -- text-side sensitivity plus the ``other`` protocol: replace the
  prompt's action with a different Kinetics-400 class, or with an out-of-vocabulary
  action that must be reported as ``other`` instead of being forced into a class.
* ``spatial-mirror``  -- sensitivity: mirror the video horizontally/vertically, so the
  left/right (above/below) verdict must flip.
* ``objects-occlusion`` -- sensitivity: progressively occlude one target object.

Prompt-side sets are built here; video-side transforms are emitted as an ffmpeg/occlusion
plan that runs on the scoring host (``--emit-video-plan``).

Usage::

    uv run --no-sync python scripts/build_deterministic_sets.py --captions <scene captions.jsonl>
    uv run --no-sync python scripts/build_deterministic_sets.py --captions <...> --emit-video-plan
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import sources as S  # noqa: E402

# Curated same-scene synonym table. Every pair was reviewed so that the required
# scene does not change (this is the whole point of the invariance test).
SCENE_SYNONYMS: dict[str, str] = {
    "ocean": "sea",
    "sea": "ocean",
    "beach": "seaside",
    "seaside": "beach",
    "forest": "woods",
    "woods": "forest",
    "cave": "cavern",
    "cavern": "cave",
    "garden": "yard",
    "yard": "garden",
    "street": "road",
    "road": "street",
    "shop": "store",
    "store": "shop",
    "bathroom": "washroom",
    "washroom": "bathroom",
    "bedroom": "bedchamber",
    "bedchamber": "bedroom",
}


def substitute_synonym(prompt: str) -> tuple[str, str] | None:
    """Replace the first synonym-table word; return ``(new_prompt, replaced_pair)``."""
    words = prompt.split()
    for index, word in enumerate(words):
        key = R.normalize_phrase(word)
        if key in SCENE_SYNONYMS:
            replacement = SCENE_SYNONYMS[key]
            words[index] = replacement
            return " ".join(words), f"{key}->{replacement}"
    return None


def build_scene_synonym(captions_path: Path, *, from_gold: bool = False) -> list[dict[str, Any]]:
    """Build synonym pairs; with ``from_gold`` every pair keeps its annotated label."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in captions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if from_gold:
            row = {
                "prompt": row["input"]["prompt"],
                "caption": row["input"]["caption"],
                "video_path": row.get("sample_id"),
                "frame_path": None,
                "generator": row.get("meta", {}).get("generator"),
                "split": row.get("meta", {}).get("split"),
                "gold_label": row.get("target"),
            }
        elif row.get("status") != "ok":
            continue
        result = substitute_synonym(row["prompt"])
        if result is None:
            continue
        synonym_prompt, pair = result
        key = (row["prompt"], str(row.get("video_path")))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "family": "scene-synonym",
                "pair": pair,
                "original_prompt": row["prompt"],
                "variant_prompt": synonym_prompt,
                "caption": row["caption"],
                "video_path": row["video_path"],
                "frame_path": row.get("frame_path"),
                "generator": row.get("generator"),
                "split": row.get("split"),
                **({"gold_label": row["gold_label"]} if row.get("gold_label") else {}),
            }
        )
    return rows


def build_action_swap(action_prompts: list[str], vocab: S.K400Vocabulary) -> list[dict[str, Any]]:
    """Replace the prompt's action class with another class, and add an OOV variant."""
    rows: list[dict[str, Any]] = []
    labels = list(vocab.labels)
    for index, prompt in enumerate(action_prompts):
        original = None
        for label in labels:
            if label in prompt.lower():
                original = label
                break
        if original is None:
            continue
        other = labels[(labels.index(original) + 97) % len(labels)]
        if other == original:
            continue
        rows.append(
            {
                "family": "action-swap",
                "kind": "class_swap",
                "original_prompt": prompt,
                "variant_prompt": prompt.lower().replace(original, other),
                "original_action": original,
                "variant_action": other,
            }
        )
        oov = "assembling furniture"
        if oov in prompt.lower():
            continue
        rows.append(
            {
                "family": "action-swap",
                "kind": "out_of_vocabulary",
                "original_prompt": prompt,
                "variant_prompt": prompt.lower().replace(original, oov),
                "original_action": original,
                "variant_action": "other",
            }
        )
    return rows


def build_video_plan(captions_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Mirror- and occlusion-side plans (executed on the scoring host)."""
    scene_rows = []
    for line in captions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            scene_rows.append({"video_path": row["video_path"], "prompt": row["prompt"], "generator": row.get("generator")})
    return {
        "spatial-mirror": [
            {"video_path": row["video_path"], "prompt": row["prompt"], "transforms": ["hflip", "vflip"]} for row in scene_rows
        ],
        "objects-occlusion": [
            {"video_path": row["video_path"], "prompt": row["prompt"], "levels": [0.0, 0.25, 0.5, 0.75, 1.0]} for row in scene_rows
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", required=True, help="caption_frames.py output (real frames + official Tag2Text captions)")
    parser.add_argument("--action-prompts", default=None, help="JSONL with action prompts (defaults to the K400 template corpus)")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "deterministic"))
    parser.add_argument("--emit-video-plan", action="store_true")
    parser.add_argument("--synonyms-from-gold", action="store_true", help="treat --captions as a scene gold file so every pair has an annotated label")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    captions_path = Path(args.captions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_rows = build_scene_synonym(captions_path, from_gold=args.synonyms_from_gold)
    R.write_jsonl(out_dir / "scene-synonym.jsonl", scene_rows)
    print(json.dumps({"scene_synonym_pairs": len(scene_rows), "pairs": dict(Counter(row["pair"] for row in scene_rows))}, ensure_ascii=False))

    action_prompts: list[str] = []
    action_path = Path(args.action_prompts) if args.action_prompts else ROOT / "data" / "processed" / "clean-0002" / "candidates" / "action.jsonl"
    if action_path.exists():
        for line in action_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                action_prompts.append(json.loads(line)["input"]["prompt"])
    action_rows = build_action_swap(action_prompts, S.load_k400())
    R.write_jsonl(out_dir / "action-swap.jsonl", action_rows)
    print(json.dumps({"action_swap_pairs": len(action_rows), "kinds": dict(Counter(row["kind"] for row in action_rows))}, ensure_ascii=False))

    if args.emit_video_plan:
        plan = build_video_plan(captions_path)
        (out_dir / "video-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"video_plan_families": {k: len(v) for k, v in plan.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
