#!/usr/bin/env python3
"""Score the deterministic (metamorphic) sets: Official criterion vs Repair.

Currently implements the two prompt-side families:

``scene-synonym``
    Official criterion (VBench scene): every word of the scene phrase must appear as a
    substring of the Tag2Text caption. A same-scene synonym therefore fails even though
    a human would call the scene supported -- this is the invariance the repair targets.
    Two official readings are reported: the whole prompt words, and only the scene noun.

``action-swap``
    Deterministic text-side interventions. ``class_swap`` must move the verdict
    (sensitivity); ``out_of_vocabulary`` must become ``other`` and therefore must not be
    scored as the original class (protocol check).

Repair labels are produced by the project models (``--scene-model`` / ``--adapters``);
without them the script reports the official half only.

Usage::

    uv run --no-sync python scripts/score_deterministic.py --family scene-synonym
    uv run --no-sync python scripts/score_deterministic.py --family scene-synonym \
        --base-model /home/winbeau/models/Qwen3-0.6B --scene-model runs/formal/v4-8b/scene --limit 40
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import records as R  # noqa: E402

SCENE_STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and"}


def official_scene_pass(scene_key: str, caption: str) -> bool:
    """VBench scene rule: every word of the key appears in the caption."""
    words = [word for word in scene_key.split(" ") if word]
    return bool(words) and all(word in caption for word in words)


def scene_noun(prompt: str) -> str:
    words = [word for word in prompt.split() if word.lower() not in SCENE_STOPWORDS]
    return words[-1] if words else prompt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="scene-synonym")
    parser.add_argument("--data", default=None, help="path to the deterministic set JSONL")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--scene-model", default=None)
    parser.add_argument("--adapters", action="append", default=[], help="task=path (for the action family)")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args(argv)


def score_scene_synonym(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired: list[dict[str, Any]] = []
    for row in rows:
        caption = row["caption"]
        original, variant = row["original_prompt"], row["variant_prompt"]
        paired.append(
            {
                **row,
                "official_original": official_scene_pass(original, caption),
                "official_variant": official_scene_pass(variant, caption),
                "official_noun_original": official_scene_pass(scene_noun(original), caption),
                "official_noun_variant": official_scene_pass(scene_noun(variant), caption),
                "repair_original": None,
                "repair_variant": None,
            }
        )
    return {"rows": paired}


def attach_repair(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.scene_model or not args.base_model:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vbench_prompts_compile.inference import SceneVerifier

    path = Path(args.scene_model)
    if not path.is_absolute():
        path = ROOT / path
    if (path / "adapter_config.json").exists():
        verifier = SceneVerifier(base_model_path=args.base_model, adapter_path=str(path), local_files_only=not args.allow_download)
    else:
        verifier = SceneVerifier(model_path=str(path), local_files_only=not args.allow_download)
    for row in rows:
        row["repair_original"] = verifier.predict(row["original_prompt"], row["caption"], max_new_tokens=8, stop_at_newline=True).value
        row["repair_variant"] = verifier.predict(row["variant_prompt"], row["caption"], max_new_tokens=8, stop_at_newline=True).value


def summarize_scene_synonym(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    official_orig = sum(row["official_original"] for row in rows)
    official_var = sum(row["official_variant"] for row in rows)
    false_fail = sum(1 for row in rows if row["official_original"] and not row["official_variant"])
    noun_orig = sum(row["official_noun_original"] for row in rows)
    noun_var = sum(row["official_noun_variant"] for row in rows)
    summary: dict[str, Any] = {
        "family": "scene-synonym",
        "pairs": total,
        "official_full_prompt": {"original_pass": official_orig, "variant_pass": official_var, "false_failures": false_fail},
        "official_scene_noun_only": {"original_pass": noun_orig, "variant_pass": noun_var, "false_failures": sum(1 for row in rows if row["official_noun_original"] and not row["official_noun_variant"])},
        "by_pair": {},
    }
    for pair in sorted({row["pair"] for row in rows}):
        subset = [row for row in rows if row["pair"] == pair]
        summary["by_pair"][pair] = {
            "n": len(subset),
            "official_false_failures": sum(1 for row in subset if row["official_original"] and not row["official_variant"]),
        }
    if rows and rows[0]["repair_original"] is not None:
        stable = sum(1 for row in rows if row["repair_original"] == row["repair_variant"])
        supported = sum(1 for row in rows if row["repair_original"] == "supported" and row["repair_variant"] == "supported")
        summary["repair"] = {
            "label_stability": stable / total,
            "supported_both": supported,
            "supported_original": sum(1 for row in rows if row["repair_original"] == "supported"),
            "supported_variant": sum(1 for row in rows if row["repair_variant"] == "supported"),
            "label_counts": dict(Counter(row["repair_original"] for row in rows)),
        }
    return summary


def score_action_swap(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Repair-side check of the action interventions (official side needs UMT on the GPU host).

    * class_swap: the parsed action set must change to the new class (sensitivity);
    * out_of_vocabulary: the parsed action must be exactly ["other"] (protocol), never
      the original class and never a guessed K400 class.
    """
    counters: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    if args.adapters:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from vbench_prompts_compile.inference import AdapterRouter
        from vbench_prompts_compile import training as T

        adapters = {}
        for item in args.adapters:
            task, _, path = item.partition("=")
            adapters[task] = path
        router = AdapterRouter(base_model_path=args.base_model, adapters=adapters, local_files_only=not args.allow_download)
    else:
        router = None
    for row in rows:
        entry = dict(row)
        if router is not None:
            original = router.predict("action", T.user_text({"task": "action", "input": {"prompt": row["original_prompt"]}}), max_new_tokens=64)
            variant = router.predict("action", T.user_text({"task": "action", "input": {"prompt": row["variant_prompt"]}}), max_new_tokens=64)
            entry["repair_original"] = (original.value or {}).get("actions")
            entry["repair_variant"] = (variant.value or {}).get("actions")
            if row["kind"] == "class_swap":
                moved = bool(entry["repair_variant"]) and row["variant_action"] in entry["repair_variant"]
                counters["class_swap_moved" if moved else "class_swap_not_moved"] += 1
            else:
                ok = entry["repair_variant"] == ["other"]
                counters["oov_reported_other" if ok else "oov_not_other"] += 1
                if entry["repair_original"] == ["other"]:
                    counters["original_already_other"] += 1
        results.append(entry)
    return {"rows": results, "counters": dict(counters)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data) if args.data else ROOT / "data" / "deterministic" / f"{args.family}.jsonl"
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if args.family == "scene-synonym":
        payload = score_scene_synonym(rows)
        attach_repair(payload["rows"], args)
        summary = summarize_scene_synonym(payload["rows"])
    elif args.family == "action-swap":
        payload = score_action_swap(rows, args)
        summary = {"family": "action-swap", "pairs": len(rows), "repair": payload["counters"]}
    else:
        print(f"family {args.family!r} is not implemented yet", file=sys.stderr)
        return 2
    out = Path(args.output) if args.output else ROOT / "output" / "deterministic" / f"{args.family}-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": payload["rows"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
