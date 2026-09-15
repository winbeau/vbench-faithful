"""Metadata-only base selection for VBench-CF.

Plan section 3.3 forbids selecting bases by looking at Official or Audit scores.
Selection here therefore depends on exactly three things: the frozen E0 prompt
split, the prompt text, and a fixed seed.  Nothing in this module reads a metric
score or a model prediction.

Bases are drawn one per prompt so that no prompt is pseudo-replicated inside a
dimension; the generator for each prompt is chosen by the same stable hash.  The
frozen split is prompt-disjoint, so dev and test bases never share a prompt.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .common import ROOT, stable_sample, write_jsonl

MANIFEST = ROOT / "data/processed/e0_scoring_manifest.csv"
SPLIT_SOURCE = ROOT / "data/processed/pairwise_master_split.csv"

SEED = 20260914

# Plan section 4 recommended budget: (dev bases, test bases).
BUDGET: dict[str, tuple[int, int]] = {
    "dynamics_degree": (10, 30),
    "subject_consistency": (5, 20),
    "human_action": (5, 20),
    "spatial_relationship": (10, 30),
    "scene": (5, 20),
    "multiplt_object": (5, 20),
    "motion_smoothness": (5, 20),
}

GENERATORS = ("cogvideo", "lavie", "modelscope", "videocraft")

DIRECTIONAL = re.compile(r"^(?P<subject>.+?) on the (?P<relation>left|right|top|bottom) of (?P<object>.+?), (?P<view>.+)$")
HUMAN_ACTION = re.compile(r"^A person is (?P<action>.+)$")


class EligibilityError(ValueError):
    """Raised when a prompt cannot support its dimension's contract."""


def load_prompts() -> dict[tuple[str, str], str]:
    """Map (dimension, prompt_id) -> prompt text from the frozen split file."""
    mapping: dict[tuple[str, str], str] = {}
    with SPLIT_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dimension"], row["prompt_id"])
            text = row["prompt_en"]
            existing = mapping.get(key)
            if existing is not None and existing != text:
                raise EligibilityError(f"prompt_id {key} maps to two different prompts")
            mapping[key] = text
    return mapping


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_prompt(dimension: str, prompt: str) -> dict[str, Any]:
    """Extract the dimension-specific structure a transform needs.

    Raises `EligibilityError` when the prompt cannot support the contract, so
    that ineligible prompts are excluded rather than silently mistransformed.
    """
    prompt = prompt.strip()
    if not prompt:
        raise EligibilityError("empty prompt")
    if dimension == "spatial_relationship":
        # Plan section 9.2: `inside of` is unsupported by the locked Official
        # evaluator and must be excluded from comparisons, not scored as zero.
        if "inside of" in prompt:
            raise EligibilityError("`inside of` is unsupported by locked Official VBench")
        match = DIRECTIONAL.match(prompt)
        if not match:
            raise EligibilityError("prompt does not encode an ordered directional relation")
        return {
            "subject": match.group("subject").strip(),
            "object": match.group("object").strip(),
            "relation": match.group("relation"),
            "view": match.group("view").strip(),
        }
    if dimension == "multiplt_object":
        parts = prompt.split(" and ")
        if len(parts) != 2:
            raise EligibilityError(f"expected exactly two named targets, got {len(parts)}")
        return {"targets": [strip_article(part) for part in parts]}
    if dimension == "human_action":
        match = HUMAN_ACTION.match(prompt)
        if not match:
            raise EligibilityError("prompt is not an explicit `A person is <action>` prompt")
        return {"action": match.group("action").strip()}
    return {}


def strip_article(text: str) -> str:
    text = text.strip()
    for article in ("an ", "a ", "the "):
        if text.lower().startswith(article):
            return text[len(article) :].strip()
    return text


def choose_wrong_action(correct: str, vocabulary: Iterable[str]) -> str:
    """Pick a different Kinetics-400 action name by stable hash."""
    candidates = sorted({word for word in vocabulary if word != correct})
    if not candidates:
        raise EligibilityError("no alternative action name available")
    return stable_sample(candidates, 1, SEED)[0]


def select_dimension(
    dimension: str,
    prompts: dict[tuple[str, str], str],
    manifest: list[dict[str, str]],
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose dev/test bases for one dimension from eligible prompts only."""
    dev_target, test_target = BUDGET[dimension]
    rows = [row for row in manifest if row["dimension"] == dimension]
    if not rows:
        raise EligibilityError(f"no manifest rows for {dimension}")

    by_prompt: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    rejected: dict[str, int] = {}
    for row in rows:
        key = (dimension, row["prompt_id"])
        text = prompts.get(key)
        if text is None:
            rejected["prompt_text_missing"] = rejected.get("prompt_text_missing", 0) + 1
            continue
        try:
            parse_prompt(dimension, text)
        except EligibilityError as error:
            reason = str(error)
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        by_prompt.setdefault((row["split"], row["prompt_id"]), {})[row["generator"]] = row

    selected: list[dict[str, Any]] = []
    for split, target in (("dev", dev_target), ("test", test_target)):
        keys = sorted(key for key in by_prompt if key[0] == split)
        if len(keys) < target:
            raise EligibilityError(
                f"{dimension}/{split}: {len(keys)} eligible prompts < requested {target}"
            )
        chosen = stable_sample(keys, target, seed)
        for ordinal, key in enumerate(sorted(chosen)):
            generators = sorted(by_prompt[key])
            generator = stable_sample(generators, 1, f"{seed}:{key[1]}")[0]
            row = by_prompt[key][generator]
            text = prompts[(dimension, row["prompt_id"])]
            selected.append(
                {
                    "base_id": f"{dimension}-{row['video_uid']}",
                    "dimension": dimension,
                    "split": split,
                    "ordinal": ordinal,
                    "prompt_id": row["prompt_id"],
                    "prompt_en": text,
                    "group_id": row["group_id"],
                    "generator": row["generator"],
                    "video_uid": row["video_uid"],
                    "relative_video_path": row["relative_video_path"],
                    "parsed": parse_prompt(dimension, text),
                }
            )
    summary = {
        "dimension": dimension,
        "requested": {"dev": dev_target, "test": test_target},
        "selected": {
            "dev": sum(1 for row in selected if row["split"] == "dev"),
            "test": sum(1 for row in selected if row["split"] == "test"),
        },
        "eligible_prompts": {
            split: sum(1 for key in by_prompt if key[0] == split) for split in ("dev", "test")
        },
        "rejected_prompt_rows": rejected,
        "generators": {
            generator: sum(1 for row in selected if row["generator"] == generator)
            for generator in GENERATORS
        },
    }
    return selected, summary


def select_all(seed: int = SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = load_prompts()
    manifest = load_manifest()
    bases: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dimension in BUDGET:
        selected, summary = select_dimension(dimension, prompts, manifest, seed)
        bases.extend(selected)
        summaries.append(summary)
        print(
            f"{dimension:22s} dev {summary['selected']['dev']:3d}/{summary['requested']['dev']:3d}"
            f"  test {summary['selected']['test']:3d}/{summary['requested']['test']:3d}"
            f"  eligible prompts {summary['eligible_prompts']}"
        )
    overlap = dev_test_prompt_overlap(bases)
    if overlap:
        raise EligibilityError(f"prompt leaked across splits: {sorted(overlap)[:5]}")
    return bases, summaries


def dev_test_prompt_overlap(bases: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Prompts must never appear in both dev and test (plan section 3.4)."""
    seen: dict[tuple[str, str], set[str]] = {}
    for row in bases:
        seen.setdefault((row["dimension"], row["prompt_id"]), set()).add(row["split"])
    return {key for key, splits in seen.items() if len(splits) > 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/counterfactual/bases.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "output/counterfactual/bases_summary.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    bases, summaries = select_all(args.seed)
    count = write_jsonl(args.output, bases)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps({"bases": count, "per_dimension": summaries, "seed": args.seed}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "bases": count, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
