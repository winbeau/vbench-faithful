#!/usr/bin/env python3
"""Scene gold data from real frames: Tag2Text captions + gpt-5.6-luna vision labels (B5).

Input is the output of ``scripts/semantic/caption_frames.py`` (one real frame per video, its
prompt and the official Tag2Text caption). Three pair types are built:

* ``supported``    -- prompt paired with its own frame/caption;
* ``contradicted`` -- prompt paired with the frame/caption of a *different* scene prompt;
* ``insufficient`` -- a prompt that states no scene requirement, paired with any frame.

The label always comes from the vision model looking at the **frame** (two passes +
arbitration); the caption is only what the trained model will see. Expected-vs-actual
pair agreement is recorded, never silently "fixed".

Usage::

    python scripts/semantic/annotate_scene_vision.py \
        --captions data/gold/scene-dev-captions.jsonl --split-name dev \
        --output data/gold/scene-gold-dev.jsonl --supported 40 --contradicted 40 \
        --insufficient 30 --budget 400
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/prompt-compiler/src"))

sys.path.insert(0, str(ROOT))
from scripts.semantic.paths import DATA_ROOT, RAW_ROOT, FIXTURES_ROOT
from vbench_prompts_compile import annotate as A  # noqa: E402
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import annotation_jobs as J  # noqa: E402
from vbench_prompts_compile.llm import BudgetExceeded, ChatClient, RequestBudget, load_tokens  # noqa: E402

SCENE_WORDS = (
    "beach", "ocean", "sea", "river", "lake", "forest", "desert", "mountain", "city", "street",
    "village", "room", "bedroom", "kitchen", "bathroom", "office", "classroom", "library", "park",
    "garden", "museum", "gallery", "cafe", "restaurant", "bar", "hospital", "laboratory", "farm",
    "factory", "stadium", "arena", "subway", "bus", "train", "airport", "station", "church", "temple",
    "cave", "reef", "underwater", "space", "planet", "snow", "rain", "sky", "sunset", "sunrise",
    "night", "indoor", "outdoor", "field", "meadow", "castle", "temple", "shop", "market", "bridge",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--split-name", default="dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--supported", type=int, default=40)
    parser.add_argument("--contradicted", type=int, default=40)
    parser.add_argument("--insufficient", type=int, default=30)
    parser.add_argument("--scene-less-file", default=None, help="JSONL with scene-less prompts (one prompt per line)")
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--gate-provider", default="deepseek", help="text gate for the insufficient pool; set to '' to disable")
    parser.add_argument("--gate-model", default=None)
    parser.add_argument("--provider", default="chiyi")
    parser.add_argument("--model", default=None)
    parser.add_argument("--token-file", default=str(ROOT / "token.txt"))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--items-plan", help="frozen Item JSONL; reuse it without rebuilding/gating")
    parser.add_argument("--budget-purpose", default=None)
    parser.add_argument("--max-items", type=int, default=None, help="stop after this many new items; resume the same plan later")
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("frame_path"):
                rows.append(row)
    return rows


def scene_less_prompts(path: Path | None, count: int, seed: int) -> list[str]:
    if path is not None and path.exists():
        return [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:count]
    from vbench_prompts_compile.sources import load_moviegen_prompts

    rng = random.Random(seed)
    refs = [ref.text for ref in load_moviegen_prompts()]
    rng.shuffle(refs)
    picked = [text for text in refs if not any(word in text.lower() for word in SCENE_WORDS)]
    return picked[:count]


def build_items(args: argparse.Namespace, gate_client=None, budget=None) -> tuple[list[tuple[A.Item, str]], dict[str, Any]]:
    rows = load_rows(Path(args.captions))
    if not rows:
        raise SystemExit("no usable caption rows")
    rng = random.Random(args.seed)
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt"], []).append(row)
    prompts = sorted(by_prompt)
    # Keep cross-pair dependencies inside disjoint blocks of two (last block
    # may contain three). Otherwise a cycle joins the entire test set into one
    # shared-source component and a family bootstrap has no independent units.
    blocks = [prompts[i:i + 2] for i in range(0, len(prompts), 2)]
    if len(blocks) > 1 and len(blocks[-1]) == 1:
        blocks[-2].extend(blocks.pop())
    block_by_prompt = {p: block for block in blocks for p in block}
    items: list[tuple[A.Item, str]] = []

    # Round-robin across distinct prompts: a first-N slice clusters in a handful of
    # families, which destroys the independence the scorecard needs (P0-3).
    supported_rows: list[dict[str, Any]] = []
    cursors = {prompt: 0 for prompt in prompts}
    while len(supported_rows) < args.supported:
        progressed = False
        for prompt in prompts:
            if len(supported_rows) >= args.supported:
                break
            rows_for_prompt = by_prompt[prompt]
            if cursors[prompt] < len(rows_for_prompt):
                supported_rows.append(rows_for_prompt[cursors[prompt]])
                cursors[prompt] += 1
                progressed = True
        if not progressed:
            break
    for row in supported_rows:
        items.append((_item(row, row, "supported", args.split_name), "supported"))

    # Cross-pair every source prompt with different target prompts, cycling targets so
    # all prompt pairs (families) contribute instead of one repeated pair.
    made = 0
    source_rows = supported_rows or rows
    partner_cursors = {prompt: 0 for prompt in prompts}
    seen_pairs: set[str] = set()
    for number, source in enumerate(source_rows):
        if made >= args.contradicted:
            break
        block = block_by_prompt[source["prompt"]]
        for offset in range(1, len(block) + 1):
            target_prompt = block[(block.index(source["prompt"]) + offset + number // len(prompts)) % len(block)]
            if target_prompt == source["prompt"]:
                continue
            candidates = by_prompt[target_prompt]
            for _ in candidates:
                partner = candidates[partner_cursors[target_prompt] % len(candidates)]
                partner_cursors[target_prompt] += 1
                item = _item(source, partner, "contradicted", args.split_name)
                if item.item_id not in seen_pairs:
                    seen_pairs.add(item.item_id)
                    items.append((item, "contradicted"))
                    made += 1
                    break
            else:
                continue
            break

    scene_less = scene_less_prompts(Path(args.scene_less_file) if args.scene_less_file else None, args.insufficient * 3, args.seed)
    gate_stats = {"candidates": len(scene_less), "no_scene": 0, "has_scene": 0, "unknown": 0}
    accepted_scene_less: list[str] = []
    for prompt in scene_less:
        if len(accepted_scene_less) >= args.insufficient:
            break
        if gate_client is None:
            accepted_scene_less.append(prompt)
            continue
        requires, _usage = A.gate_scene_requirement(
            gate_client,
            prompt,
            charge=(lambda entry, provider=gate_client.provider: budget.charge(entry, provider=provider)) if budget is not None else None,
            item_id=f"gate:{R.sha256_text(prompt)[:10]}",
        )
        if requires is False:
            gate_stats["no_scene"] += 1
            accepted_scene_less.append(prompt)
        elif requires is True:
            gate_stats["has_scene"] += 1
        else:
            gate_stats["unknown"] += 1
    for number, prompt in enumerate(accepted_scene_less):
        frame_prompt = prompts[number % len(prompts)]
        frame_row = by_prompt[frame_prompt][(number // len(prompts)) % len(by_prompt[frame_prompt])]
        item = A.Item(
            item_id="scene-insufficient-" + R.sha256_text(R.canonical_json([args.split_name, prompt, frame_row["video_path"], frame_row.get("frame_index"), frame_row["frame_path"]]))[:24],
            task="scene",
            prompt=prompt,
            caption=frame_row["caption"],
            image=frame_row["frame_path"],
            group_id=f"scene-less:{R.sha256_text(prompt)[:10]}",
            bucket=R.length_bucket(R.word_count(prompt)),
            source="scene_sceneless_prompt",
            source_id=f"split={args.split_name}",
            meta={"pair_type": "insufficient", "split": args.split_name, "frame_source_prompt": frame_row["prompt"], "frame_video": frame_row["video_path"], "frame_index": frame_row.get("frame_index"), "licence": "prompt pool CC-BY-NC-4.0; frame from VBench public videos"},
        )
        items.append((item, "insufficient"))
    for item, _ in items:
        block = block_by_prompt[item.meta["frame_source_prompt"]]
        item.meta["dependency_family"] = "scene-block:" + R.sha256_text(R.canonical_json(block))[:16]
        item.meta["dependency_source_prompts"] = block
    return items, gate_stats


def _item(source_row: dict[str, Any], frame_row: dict[str, Any], pair_type: str, split: str) -> A.Item:
    return A.Item(
        item_id=f"scene-{pair_type}-" + R.sha256_text(R.canonical_json([split, source_row["prompt"], frame_row["video_path"], frame_row.get("frame_index"), frame_row["frame_path"]]))[:24],
        task="scene",
        prompt=source_row["prompt"],
        caption=frame_row["caption"],
        image=frame_row["frame_path"],
        group_id=f"scene-prompt:{R.sha256_text(source_row['prompt'])[:10]}",
        bucket=R.length_bucket(R.word_count(source_row["prompt"])),
        source="vbench_human_preference_scene",
        source_id=f"split={split}|gen={frame_row.get('generator')}",
        meta={
            "pair_type": pair_type,
            "split": split,
            "frame_source_prompt": frame_row["prompt"],
            "frame_video": frame_row["video_path"],
            "frame_index": frame_row.get("frame_index"),
            "generator": frame_row.get("generator"),
            "licence": "VBench 1.0 public human-preference videos; captions from official Tag2Text",
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with J.job_lock(out_path.with_suffix(".lock")):
        return run(args, out_path)


def run(args, out_path: Path) -> int:
    plan_path = Path(args.items_plan) if args.items_plan else out_path.with_suffix(".plan.jsonl")
    budget = None
    gate_stats = {"source": "frozen_plan"}
    if plan_path.exists():
        allowed = set(A.Item.__dataclass_fields__)
        items = [(A.Item(**{k: v for k, v in row.items() if k in allowed}), row["meta"]["pair_type"])
                 for row in J.read_jsonl(plan_path)]
    else:
        gate_client = None
        if not args.dry_run and args.gate_provider:
            tokens = load_tokens(args.token_file)
            budget = RequestBudget(ROOT / "output" / "annotation", args.budget_purpose or f"scene-{args.split_name}", authorized_total=args.budget)
            gate_client = J.ResumableChatClient(ChatClient(provider=args.gate_provider, api_key=tokens.get(args.gate_provider), model=args.gate_model), budget, out_path.with_suffix(".gate-requests.jsonl"))
        items, gate_stats = build_items(args, gate_client, None)
        R.write_jsonl(plan_path, [item.as_dict() for item, _ in items])
    ids = [item.item_id for item, _ in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate annotation item IDs in plan")
    print(json.dumps({"items": len(items), "pair_types": dict(Counter(pair for _, pair in items)), "scene_gate": gate_stats}), flush=True)
    if args.dry_run:
        return 0
    missing = [item.image for item, _ in items if not item.image or not Path(item.image).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} planned images missing; first: {missing[0]}")
    tokens = load_tokens(args.token_file)
    budget = budget or RequestBudget(ROOT / "output" / "annotation", args.budget_purpose or f"scene-{args.split_name}", authorized_total=args.budget)
    client = J.ResumableChatClient(ChatClient(provider=args.provider, api_key=tokens.get(args.provider), model=args.model), budget, out_path.with_suffix(".requests.jsonl"))
    J.bind_job(out_path.with_suffix(".job.json"), {"plan_sha256": R.sha256_text(plan_path.read_text()),
        "provider": client.provider, "model": client.model, "max_tokens": args.max_tokens,
        "instruction_sha256": R.sha256_text(A.INSTRUCTIONS["scene"] + A.SYSTEM_VISION)})
    gold = J.read_jsonl(out_path)
    done = {r["sample_id"] for r in gold}
    if len(done) != len(gold) or not done <= set(ids):
        raise ValueError("existing output has duplicate or unplanned IDs; preserve it and use a new output")
    quarantine_path = out_path.with_name(out_path.stem + "-quarantine.jsonl")
    quarantine = J.read_jsonl(quarantine_path)
    attempted = 0
    stopped = None
    for item, pair_type in items:
        if item.item_id in done:
            continue
        if args.max_items is not None and attempted >= args.max_items:
            break
        try:
            result = A.annotate_item(client, item, action_resolver=None, max_tokens=args.max_tokens)
        except (J.ProviderUnavailable, BudgetExceeded) as error:
            stopped = str(error)
            break
        attempted += 1
        entry = {"item_id": item.item_id, "pair_type": pair_type, "status": result.status, "agreement": result.agreement, "errors": result.errors, "target": result.target}
        if result.target is None:
            entry.update({"prompt": item.prompt, "caption": item.caption, "passes": result.passes})
            quarantine.append(entry)
            J.append_jsonl(quarantine_path, entry)
            continue
        record = A.result_to_record(result, item)
        record["sample_id"] = item.item_id
        record["meta"].update({"pair_type": pair_type, "expected_label": pair_type, "pair_label_agrees": result.target == pair_type, "visual_truth": result.target, "annotation_passes": result.passes})
        record = R.add_length_meta(record)
        J.append_jsonl(out_path, record)
        gold.append(record)
        done.add(item.item_id)
        print(json.dumps({"completed": len(done), "planned": len(items), "used_requests": budget.used}), flush=True)
    expected_counts = Counter(pair for _, pair in items)
    agree_counts = Counter(r["meta"]["pair_type"] for r in gold if r["meta"]["pair_label_agrees"])
    report = {
        "split": args.split_name,
        "items": len(items),
        "records": len(gold),
        "quarantined": len({r["item_id"] for r in quarantine} - done),
        "missing": len(items) - len(done),
        "stopped": stopped,
        "complete": len(done) == len(items),
        "label_counts": dict(Counter(record["target"] for record in gold)),
        "pair_type_counts": dict(expected_counts),
        "pair_label_agreement": {key: f"{agree_counts[key]}/{expected_counts[key]}" for key in expected_counts},
        "budget": budget.summary(),
        "usage": client.usage(),
        "plan_sha256": R.sha256_text(plan_path.read_text(encoding="utf-8")),
        "captioner": "official VBench Tag2Text (swin_b, 14m)",
        "labeller": f"{args.provider}:{client.model}",
        "scene_gate": gate_stats,
        "note": "labels come from the frame (vision model, two passes + arbitration); caption is only the model input; insufficient items come from the text gate reporting no scene requirement",
    }
    J.atomic_json(out_path.with_name(out_path.stem + "-report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
