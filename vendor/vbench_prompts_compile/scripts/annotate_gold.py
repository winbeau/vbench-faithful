#!/usr/bin/env python3
"""Annotate prompts with DeepSeek (two passes + arbitration) into LLM-labelled gold sets.

Scope of this entry point: text-only tasks (spatial, action, objects). Scene items
need frames and real captions and are produced by the caption/vision pipeline.

Every provider call is charged to ``output/annotation/BUDGET-annotate-<name>.json``
before it is sent; ``--budget`` is a hard cap for the run.

Usage::

    # MovieGen long-prompt pilot (decision B1: existing long prompts first)
    uv run --no-sync python scripts/annotate_gold.py --name longprompt-pilot \\
        --source moviegen --buckets 50-99,100-200 --budget 300

    # full run over a bucket mix
    uv run --no-sync python scripts/annotate_gold.py --name gold-v1 \\
        --source moviegen+seed --buckets 1-14,15-49,50-99,100-200 --limit-per-task 120 --budget 4000
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import annotate as A  # noqa: E402
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import sources as S  # noqa: E402
from vbench_prompts_compile.llm import ChatClient, RequestBudget, load_tokens  # noqa: E402


def build_moviegen_items(tasks: Sequence[str], buckets: Sequence[str], limit_per_task: int | None, seed: int) -> list[A.Item]:
    import random

    refs = [ref for ref in S.load_moviegen_prompts() if R.length_bucket(ref.word_count) in set(buckets)]
    rng = random.Random(seed)
    rng.shuffle(refs)
    items: list[A.Item] = []
    for task in tasks:
        for index, ref in enumerate(refs):
            if limit_per_task is not None and index >= limit_per_task:
                break
            items.append(
                A.Item(
                    item_id=f"{task}-moviegen-{ref.sha256[:12]}",
                    task=task,
                    prompt=ref.text,
                    group_id=f"moviegen:{ref.sha256[:12]}",
                    bucket=R.length_bucket(ref.word_count),
                    source="moviegen",
                    source_id=f"{ref.source_file}:{ref.line}",
                    meta={"word_count": ref.word_count, "licence": "CC-BY-NC-4.0"},
                )
            )
    return items


def build_seed_items(tasks: Sequence[str], buckets: Sequence[str], limit_per_task: int | None, seed: int) -> list[A.Item]:
    """Seeds from the existing weak corpora (VG templates, K400 templates, fixtures)."""
    import random

    rng = random.Random(seed + 7)
    pools: dict[str, list[A.Item]] = defaultdict(list)
    vg = [row for row in R.read_jsonl(ROOT / "data" / "processed" / "formal-0001" / "candidates" / "spatial.jsonl")]
    rng.shuffle(vg)
    for row in vg[: (limit_per_task or 200)]:
        for task in ("spatial", "objects"):
            if task not in tasks or task not in ("spatial", "objects"):
                continue
            prompt = row["input"]["prompt"]
            if R.length_bucket(R.word_count(prompt)) not in set(buckets):
                continue
            pools[task].append(
                A.Item(
                    item_id=f"{task}-vg-{row['source_id']}",
                    task=task,
                    prompt=prompt,
                    group_id=row["group_id"],
                    bucket=R.length_bucket(R.word_count(prompt)),
                    source="visual_genome_prompt",
                    source_id=row["source_id"],
                    meta={"licence": "CC-BY-4.0"},
                )
            )
    k400 = R.read_jsonl(ROOT / "data" / "processed" / "formal-0001" / "candidates" / "action.jsonl")
    rng.shuffle(k400)
    for row in k400[: (limit_per_task or 200)]:
        if "action" not in tasks:
            continue
        prompt = row["input"]["prompt"]
        if R.length_bucket(R.word_count(prompt)) not in set(buckets):
            continue
        pools["action"].append(
            A.Item(
                item_id=f"action-k400-{row['source_id']}",
                task="action",
                prompt=prompt,
                group_id=row["group_id"],
                bucket=R.length_bucket(R.word_count(prompt)),
                source="k400_prompt",
                source_id=row["source_id"],
            )
        )
    items: list[A.Item] = []
    for task in tasks:
        items.extend(pools.get(task, [])[: (limit_per_task or len(pools.get(task, [])))])
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tasks", default="spatial,action,objects")
    parser.add_argument("--source", default="moviegen", help="moviegen | seed | moviegen+seed")
    parser.add_argument("--buckets", default="1-14,15-49,50-99,100-200")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--budget", type=int, default=200, help="hard cap of provider requests for this run")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--token-file", default=str(ROOT / "token.txt"))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    items: list[A.Item] = []
    if "moviegen" in args.source:
        items += build_moviegen_items(tasks, buckets, args.limit_per_task, args.seed)
    if "seed" in args.source:
        items += build_seed_items(tasks, buckets, args.limit_per_task, args.seed)
    if not items:
        print("no items selected", file=sys.stderr)
        return 2

    out_dir = ROOT / "data" / "gold" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(out_dir / "items.jsonl", [item.as_dict() for item in items])
    print(json.dumps({"items": len(items), "by_task": dict(Counter(i.task for i in items)), "by_bucket": dict(Counter(i.bucket for i in items))}, ensure_ascii=False))
    if args.dry_run:
        return 0

    tokens = load_tokens(args.token_file)
    client = ChatClient(provider=args.provider, api_key=tokens.get(args.provider), model=args.model)
    budget = RequestBudget(ROOT / "output" / "annotation", f"annotate-{args.name}", authorized_total=args.budget)
    vocab = S.load_k400()
    resolver = vocab.resolve

    gold: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for item in items:
        if budget.remaining() <= 0:
            print("budget exhausted; stopping", file=sys.stderr)
            break
        result = A.annotate_item(
            client,
            item,
            action_resolver=resolver,
            max_tokens=args.max_tokens,
            charge=lambda entry: budget.charge(entry, provider=args.provider),
        )
        entry = {
            "item_id": item.item_id,
            "task": item.task,
            "bucket": item.bucket,
            "status": result.status,
            "agreement": result.agreement,
            "errors": result.errors,
            "target": result.target,
            "usage": result.usage,
        }
        reports.append(entry)
        if result.target is not None:
            gold.append(R.add_length_meta(A.result_to_record(result, item, action_resolver=resolver)))
        else:
            quarantine.append({**entry, "prompt": item.prompt, "passes": result.passes})
        budget.append({"kind": "result", "item_id": item.item_id, "status": result.status, "agreement": result.agreement, "errors": result.errors})

    R.write_jsonl(out_dir / "gold.jsonl", gold)
    R.write_jsonl(out_dir / "quarantine.jsonl", quarantine)
    status_counts = Counter(entry["status"] for entry in reports)
    report = {
        "name": args.name,
        "source": args.source,
        "buckets": buckets,
        "tasks": tasks,
        "items": len(items),
        "processed": len(reports),
        "status_counts": dict(status_counts),
        "agreement_rate": (sum(1 for entry in reports if entry["agreement"]) / len(reports)) if reports else 0.0,
        "by_task": {task: dict(Counter(entry["status"] for entry in reports if entry["task"] == task)) for task in tasks},
        "by_bucket": {bucket: dict(Counter(entry["status"] for entry in reports if entry["bucket"] == bucket)) for bucket in buckets},
        "error_counts": dict(Counter(error for entry in reports for error in entry["errors"])),
        "target_kinds": {
            "action_other": sum(1 for record in gold if record["task"] == "action" and R.ACTION_OTHER in record["target"]["actions"]),
            "empty_results": sum(1 for record in gold if isinstance(record["target"], dict) and any(isinstance(v, list) and not v for v in record["target"].values())),
        },
        "budget": budget.summary(),
        "quality": A.ANNOTATION_QUALITY,
        "note": "LLM labels (two passes + arbitration), not human gold",
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
