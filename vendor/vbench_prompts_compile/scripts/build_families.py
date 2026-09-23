#!/usr/bin/env python3
"""Build prompt families: one seed + rewritten length variants in other buckets.

Decision B1 says long prompts are obtained by **rewriting, never truncation**. A
family is a seed prompt plus its rewrites; all members share ``group_id`` so a
train/dev split can never separate them.

Semantic preservation is checked mechanically: the seed's own label set must still
be extractable from the rewrite (the rewrite is re-annotated with the same
two-pass + arbitration pipeline and must be a *superset* of the seed target).
Rewrites that lose a relation, action or entity are quarantined.

Usage::

    uv run --no-sync python scripts/build_families.py --name families-pilot \\
        --seeds data/gold/longprompt-pilot-v4/gold.jsonl --buckets 1-14,201-400 \\
        --limit-per-task 5 --budget 300
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import annotate as A  # noqa: E402
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import sources as S  # noqa: E402
from vbench_prompts_compile.llm import ChatClient, RequestBudget, load_tokens  # noqa: E402

BUCKET_RANGES: dict[str, tuple[int, int]] = {
    "1-14": (8, 14),
    "15-49": (22, 45),
    "50-99": (55, 95),
    "100-200": (110, 190),
    "201-400": (210, 380),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", required=True, help="gold.jsonl produced by annotate_gold.py")
    parser.add_argument("--buckets", default="1-14,201-400")
    parser.add_argument("--tasks", default="spatial,action,objects")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--token-file", default=str(ROOT / "token.txt"))
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--rewrite-max-tokens", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    for bucket in buckets:
        if bucket not in BUCKET_RANGES:
            print(f"unknown bucket {bucket}", file=sys.stderr)
            return 2

    seed_rows = [row for row in R.read_jsonl(args.seeds) if row["task"] in tasks]
    selected: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for row in seed_rows:
        if args.limit_per_task is not None and seen[row["task"]] >= args.limit_per_task:
            continue
        seen[row["task"]] += 1
        selected.append(row)
    if not selected:
        print("no seeds selected", file=sys.stderr)
        return 2

    out_dir = ROOT / "data" / "gold" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [
        {"sample_id": row["sample_id"], "task": row["task"], "group_id": row["group_id"], "seed_bucket": row.get("meta", {}).get("length_bucket"), "target_bucket": bucket}
        for row in selected
        for bucket in buckets
        if row.get("meta", {}).get("length_bucket") != bucket
    ]
    (out_dir / "plan.json").write_text(json.dumps({"variants": len(plan), "items": plan}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seeds": len(selected), "variants": len(plan), "by_task": dict(Counter(row["task"] for row in selected))}, ensure_ascii=False))
    if args.dry_run:
        return 0

    tokens = load_tokens(args.token_file)
    client = ChatClient(provider=args.provider, api_key=tokens.get(args.provider), model=args.model)
    budget = RequestBudget(ROOT / "output" / "annotation", f"families-{args.name}", authorized_total=args.budget)
    resolver = S.load_k400().resolve

    gold: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in selected:
        for bucket in buckets:
            if row.get("meta", {}).get("length_bucket") == bucket:
                continue
            if budget.remaining() <= 0:
                print("budget exhausted; stopping", file=sys.stderr)
                break
            min_words, max_words = BUCKET_RANGES[bucket]
            rewrite = A.rewrite_prompt(
                client,
                row["input"]["prompt"],
                min_words=min_words,
                max_words=max_words,
                max_tokens=args.rewrite_max_tokens,
                charge=lambda entry, provider=args.provider: budget.charge(entry, provider=provider),
                item_id=row["sample_id"],
            )
            entry: dict[str, Any] = {
                "seed_sample_id": row["sample_id"],
                "task": row["task"],
                "group_id": row["group_id"],
                "target_bucket": bucket,
                "rewrite_words": rewrite.words,
                "rewrite_errors": rewrite.errors,
            }
            if rewrite.text is None or not rewrite.in_range:
                entry["status"] = "rewrite_failed"
                quarantine.append({**entry, "seed_prompt": row["input"]["prompt"]})
                budget.append({"kind": "result", **entry})
                continue
            item = A.Item(
                item_id=f"{row['sample_id']}--{bucket}",
                task=row["task"],
                prompt=rewrite.text,
                group_id=row["group_id"],
                bucket=R.length_bucket(rewrite.words),
                source=row["source"],
                source_id=f"{row['source_id']}|rewrite:{bucket}",
                meta={"derivation": "rewrite", "seed_sample_id": row["sample_id"], "seed_bucket": row.get("meta", {}).get("length_bucket"), "target_bucket": bucket},
            )
            result = A.annotate_item(
                client,
                item,
                action_resolver=resolver,
                max_tokens=args.max_tokens,
                charge=lambda charge_entry, provider=args.provider: budget.charge(charge_entry, provider=provider),
            )
            if result.target is None:
                entry.update({"status": "annotation_failed", "errors": result.errors, "rewrite": rewrite.text})
                quarantine.append({**entry, "seed_prompt": row["input"]["prompt"]})
                budget.append({"kind": "result", **entry})
                continue
            preserved, missing = A.target_preserved(row["target"], result.target, row["task"])
            entry.update({"status": "accepted" if preserved else "semantics_lost", "missing": missing, "agreement": result.agreement, "usage": result.usage})
            if not preserved:
                quarantine.append({**entry, "rewrite": rewrite.text, "seed_prompt": row["input"]["prompt"], "seed_target": row["target"], "rewritten_target": result.target})
                budget.append({"kind": "result", **entry})
                continue
            record = A.result_to_record(result, item, action_resolver=resolver)
            record["meta"].update(
                {
                    "derivation": "rewrite",
                    "seed_sample_id": row["sample_id"],
                    "seed_bucket": row.get("meta", {}).get("length_bucket"),
                    "target_bucket": bucket,
                    "rewrite_words": rewrite.words,
                    "family_id": row["group_id"],
                }
            )
            gold.append(R.add_length_meta(record))
            budget.append({"kind": "result", **entry})

    R.write_jsonl(out_dir / "gold.jsonl", gold)
    R.write_jsonl(out_dir / "quarantine.jsonl", quarantine)
    report = {
        "name": args.name,
        "seeds": args.seeds,
        "buckets": buckets,
        "seeds_used": len(selected),
        "variants_planned": len(plan),
        "variants_accepted": len(gold),
        "quarantined": len(quarantine),
        "by_task": {task: dict(Counter(record["task"] for record in gold if record["task"] == task)) for task in tasks},
        "by_bucket": {bucket: sum(1 for record in gold if record["meta"].get("target_bucket") == bucket) for bucket in buckets},
        "quarantine_reasons": dict(Counter(entry["status"] for entry in quarantine)),
        "budget": budget.summary(),
        "note": "rewritten length variants, semantics checked by re-annotation superset; LLM labels",
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
