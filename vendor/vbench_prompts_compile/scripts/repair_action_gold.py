#!/usr/bin/env python3
"""Re-annotate action gold items whose label is the `other` sentinel.

The first annotation round asked the model to decide K400 membership itself, which
produced false `other` labels for real Kinetics-400 classes ("frying vegetables",
"busking", "making jewelry"). The instruction now asks only for the prompt's own
action phrase; this script re-annotates the sentinel items under the new rules so
the deterministic resolver decides membership instead of the model.

The previous label is kept in ``meta.repair`` for audit.

Usage::

    uv run --no-sync python scripts/repair_action_gold.py --out data/gold/action-repair \\
        --gold-dirs seeds-short,families-mid,longprompt-pilot-v4 --budget 900
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import annotate as A  # noqa: E402
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import sources as S  # noqa: E402
from vbench_prompts_compile.llm import ChatClient, RequestBudget, load_tokens  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="action", choices=["action", "objects"])
    parser.add_argument("--out", default="data/gold/action-repair")
    parser.add_argument("--gold-dirs", default="seeds-short,families-mid,longprompt-pilot-v4")
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--token-file", default=str(ROOT / "token.txt"))
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--max-tokens", type=int, default=384)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict] = []
    for name in [item.strip() for item in args.gold_dirs.split(",") if item.strip()]:
        path = ROOT / "data" / "gold" / name / "gold.jsonl"
        if path.exists():
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.task == "action":
        targets = [row for row in rows if row["task"] == "action" and "other" in row["target"]["actions"]]
    else:
        # Objects: re-annotate every gold row; the earlier rows were canonicalised with a
        # buggy stemmer ("hills" -> "hil"), so the labels have to be rebuilt.
        targets = [row for row in rows if row["task"] == "objects"]
    print(json.dumps({"candidates": len(targets), "from_total": len(rows)}), flush=True)

    tokens = load_tokens(args.token_file)
    client = ChatClient(provider=args.provider, api_key=tokens.get(args.provider))
    budget = RequestBudget(ROOT / "output" / "annotation", f"{args.task}-repair", authorized_total=args.budget)
    resolver = S.load_k400().resolve

    repaired: list[dict] = []
    still_other: list[dict] = []
    for row in targets:
        if budget.remaining() <= 0:
            print("budget exhausted", file=sys.stderr)
            break
        item = A.Item(
            item_id=row["sample_id"],
            task=args.task,
            prompt=row["input"]["prompt"],
            group_id=row["group_id"],
            bucket=row.get("meta", {}).get("length_bucket", "1-14"),
            source=row["source"],
            source_id=row["source_id"],
        )
        result = A.annotate_item(
            client,
            item,
            action_resolver=resolver,
            max_tokens=args.max_tokens,
            charge=lambda entry, provider=args.provider: budget.charge(entry, provider=provider),
        )
        if result.target is None:
            continue
        record = A.result_to_record(result, item, action_resolver=resolver)
        record["meta"]["repair"] = {
            "previous_target": row["target"],
            "previous_status": row.get("meta", {}).get("status"),
            "reason": "other sentinel re-annotated under phrase-transcription rules",
        }
        if args.task == "action":
            marker = "other" if "other" in record["target"]["actions"] else "k400"
        else:
            marker = "objects"
        (still_other if marker == "other" else repaired).append(record)
        budget.append({"kind": "result", "sample_id": row["sample_id"], "marker": marker, "target": record["target"]})

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(out_dir / "gold.jsonl", repaired + still_other)
    report = {
        "candidates": len(targets),
        "repaired_to_k400": len(repaired),
        "still_other": len(still_other),
        "budget": budget.summary(),
        "note": "override records; merge with --gold-override in build_formal_sets.py",
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
