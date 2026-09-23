#!/usr/bin/env python3
"""Label student-visible prompt/caption evidence, resuming every API pass.

Visual labels and conflicts remain in metadata. Duplicate historical observations
are merged explicitly; successful text responses are shared for identical inputs.
The full input denominator, failures, request count and token usage are reported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import annotate as A  # noqa: E402
from vbench_prompts_compile import annotation_jobs as J  # noqa: E402
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile.llm import BudgetExceeded, ChatClient, RequestBudget, load_tokens  # noqa: E402

EVIDENCE_INSTRUCTION = (
    "You read one text-to-video prompt and one automatic caption of a single video frame. "
    "You do NOT see the frame. Decide whether the caption gives evidence about the scene the "
    "prompt requires (a place or environment such as a beach, ocean, forest, street, bathroom, office). "
    'Answer exactly {"label":"supported"} or {"label":"contradicted"} or {"label":"insufficient"}. '
    "supported: the caption clearly describes the scene the prompt requires, even if it uses different words. "
    "contradicted: the caption clearly describes a different scene. "
    "insufficient: the prompt requires no scene, or the caption does not mention the scene or is too vague to decide. "
    "Never infer the scene from the prompt alone when the caption is silent."
)
EVIDENCE_SYSTEM = (
    "You label text pairs for a research dataset. Judge only what the caption states. "
    "Answer with a single JSON object and nothing else."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, action="append", help="repeat to merge historical batches")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, default=1200, help="cumulative request cap for this purpose")
    parser.add_argument("--budget-purpose", default="scene-evidence")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None, help="new items this run; keeps the full plan")
    parser.add_argument("--token-file", default=str(ROOT / "token.txt"))
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(args, out: Path) -> int:
    rows, audit = J.merge_observations([r for p in args.gold for r in R.read_jsonl(p)])
    if args.limit is not None:
        rows = rows[:args.limit]
    plan = {"input": args.gold, "observations": len(rows), "audit": audit,
            "input_sha256": {p: R.sha256_text(Path(p).read_text(encoding="utf-8")) for p in args.gold},
            "instruction_sha256": R.sha256_text(EVIDENCE_SYSTEM + EVIDENCE_INSTRUCTION),
            "provider": args.provider, "model": args.model, "max_tokens": args.max_tokens, "limit": args.limit}
    plan_path = out.with_suffix(".plan.json")
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("annotation inputs/config changed; use a new output, do not mix runs")
    J.atomic_json(plan_path, plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    tokens = load_tokens(args.token_file)
    budget = RequestBudget(ROOT / "output" / "annotation", args.budget_purpose, authorized_total=args.budget)
    client = J.ResumableChatClient(ChatClient(provider=args.provider, api_key=tokens.get(args.provider), model=args.model), budget, out.with_suffix(".requests.jsonl"))
    completed = J.read_jsonl(out)
    done = {r["sample_id"] for r in completed}
    if len(done) != len(completed) or not done <= {r["sample_id"] for r in rows}:
        raise ValueError("existing output has duplicate or unplanned IDs; use a new output")
    attempted = 0
    stopped = None
    for row in rows:
        if row["sample_id"] in done:
            continue
        if args.max_items is not None and attempted >= args.max_items:
            break
        item = A.Item(item_id=row["sample_id"], task="scene", prompt=row["input"]["prompt"],
                      caption=row["input"]["caption"], group_id=row["group_id"],
                      bucket=row.get("meta", {}).get("length_bucket", "1-14"),
                      source=row["source"], source_id=row["source_id"])
        try:
            result = A.annotate_item(client, item, action_resolver=None, max_tokens=args.max_tokens,
                                     instruction_override=EVIDENCE_INSTRUCTION, system_override=EVIDENCE_SYSTEM)
        except (BudgetExceeded, J.ProviderUnavailable) as error:
            stopped = str(error)
            break
        attempted += 1
        if result.target is None:
            J.append_jsonl(out.with_suffix(".failures.jsonl"), asdict(result))
            continue
        updated = {**row, "target": result.target, "quality": "llm_annotated_arbitrated", "meta": {
            **row["meta"], "caption_evidence_label": result.target, "evidence_status": result.status,
            "evidence_agreement": result.agreement, "evidence_passes": result.passes,
            "label_source": f"{args.provider}:{client.model}:text_prompt_plus_caption"}}
        updated = R.add_length_meta(updated)
        J.append_jsonl(out, updated)
        completed.append(updated)
        done.add(row["sample_id"])
        print(json.dumps({"completed": len(done), "planned": len(rows), "used_requests": budget.used}), flush=True)
    report = {**plan, "relabelled": len(done), "missing": len(rows) - len(done),
              "complete": len(done) == len(rows), "stopped": stopped,
              "label_changes_vs_known_visual_truth": sum(r["target"] != r["meta"]["visual_truth"] for r in completed if r["meta"]["visual_truth"] is not None),
              "evidence_label_counts": dict(Counter(r["target"] for r in completed)),
              "budget": budget.summary(), "usage": client.usage(),
              "note": "LLM labels, not human gold; no image sent; visual conflicts remain unknown"}
    J.atomic_json(out.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    with J.job_lock(out.with_suffix(".lock")):
        return run(args, out)


if __name__ == "__main__":
    raise SystemExit(main())
