#!/usr/bin/env python3
"""Run the authorised teacher pilot (<=19 charged requests) and ingest results.

Safety properties:

* the plan is resolved and written *before* any request is sent;
* every POST is charged to ``output/teacher/BUDGET.json`` before it leaves;
* ``--confirm-budget N`` must match the resolved item count exactly;
* failures are recorded, never repaired; at most one retry, also charged;
* no key material is printed or written, and HTML/error bodies are not echoed.

Usage::

    uv run --no-sync python scripts/run_teacher_pilot.py --config configs/teacher/pilot-01.json --dry-run
    uv run --no-sync python scripts/run_teacher_pilot.py --config configs/teacher/pilot-01.json --confirm-budget 19
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import pilot as P  # noqa: E402
from vbench_prompts_compile import records as R  # noqa: E402
from vbench_prompts_compile import sources as S  # noqa: E402
from vbench_prompts_compile.teacher import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    BudgetExceeded,
    BudgetLedger,
    DeepSeekClient,
    TeacherError,
    build_request,
    ingest_teacher_output,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/teacher/pilot-01.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--build-id", default="local-0001")
    parser.add_argument("--token-file", default=str(ROOT / "token.txt"))
    parser.add_argument("--confirm-budget", type=int, default=0, help="must equal the resolved request count to spend")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only-group", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = P.load_config(ROOT / args.config)
    items = P.resolve_items(config)
    if args.only_group:
        items = [item for item in items if item.group == args.only_group]
    run_id = args.run_id or utc_stamp()
    run_dir = DEFAULT_OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    limits = config["limits"]
    max_output_tokens = min(int(args.max_output_tokens or limits["max_output_tokens"]), int(limits["max_output_tokens"]))
    plan = {
        "pilot_id": config["pilot_id"],
        "run_id": run_id,
        "config": args.config,
        "model": config["model"],
        "base_url": config["base_url"],
        "authorized_total_requests": int(config["authorized_total_requests"]),
        "requested_requests": len(items),
        "max_output_tokens": max_output_tokens,
        "temperature": args.temperature,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": [item.as_plan_dict() for item in items],
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"run_id={run_id} items={len(items)} model={config['model']} max_output_tokens={max_output_tokens}")
    for item in items:
        print(f"  {item.item_id:22} {item.task:8} {R.length_bucket(item.entry.word_count):9} {item.entry.source:16} {item.entry.source_id}")
    if args.dry_run:
        print("dry run: no request sent")
        return 0

    if args.confirm_budget != len(items):
        print(f"refusing to spend: --confirm-budget {args.confirm_budget} != resolved items {len(items)}", file=sys.stderr)
        return 2

    ledger = BudgetLedger(
        run_dir,
        authorized_total=int(config["authorized_total_requests"]),
        max_output_tokens=max_output_tokens,
        budget_path=DEFAULT_OUTPUT_ROOT / "BUDGET.json",
    )
    if ledger.remaining() < len(items):
        print(f"refusing to spend: remaining budget {ledger.remaining()} < items {len(items)}", file=sys.stderr)
        return 2

    vocab = S.load_k400()
    client = DeepSeekClient(
        token_path=args.token_file,
        base_url=str(config["base_url"]),
        model=str(config["model"]),
        timeout_seconds=float(limits["timeout_seconds"]),
    )

    candidates_path = run_dir / "candidates.jsonl"
    quarantine_path = run_dir / "quarantine.jsonl"
    results: list[dict[str, Any]] = []
    with candidates_path.open("w", encoding="utf-8") as candidates_handle, quarantine_path.open("w", encoding="utf-8") as quarantine_handle:
        for item in items:
            system, user = build_request(item.task, item.prompt, item.caption)
            charge = {
                "item_id": item.item_id,
                "group": item.group,
                "task": item.task,
                "bucket": item.bucket,
                "prompt_sha256": item.entry.sha256,
                "max_tokens": max_output_tokens,
                "model": config["model"],
            }
            try:
                charge_index = ledger.charge(charge)
            except BudgetExceeded as error:
                print(f"budget stop: {error}", file=sys.stderr)
                break
            attempts = 0
            response = None
            error_note = None
            while attempts < 2 and response is None:
                attempts += 1
                if attempts > 1:  # a retry is a new, charged request
                    try:
                        charge_index = ledger.charge({**charge, "retry_of": charge_index})
                    except BudgetExceeded as error:
                        error_note = f"budget_stop:{error}"
                        break
                try:
                    response = client.chat(user=user, system=system, max_tokens=max_output_tokens, temperature=args.temperature)
                except TeacherError as error:
                    error_note = str(error)
                    ledger.append_result({"item_id": item.item_id, "attempt": attempts, "error": error_note, "charge_index": charge_index})
            if response is None:
                entry = {
                    "item_id": item.item_id,
                    "task": item.task,
                    "bucket": item.bucket,
                    "source": item.entry.source,
                    "source_id": item.entry.source_id,
                    "status": "transport_error",
                    "error": error_note,
                    "charge_index": charge_index,
                    "attempts": attempts,
                }
                results.append(entry)
                quarantine_handle.write(json.dumps({"item_id": item.item_id, "reason": "transport_error", "detail": error_note}, ensure_ascii=False) + "\n")
                print(f"  [error] {item.item_id}: {error_note}")
                continue
            ledger.append_result({"item_id": item.item_id, "attempt": attempts, "charge_index": charge_index, **response.as_metadata(), "response_sha256": R.sha256_text(response.text), "finish_reason": response.finish_reason})
            ingest = ingest_teacher_output(
                task=item.task,
                prompt=item.prompt,
                caption=item.caption,
                response=response,
                source=item.entry.source,
                source_id=item.entry.source_id,
                group_id=item.entry.group_id,
                max_output_tokens=max_output_tokens,
                action_resolver=vocab.resolve,
            )
            status = "candidate" if ingest.record is not None else "quarantined"
            entry = {
                "item_id": item.item_id,
                "task": item.task,
                "bucket": item.bucket,
                "source": item.entry.source,
                "source_id": item.entry.source_id,
                "status": status,
                "errors": list(ingest.errors),
                "warnings": sorted(set(ingest.warnings)),
                "target": ingest.target if ingest.record is not None else None,
                "charge_index": charge_index,
                "attempts": attempts,
                **response.as_metadata(),
            }
            results.append(entry)
            if ingest.record is not None:
                candidates_handle.write(R.canonical_json(ingest.record) + "\n")
                print(f"  [ok]    {item.item_id}: {R.canonical_json(ingest.record['target'])[:90]}")
            else:
                quarantine_handle.write(
                    json.dumps(
                        {
                            "item_id": item.item_id,
                            "task": item.task,
                            "bucket": item.bucket,
                            "reason": ";".join(ingest.errors),
                            "raw_response": response.text,
                            "source": item.entry.source,
                            "source_id": item.entry.source_id,
                            "prompt_sha256": item.entry.sha256,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(f"  [quar]  {item.item_id}: {';'.join(ingest.errors)}")

    summary = {
        "pilot_id": config["pilot_id"],
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config["model"],
        "requests_sent": len(results),
        "authorized_total_requests": int(config["authorized_total_requests"]),
        "used_requests_after_run": ledger.used,
        "status_counts": dict(Counter(entry["status"] for entry in results)),
        "task_status": {task: dict(Counter(e["status"] for e in results if e["task"] == task)) for task in sorted({e["task"] for e in results})},
        "error_counts": dict(Counter(error for entry in results for error in entry.get("errors", []))),
        "buckets": dict(Counter(entry["bucket"] for entry in results)),
        "completion_tokens": sum(int(entry.get("completion_tokens", 0)) for entry in results),
        "prompt_tokens": sum(int(entry.get("prompt_tokens", 0)) for entry in results),
        "finish_reasons": dict(Counter(entry.get("finish_reason", "") for entry in results)),
        "results": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Persist into the ignored processed build as well, keeping provenance.
    build_dir = ROOT / "data" / "processed" / args.build_id / "teacher"
    build_dir.mkdir(parents=True, exist_ok=True)
    if candidates_path.exists():
        records = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        R.write_jsonl(build_dir / f"candidates-{run_id}.jsonl", records)
    for name in ("candidates.jsonl", "quarantine.jsonl", "plan.json", "ledger.jsonl", "summary.json"):
        source = run_dir / name
        if source.exists() and name != "candidates.jsonl":
            (build_dir / f"{source.stem}-{run_id}{source.suffix}").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
