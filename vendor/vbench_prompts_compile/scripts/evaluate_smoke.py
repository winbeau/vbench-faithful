#!/usr/bin/env python3
"""Evaluate a smoke checkpoint: adapted-vs-base predictions plus honest coverage.

Always reports the full denominator: parse failures and missing predictions are
counted, never filtered out of the metrics. Nothing here is a final benchmark.

Usage::

    uv run --no-sync --extra train python scripts/evaluate_smoke.py \
        --task spatial --data data/smoke/local-0001/spatial.jsonl \
        --base-model models/tiny-qwen-smoke --adapter spatial=runs/smoke/cpu-tiny-spatial \
        --output-dir runs/smoke/eval-cpu-tiny-spatial --compare-base
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import metrics as M  # noqa: E402
from vbench_prompts_compile import training as T  # noqa: E402
from vbench_prompts_compile.inference import AdapterRouter, Prediction, SceneVerifier  # noqa: E402
from vbench_prompts_compile.provenance import environment, git_state, sha256_file  # noqa: E402
from vbench_prompts_compile.sources import load_k400  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["spatial", "action", "objects", "scene"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default=None, help="frozen backbone for the parse model")
    parser.add_argument("--adapter", action="append", default=[], help="task=path (repeatable; parse tasks only)")
    parser.add_argument("--scene-model", default=None, help="path to the independent Scene checkpoint or merged model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compare-base", action="store_true", help="also score the un-finetuned backbone")
    parser.add_argument("--canonicalize-actions", action="store_true", help="map action phrases through the frozen K400 vocabulary before scoring")
    parser.add_argument('--action-interface', choices=['legacy-model', 'repair-v2', 'repair-v2.1'], default='legacy-model',
                        help='keep native model evaluation by default; explicitly select the repaired system')
    parser.add_argument("--canonicalize-entities", action="store_true", help="normalise entity names before scoring")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--scene-stop-at-newline", action="store_true", help="Scene only: stop decoding at the first newline (label-only contract)")
    return parser.parse_args(argv)


def load_task_records(path: Path, task: str) -> list[dict[str, Any]]:
    resolver = load_k400().resolve if task == "action" else None
    return T.load_records(path, task, action_resolver=resolver)


def canonicalize_gold(records: list[dict[str, Any]], *, actions: bool, entities: bool) -> int:
    """Apply the same deterministic normalisation to the gold targets.

    Scoring a canonicalised prediction against a raw gold target (or the reverse) is a
    harness artefact, so both sides go through the same frozen function when the
    canonicalisation flag is on.
    """
    from vbench_prompts_compile.inference import canonicalize_actions, canonicalize_entities

    resolver = load_k400().resolve if actions else None
    changed = 0
    for record in records:
        target = record["target"]
        if actions and record["task"] == "action":
            target, count = canonicalize_actions(target, resolver)
            changed += count
        if entities and record["task"] == "objects":
            target, count = canonicalize_entities(target)
            changed += count
        record["target"] = target
    return changed


def score(task: str, records: Sequence[dict[str, Any]], predict: Callable[[dict[str, Any]], Prediction]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        prediction = predict(record)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "task": task,
                "gold": record["target"],
                "pred": prediction.value,
                "raw": prediction.raw,
                "errors": list(prediction.errors),
            }
        )
    return rows, M.evaluate_task(task, rows)


def write_predictions(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    records = load_task_records(data_path, args.task)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("no records to evaluate", file=sys.stderr)
        return 2
    gold_changed = canonicalize_gold(records, actions=args.canonicalize_actions, entities=args.canonicalize_entities)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    metrics: dict[str, Any] = {}
    if args.task == "scene":
        if not args.scene_model:
            print("--scene-model is required for the scene task", file=sys.stderr)
            return 2
        scene_path = Path(args.scene_model)
        if not scene_path.is_absolute():
            scene_path = ROOT / scene_path
        adapter_file = scene_path / "adapter_config.json"
        if adapter_file.exists():
            if not args.base_model:
                print("scene adapter detected; --base-model is required to attach it", file=sys.stderr)
                return 2
            verifier = SceneVerifier(base_model_path=args.base_model, adapter_path=str(scene_path), local_files_only=not args.allow_download)
        else:
            verifier = SceneVerifier(model_path=str(scene_path), local_files_only=not args.allow_download)
        rows, summary = score(
            args.task,
            records,
            lambda record: verifier.predict(
                record["input"]["prompt"],
                record["input"]["caption"],
                max_new_tokens=args.max_new_tokens,
                stop_at_newline=args.scene_stop_at_newline,
            ),
        )
        metrics["finetuned"] = summary
    else:
        adapters: dict[str, str] = {}
        for item in args.adapter:
            if "=" not in item:
                print(f"--adapter expects task=path, got {item!r}", file=sys.stderr)
                return 2
            task, path = item.split("=", 1)
            adapters[task] = path
        if args.task not in adapters:
            print(f"no adapter supplied for task {args.task!r}", file=sys.stderr)
            return 2
        if not args.base_model:
            print("--base-model is required for parse tasks", file=sys.stderr)
            return 2
        router = AdapterRouter(base_model_path=args.base_model, adapters=adapters, local_files_only=not args.allow_download)
        rows, summary = score(
            args.task,
            records,
            lambda record: router.predict(
                args.task,
                router.user_text(record),
                max_new_tokens=args.max_new_tokens,
                canonicalize_action_output=args.canonicalize_actions,
                canonicalize_entity_output=args.canonicalize_entities,
                action_interface=args.action_interface, action_prompt=record['input']['prompt'],
            ),
        )
        metrics["finetuned"] = summary
        metrics["adapter_names"] = router.adapter_names()
        if args.compare_base:
            base_rows, base_summary = score(args.task, records, lambda record: router.predict_base(args.task, router.user_text(record), max_new_tokens=args.max_new_tokens))
            metrics["base"] = base_summary
            write_predictions(output_dir / "predictions-base.jsonl", base_rows)

    write_predictions(output_dir / "predictions.jsonl", rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": args.task,
        "data": {"path": str(data_path), "sha256": sha256_file(data_path), "samples": len(records)},
        "git": git_state(),
        "environment": environment(),
        "max_new_tokens": args.max_new_tokens,
        "greedy_decoding": True,
        "scene_stop_at_newline": args.scene_stop_at_newline,
        "canonicalize_actions": args.canonicalize_actions,
        "canonicalize_entities": args.canonicalize_entities,
        "action_interface": args.action_interface,
        "gold_canonicalisation_changes": gold_changed,
        "metrics": metrics,
        "caveat": "engineering smoke on fixtures/candidates; not a generalization or benchmark result",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
