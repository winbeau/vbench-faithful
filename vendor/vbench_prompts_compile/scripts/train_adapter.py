#!/usr/bin/env python3
"""Train one parse adapter (spatial/action/objects) with the pinned TRL+PEFT stack.

The entry point is intentionally thin: configuration, provenance and refusal
rules live here; optimisation is ``trl.SFTTrainer`` with ``peft.LoraConfig``.

Safety: no implicit download (``local_files_only`` by default), no truncation of
over-budget samples, no adapter reuse across tasks, and a run manifest with code,
data and environment identity is written before training starts.

Usage::

    uv run --no-sync --extra train python scripts/train_adapter.py --config configs/smoke/spatial-cpu-tiny.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import training as T  # noqa: E402
from vbench_prompts_compile.inference import PARSE_TASKS  # noqa: E402
from vbench_prompts_compile.provenance import environment, git_state, sha256_file  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None, help="override model_name_or_path (relative paths resolve against the repo root)")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true", help="explicitly permit downloading the pinned base model")
    parser.add_argument("--dry-run", action="store_true", help="validate config/data/environment and stop before training")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = T.TrainingConfig.from_json(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    if config.task not in PARSE_TASKS:
        print(f"{config.task!r} is not a parse task; use scripts/train_scene.py", file=sys.stderr)
        return 2
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.data:
        config.train_jsonl = args.data
    if args.model:
        config.model_name_or_path = args.model
    if args.model_revision:
        config.model_revision = args.model_revision
    if args.max_length is not None:
        config.max_length = args.max_length
    model_path = Path(config.model_name_or_path)
    if not model_path.is_absolute() and not model_path.exists() and (ROOT / model_path).exists():
        config.model_name_or_path = str(ROOT / model_path)
    if args.allow_download:
        config.allow_download = True
        config.local_files_only = False
    config.validate()

    if config.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    data_path = Path(config.train_jsonl)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        print(f"training data not found: {data_path}", file=sys.stderr)
        return 2
    if not config.allow_download and not Path(config.model_name_or_path).exists():
        print(
            f"base model path {config.model_name_or_path!r} does not exist; "
            "provide local weights or pass --allow-download explicitly",
            file=sys.stderr,
        )
        return 2

    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    T.copy_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config, run_dir)
    manifest = {
        "entry": "scripts/train_adapter.py",
        "task": config.task,
        "git": git_state(),
        "environment": environment(),
        "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
        "lock_sha256": sha256_file(ROOT / "uv.lock") if (ROOT / "uv.lock").exists() else None,
        "config": config.as_dict(),
    }
    T.write_run_manifest(run_dir, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("environment",)}, ensure_ascii=False)[:400])
    if args.dry_run:
        records = T.load_records(data_path, config.task)
        tokenizer = T.load_tokenizer(config)
        examples = T.render_examples(records)
        print(json.dumps({"dry_run": True, "records": len(records), "length_report": T.length_report(tokenizer, examples, max_length=config.max_length)}, ensure_ascii=False, indent=2))
        return 0

    summary = T.train(config)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config",)}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
