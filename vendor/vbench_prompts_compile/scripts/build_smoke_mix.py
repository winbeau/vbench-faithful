#!/usr/bin/env python3
"""Assemble mixed smoke sets (engineering + weak + teacher candidates) per task.

This is the dataset used by the T1/T2 smoke runs. It is deliberately *not* a
research split: engineering fixtures, weak synthetic templates and unreviewed
teacher candidates are mixed for pipeline validation, and the manifest records
each source's share plus the length-bucket coverage so no one can mistake the
mix for gold data.

Usage::

    uv run --no-sync python scripts/build_smoke_mix.py --mix-id mix-0001
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vbench_prompts_compile import records as R  # noqa: E402

DEFAULT_SOURCES = {
    "fixtures": "data/smoke/local-0001",
    "processed": "data/processed/local-0001",
    "teacher": "data/processed/local-0001/teacher",
}
LIMITS = {
    "spatial": {"engineering_only": 12, "synthetic_weak": 120, "teacher_candidate_unreviewed": 5},
    "action": {"engineering_only": 12, "synthetic_weak": 120, "teacher_candidate_unreviewed": 1},
    "objects": {"engineering_only": 12, "synthetic_weak": 60, "teacher_candidate_unreviewed": 4},
    "scene": {"engineering_only": 90, "teacher_candidate_unreviewed": 5},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_all(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(R.read_jsonl(path))
    return rows


def collect(root: Path, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += read_all(sorted((root / DEFAULT_SOURCES["fixtures"]).glob(f"{task}.jsonl")))
    rows += read_all([root / DEFAULT_SOURCES["processed"] / "candidates" / f"{task}.jsonl"])
    rows += read_all(sorted((root / DEFAULT_SOURCES["teacher"]).glob("candidates-*.jsonl")))
    rows = [row for row in rows if row.get("task") == task]
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(row["sample_id"], row)
    return list(deduped.values())


def select(rows: Sequence[dict[str, Any]], limits: dict[str, int], seed: int) -> list[dict[str, Any]]:
    by_quality: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_quality.setdefault(row["quality"], []).append(row)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for quality, limit in limits.items():
        candidates = sorted(by_quality.get(quality, []), key=lambda row: row["sample_id"])
        rng.shuffle(candidates)
        chosen.extend(candidates[:limit])
    chosen.sort(key=lambda row: row["sample_id"])
    return chosen


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mix-id", default="mix-0001")
    parser.add_argument("--heldout-limit", type=int, default=16, help="size of the unseen-by-this-run diagnostic slice")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    out_dir = root / "data" / "smoke" / args.mix_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "mix_id": args.mix_id,
        "purpose": "T1/T2 engineering smoke; mixed quality, never a research split",
        "seed": args.seed,
        "tasks": {},
    }
    for task, limits in LIMITS.items():
        rows = select(collect(root, task), limits, args.seed)
        if not rows:
            raise SystemExit(f"no records for task {task}")
        path = out_dir / f"{task}.jsonl"
        R.write_jsonl(path, rows)
        buckets = Counter(row.get("meta", {}).get("length_bucket", "unknown") for row in rows)
        manifest["tasks"][task] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "count": len(rows),
            "quality_mix": dict(Counter(row["quality"] for row in rows)),
            "source_mix": dict(Counter(row["source"] for row in rows)),
            "length_buckets": dict(sorted(buckets.items())),
            "groups": len({row["group_id"] for row in rows}),
        }
    # Diagnostic slice: same-source candidates that this smoke run never trained on.
    # Still weak/candidate data, still not a generalization claim.
    for task in LIMITS:
        mix_ids = {row["sample_id"] for row in R.read_jsonl(out_dir / f"{task}.jsonl")}
        pool = collect(root, task)
        heldout = sorted((row for row in pool if row["sample_id"] not in mix_ids), key=lambda row: row["sample_id"])[: args.heldout_limit]
        if heldout:
            path = out_dir / f"heldout-{task}.jsonl"
            R.write_jsonl(path, heldout)
            manifest["tasks"][task]["heldout"] = {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "count": len(heldout),
                "note": "same-source weak/candidate records excluded from this mix; diagnostic only",
            }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
