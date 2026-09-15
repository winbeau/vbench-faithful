#!/usr/bin/env python3
"""Merge isolated repair shards after a four-GPU dimension run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"dynamic_degree": "dynamics_degree"}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise RuntimeError(f"cannot infer fields for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/e0_scoring_manifest.csv")
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    official_dimension = ALIASES.get(args.dimension, args.dimension)
    manifest = [row for row in csv_rows(args.manifest) if row["dimension"] == official_dimension]
    order = {row["video_uid"]: index for index, row in enumerate(manifest)}
    expected = set(order)

    predictions: list[dict[str, str]] = []
    json_records: list[dict[str, object]] = []
    paired: list[dict[str, str]] = []
    for index in range(args.num_shards):
        shard = args.shards_root / f"shard{index}" / args.dimension
        predictions.extend(csv_rows(shard / "predictions.csv"))
        paired_path = shard / "paired_results.csv"
        if paired_path.is_file():
            paired.extend(csv_rows(paired_path))
        with (shard / "repair_results.jsonl").open(encoding="utf-8") as handle:
            json_records.extend(json.loads(line) for line in handle if line.strip())

    ids = [row["video_uid"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate video_uid across repair shards")
    if set(ids) != expected:
        missing = sorted(expected - set(ids))[:5]
        extra = sorted(set(ids) - expected)[:5]
        raise RuntimeError(f"shard coverage mismatch: expected={len(expected)} actual={len(ids)} missing={missing} extra={extra}")

    predictions.sort(key=lambda row: order[row["video_uid"]])
    paired.sort(key=lambda row: order[row["video_uid"]])
    latest = {str(row["video_uid"]): row for row in json_records}
    if set(latest) != expected:
        raise RuntimeError("repair JSONL coverage does not match manifest")

    output = args.output_root / args.dimension
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", predictions)
    if paired:
        write_csv(output / "paired_results.csv", paired)
        disagreements = [row for row in paired if row.get("comparison_status") == "comparable" and row.get("agreement_if_defined") == "False"]
        write_csv(output / "disagreements.csv", disagreements, list(paired[0]))
    with (output / "repair_results.jsonl").open("w", encoding="utf-8") as handle:
        for uid in sorted(expected, key=order.get):
            handle.write(json.dumps(latest[uid], ensure_ascii=False) + "\n")

    statuses = Counter(row.get("repair_status", "") for row in predictions)
    summary = {
        "dimension": args.dimension,
        "total_manifest_n": len(expected),
        "merged_prediction_n": len(predictions),
        "repair_status": dict(sorted(statuses.items())),
        "shards": args.num_shards,
    }
    (output / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
