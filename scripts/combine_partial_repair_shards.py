#!/usr/bin/env python3
"""Combine completed records from interrupted repair shard groups.

This is used when a dimension run is stopped after some videos complete and is
then resumed with a different GPU layout.  Records are keyed by the frozen
manifest ``video_uid``; each UID must occur exactly once across the input
roots, and the final output is ordered by the manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = [
    "video_uid",
    "video_path",
    "model",
    "dimension",
    "split",
    "prompt_id",
    "group_id",
    "relative_video_path",
    "explicit_target_or_metadata",
    "repair_score",
    "repair_status",
    "error",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def video_path(data_root: Path, relative: str) -> Path:
    path = data_root / relative
    if path.is_file():
        return path
    for suffix in (".mp4", ".gif"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = [row for row in rows(args.manifest) if row["dimension"] == args.dimension]
    if not manifest:
        raise RuntimeError(f"manifest has no rows for {args.dimension}")
    order = {row["video_uid"]: index for index, row in enumerate(manifest)}
    expected = set(order)

    records: dict[str, dict[str, object]] = {}
    sources: dict[str, str] = {}
    files: list[Path] = []
    for root in args.input_root:
        files.extend(sorted(root.glob(f"shard*/{args.dimension}/repair_results.jsonl")))
    if not files:
        raise RuntimeError("no repair_results.jsonl files found")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record["video_uid"])
            if uid in records:
                raise RuntimeError(f"duplicate video_uid {uid} in {sources[uid]} and {path}")
            records[uid] = record
            sources[uid] = str(path)

    actual = set(records)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise RuntimeError(
            f"coverage mismatch expected={len(expected)} actual={len(actual)} "
            f"missing={missing} extra={extra}"
        )

    prediction_rows: list[dict[str, object]] = []
    for manifest_row in sorted(manifest, key=lambda row: order[row["video_uid"]]):
        record = records[manifest_row["video_uid"]]
        path = video_path(args.data_root, manifest_row["relative_video_path"])
        prediction_rows.append(
            {
                "video_uid": manifest_row["video_uid"],
                "video_path": str(path),
                "model": manifest_row["generator"],
                "dimension": manifest_row["dimension"],
                "split": manifest_row["split"],
                "prompt_id": manifest_row["prompt_id"],
                "group_id": manifest_row["group_id"],
                "relative_video_path": manifest_row["relative_video_path"],
                "explicit_target_or_metadata": "null",
                "repair_score": record.get("repair_score"),
                "repair_status": record.get("repair_status", "pending"),
                "error": record.get("failure_or_abstention"),
            }
        )

    output = args.output_root / args.dimension
    output.mkdir(parents=True, exist_ok=True)
    with (output / "repair_results.jsonl").open("w", encoding="utf-8") as handle:
        for manifest_row in sorted(manifest, key=lambda row: order[row["video_uid"]]):
            handle.write(json.dumps(records[manifest_row["video_uid"]], ensure_ascii=False) + "\n")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(prediction_rows)

    statuses = Counter(str(row["repair_status"]) for row in prediction_rows)
    summary = {
        "dimension": args.dimension,
        "total_manifest_n": len(expected),
        "merged_prediction_n": len(prediction_rows),
        "repair_status": dict(sorted(statuses.items())),
        "input_roots": [str(root) for root in args.input_root],
        "input_files": [str(path) for path in files],
    }
    (output / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
