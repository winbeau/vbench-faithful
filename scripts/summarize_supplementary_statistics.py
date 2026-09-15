#!/usr/bin/env python3
"""Summarize supplementary intervention scores using base_id clusters.

Input rows are JSONL and may contain ``score`` plus the manifest fields.  The
script reports positive, reverse, and stable families separately, preserving
missing/blocked statuses instead of treating them as zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        mean_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[ordered[position][0]] = mean_rank
        index = end
    return output


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(items, key=lambda row: (float(row.get("level_index", 0)), str(row.get("derived_id", ""))))
    values = [finite(row.get("score")) for row in ordered]
    valid = [(index, value) for index, value in enumerate(values) if value is not None]
    result: dict[str, Any] = {"dimension": items[0].get("dimension"), "base_id": items[0].get("base_id"), "intervention_family": items[0].get("intervention_family"), "expected_relation": items[0].get("expected_relation"), "input_count": len(items), "valid_count": len(valid), "status": "complete" if valid else "blocked_or_missing"}
    if not valid:
        result.update({"macro_mean": None, "relative_range": None, "cv": None, "spearman_level_score": None, "strict_order_rate": None, "absolute_change": None})
        return result
    ys = [value for _, value in valid]
    mean = statistics.fmean(ys)
    result["macro_mean"] = mean
    result["relative_range"] = (max(ys) - min(ys)) / abs(mean) if mean else None
    result["cv"] = statistics.pstdev(ys) / abs(mean) if mean else None
    result["spearman_level_score"] = spearman([float(index) for index, _ in valid], ys)
    result["strict_order_rate"] = (sum(b > a for a, b in zip(ys, ys[1:])) / (len(ys) - 1)) if len(ys) > 1 else None
    result["absolute_change"] = abs(ys[-1] - ys[0]) if len(ys) > 1 else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Manifest-enriched score JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    required = {"dimension", "base_id", "derived_id", "intervention_family", "level", "expected_relation"}
    for line, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {line} missing required fields: {missing}")
    if len({row["derived_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate derived_id")
    if args.validate_only:
        print(json.dumps({"status": "VALID", "records": len(rows), "independent_unit": "base_id"}))
        return 0
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dimension"]), str(row["base_id"]), str(row["intervention_family"]))].append(row)
    summaries = [summarize_group(items) for _, items in sorted(groups.items())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "supplementary_statistics_v1", "independent_unit": "base_id", "records": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in summaries for key in row})
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps({"status": "COMPLETE", "groups": len(summaries), "output": str(args.output), "csv": str(args.csv) if args.csv else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
