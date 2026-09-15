#!/usr/bin/env python3
"""Convert an isolated official-backend prediction CSV to audit score schema."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["video_uid", "dimension", "split", "prompt_id", "group_id", "generator", "relative_video_path", "score", "status", "elapsed_s", "error", "official_video_boolean"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in source:
            writer.writerow({
                "video_uid": row["video_uid"], "dimension": args.dimension,
                "split": row["split"], "prompt_id": row["prompt_id"],
                "group_id": row["group_id"], "generator": row["model"],
                "relative_video_path": row["relative_video_path"],
                "score": row["repair_score"] if row.get("repair_score") not in (None, "") else "",
                "status": "success" if row.get("repair_status") == "succeeded_scalar" else row.get("repair_status", "failed"),
                "elapsed_s": "", "error": row.get("error", ""), "official_video_boolean": "",
            })
    print(f"wrote {len(source)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
