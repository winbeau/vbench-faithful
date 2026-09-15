#!/usr/bin/env python3
"""Validate supplementary table and manifest schemas without running models."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DIMENSIONS = {"dynamic_degree", "subject_consistency", "human_action", "spatial_relationship", "scene", "multiple_objects", "overall_consistency", "motion_smoothness"}
MAIN_FIELDS = ["dimension", "variant", "vbench_score", "human_preference"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    checks: dict[str, object] = {}
    if args.main_csv:
        with args.main_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if list(rows[0]) != MAIN_FIELDS if rows else True:
            raise ValueError(f"main CSV columns must be {MAIN_FIELDS}")
        if len(rows) != 16 or {row["dimension"] for row in rows} != DIMENSIONS or {row["variant"] for row in rows} != {"official", "repair"}:
            raise ValueError("main CSV must contain exactly two variants for each of eight dimensions")
        checks["main_csv_rows"] = len(rows)
    if args.tex:
        text = args.tex.read_text(encoding="utf-8")
        body = text.split(r"\midrule", 1)[-1].split(r"\bottomrule", 1)[0]
        rows = [line for line in body.splitlines() if "&" in line and line.rstrip().endswith(r"\\")]
        if len(rows) != 8 or r"\begin{tabular}{lccc}" not in text or r"\toprule" not in text or r"\bottomrule" not in text:
            raise ValueError("TeX must be an eight-row lccc booktabs table")
        checks["tex_rows"] = len(rows)
    if args.manifest:
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        required = {"schema_version", "dimension", "base_id", "derived_id", "intervention_family", "level", "expected_relation", "input_sha256", "output_sha256", "fps", "frame_count", "duration_s", "code_sha"}
        if any(not required.issubset(row) for row in rows) or len({row["derived_id"] for row in rows}) != len(rows):
            raise ValueError("manifest required fields or derived_id uniqueness failed")
        checks["manifest_rows"] = len(rows)
    print(json.dumps({"status": "VALID", **checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
