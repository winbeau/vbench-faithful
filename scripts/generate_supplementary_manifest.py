#!/usr/bin/env python3
"""Create a deterministic eight-dimension supplementary intervention manifest.

This stage does not fabricate videos or scores.  It records the planned
counterfactuals and can optionally bind rows to an existing base-video JSONL.
The output is JSONL so individual rows can be resumed on remote workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = {
    "dynamic_degree": [
        ("camera_motion", ["decrease", "baseline", "increase"], "+1"),
        ("fps_resampling_stability", ["low", "native", "high"], "0"),
    ],
    "subject_consistency": [
        ("subject_corruption", ["decrease", "baseline", "increase"], "+1"),
        ("position_exchange", ["first", "middle", "last"], "0"),
    ],
    "human_action": [
        ("action_coverage", ["decrease", "baseline", "increase"], "+1"),
        ("target_relevance", ["irrelevant", "related", "correct"], "+1"),
    ],
    "spatial_relationship": [
        ("flip_recovery", ["reversed", "baseline", "recovered"], "+1"),
        ("role_swap_stability", ["swapped", "baseline", "reciprocal"], "0"),
    ],
    "scene": [
        ("environment_coverage", ["decrease", "baseline", "increase"], "+1"),
        ("local_cue_stability", ["isolated", "baseline", "isolated_repeated"], "0"),
    ],
    "multiple_objects": [
        ("target_occlusion", ["decrease", "baseline", "increase"], "+1"),
        ("temporal_conjunction", ["separated", "baseline", "coincident"], "0"),
    ],
    "overall_consistency": [
        ("condition_replacement", ["decrease", "baseline", "increase"], "+1"),
        ("condition_recovery", ["broken", "baseline", "recovered"], "+1"),
    ],
    "motion_smoothness": [
        ("temporal_perturbation", ["decrease", "baseline", "increase"], "+1"),
        ("tail_jerk_stability", ["none", "baseline", "tail_only"], "0"),
    ],
}


def code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, help="Optional JSONL with base_id, split, prompt, and path fields")
    parser.add_argument("--output", type=Path, default=ROOT / "output/supplementary_20260914/supplementary_manifest.jsonl")
    parser.add_argument("--bases-per-dimension", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.bases_per_dimension < 1:
        parser.error("--bases-per-dimension must be positive")
    supplied = read_jsonl(args.bases) if args.bases else []
    by_dim: dict[str, list[dict[str, Any]]] = {dim: [] for dim in DIMENSIONS}
    for row in supplied:
        dim = str(row.get("dimension", ""))
        if dim in by_dim:
            by_dim[dim].append(row)
    records: list[dict[str, Any]] = []
    sha = code_sha()
    for dim, families in DIMENSIONS.items():
        bases = sorted(by_dim[dim], key=lambda row: str(row.get("base_id", "")))[: args.bases_per_dimension]
        if not bases:
            bases = [{"base_id": f"blocked_{dim}_base_{i:03d}", "split": "blocked", "prompt": None, "path": None, "status": "blocked_missing_base"} for i in range(args.bases_per_dimension)]
        for base in bases:
            base_id = str(base.get("base_id"))
            for family, levels, direction in families:
                for index, level in enumerate(levels):
                    derived_id = "cf_" + stable_id(dim, base_id, family, level, str(args.seed))
                    expected = "stable" if direction == "0" else ("higher" if index > 1 else "lower" if index < 1 else "baseline")
                    input_path = base.get("path") or base.get("video_path")
                    records.append({
                        "schema_version": "supplementary_manifest_v1",
                        "dimension": dim,
                        "base_id": base_id,
                        "derived_id": derived_id,
                        "split": str(base.get("split", "blocked")),
                        "prompt": base.get("prompt"),
                        "input_path": input_path,
                        "output_path": None,
                        "intervention_family": family,
                        "level": level,
                        "level_index": index,
                        "expected_relation": expected,
                        "direction": direction,
                        "transformation_parameters": {"seed": args.seed, "level": level, "family": family},
                        "input_sha256": None,
                        "output_sha256": None,
                        "fps": None,
                        "frame_count": None,
                        "duration_s": None,
                        "code_sha": sha,
                        "status": str(base.get("status", "planned")),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    expected_fields = {"base_id", "derived_id", "dimension", "intervention_family", "level", "expected_relation", "input_sha256", "output_sha256", "fps", "frame_count", "duration_s", "code_sha"}
    if any(not expected_fields.issubset(row) for row in records) or len({row["derived_id"] for row in records}) != len(records):
        raise RuntimeError("supplementary manifest schema or ID validation failed")
    print(json.dumps({"status": "COMPLETE", "records": len(records), "dimensions": len(DIMENSIONS), "output": str(args.output), "base_status_counts": {status: sum(row["status"] == status for row in records) for status in sorted({row["status"] for row in records})}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
