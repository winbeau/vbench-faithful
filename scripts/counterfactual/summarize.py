"""Aggregate per-dimension CPA results into the paper's main table.

Reads the `<dimension>__cpa.json` files written by `run_dimension.py`, produces
`table2.csv` and `SUMMARY.md`, and states plainly which numbers are degenerate:
for a same-rank (invariance) family a tie-margin CPA can always be pushed to 1.0
by widening the margin, so those rows are reported with the within-base
coefficient of variation alongside and are not used to claim a win on CPA alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .common import write_json
from .cpa import BUDGET
from .run_dimension import FAMILY_DESCRIPTION, invariance_stats
from .common import read_jsonl

ORDER = (
    "dynamics_degree",
    "subject_consistency",
    "human_action",
    "spatial_relationship",
    "scene",
    "multiplt_object",
    "motion_smoothness",
)


def load_rows(manifest: Path, dimension: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(manifest) if row["dimension"] == dimension]


def load_scores(scores_dir: Path, dimension: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for backend in ("official", "repair"):
        path = scores_dir / f"{dimension}__{backend}.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            if row.get("status") == "succeeded" and row.get("score") is not None:
                out[f"{row['derived_id']}::{backend}"] = float(row["score"])
    return out


def collect(scores_dir: Path, manifest: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for dimension in ORDER:
        cpa_path = scores_dir / f"{dimension}__cpa.json"
        if not cpa_path.is_file():
            entries.append({"dimension": dimension, "status": "missing"})
            continue
        cpa = json.loads(cpa_path.read_text(encoding="utf-8"))
        rows = load_rows(manifest, dimension)
        family = rows[0]["family"] if rows else "?"
        invariance = len({row["expected_rank"] for row in rows}) == 1
        scores = load_scores(scores_dir, dimension)
        entry: dict[str, Any] = {
            "dimension": dimension,
            "family": family,
            "invariance": invariance,
            "bases": len({row["base_id"] for row in rows}),
            "clips": len(rows),
            "status": "ok",
            "margin": cpa.get("official", {}).get("dev_margin"),
        }
        for backend in ("official", "repair"):
            block = cpa.get(backend, {})
            stats = block.get("test_tie_aware", {}).get(family, {})
            entry[f"{backend}_cpa"] = stats.get("cpa")
            entry[f"{backend}_ci_low"] = stats.get("ci_low")
            entry[f"{backend}_ci_high"] = stats.get("ci_high")
            entry[f"{backend}_pairs"] = stats.get("n_pairs")
            backend_scores = {
                row["derived_id"]: scores.get(f"{row['derived_id']}::{backend}")
                for row in rows
                if row["split"] == "test"
            }
            test_rows = [row for row in rows if row["split"] == "test"]
            entry[f"{backend}_coverage"] = sum(
                1 for row in test_rows if backend_scores.get(row["derived_id"]) is not None
            )
            entry[f"{backend}_test_clips"] = len(test_rows)
            if invariance:
                entry[f"{backend}_cv"] = invariance_stats(test_rows, backend_scores)["mean_cv"]
        if entry.get("official_cpa") is not None and entry.get("repair_cpa") is not None:
            entry["delta"] = round(entry["repair_cpa"] - entry["official_cpa"], 4)
        paired = cpa.get("paired", {}).get("tie_aware", {}) or {}
        entry["delta_paired"] = paired.get("delta")
        entry["delta_ci_low"] = paired.get("ci_low")
        entry["delta_ci_high"] = paired.get("ci_high")
        entries.append(entry)
    return entries


def render_summary(entries: list[dict[str, Any]], code_sha: str) -> str:
    lines = [
        "# Table 2 — Counterfactual Pair Accuracy (VBench-CF)",
        "",
        "Official VBench 1.0 versus the repaired/Audit metric on metamorphic",
        "counterfactuals. CPA is over all ordered level pairs within a family; the",
        "interval is a 95% cluster bootstrap over `base_id`; `delta` is Repair minus",
        "Official on the test split.",
        "",
        f"Scoring code SHA: `{code_sha}`.",
        "",
        "| dimension | family | bases | clips (test) | Official CPA | Repair CPA | delta | delta 95% CI (paired) |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        if entry.get("status") == "missing":
            lines.append(f"| {entry['dimension']} | — | — | — | — | — | — | — |")
            continue
        off = entry.get("official_cpa")
        rep = entry.get("repair_cpa")
        ci_low = entry.get("delta_ci_low")
        ci_high = entry.get("delta_ci_high")
        ci = "—" if ci_low is None else f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        lines.append(
            f"| {entry['dimension']} | `{entry['family']}` | {entry['bases']} | "
            f"{entry.get('official_test_clips', 0)} | "
            f"{off:.4f} | {rep:.4f} | {entry.get('delta', float('nan')):+.4f} | {ci} |"
            if off is not None and rep is not None
            else f"| {entry['dimension']} | `{entry['family']}` | {entry['bases']} | — | — | — | — | — |"
        )

    invariance_rows = [entry for entry in entries if entry.get("invariance") and entry.get("status") == "ok"]
    if invariance_rows:
        lines += [
            "",
            "## Invariance families — dispersion, not CPA",
            "",
            "For these families every level shares one expected rank, so a tie-margin",
            "CPA is degenerate: widening the dev margin until every difference counts",
            "as a tie yields 1.0 regardless of how unstable the metric is. The",
            "within-base coefficient of variation is the meaningful number, and a",
            "Repair CV of 0 with a large Official CV is the actual result.",
            "",
            "| dimension | Official CV | Repair CV | Official CPA | Repair CPA |",
            "|---|---:|---:|---:|---:|",
        ]
        for entry in invariance_rows:
            ocv = entry.get("official_cv")
            rcv = entry.get("repair_cv")
            lines.append(
                f"| {entry['dimension']} | {'—' if ocv is None else f'{ocv:.4f}'} | "
                f"{'—' if rcv is None else f'{rcv:.4f}'} | "
                f"{entry.get('official_cpa', float('nan')):.4f} | {entry.get('repair_cpa', float('nan')):.4f} |"
            )

    lines += [
        "",
        "## Coverage",
        "",
        "| dimension | Official scored / test | Repair scored / test |",
        "|---|---:|---:|",
    ]
    for entry in entries:
        if entry.get("status") == "missing":
            continue
        lines.append(
            f"| {entry['dimension']} | {entry.get('official_coverage', 0)} / {entry.get('official_test_clips', 0)} | "
            f"{entry.get('repair_coverage', 0)} / {entry.get('repair_test_clips', 0)} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- These are **first-run measurements from the current metric packages**. Model,",
        "  CUDA and weight parity against the frozen E0 baselines has **not** been",
        "  verified, so absolute values are not reproduced official numbers.",
        "- Clips whose backend raised are recorded as `failed` and excluded from the CPA",
        "  denominator; they are never scored as zero, so coverage is reported separately.",
        "- A same-rank family's CPA must not be quoted without its dispersion statistics.",
        "- Overall Consistency is out of scope for this round.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = collect(args.scores, args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    code_sha = next(
        (row.get("code_sha") for row in read_jsonl(args.manifest) if row.get("code_sha")), "unknown"
    )

    fields = [
        "dimension", "family", "bases", "clips", "invariance",
        "official_cpa", "official_ci_low", "official_ci_high",
        "repair_cpa", "repair_ci_low", "repair_ci_high", "delta",
        "delta_paired", "delta_ci_low", "delta_ci_high",
        "official_cv", "repair_cv", "official_coverage", "repair_coverage",
        "official_test_clips", "repair_test_clips", "margin",
    ]
    with (args.output / "table2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry.get(key) for key in fields})

    write_json(args.output / "table2.json", {"code_sha": code_sha, "dimensions": entries})
    (args.output / "SUMMARY.md").write_text(render_summary(entries, code_sha), encoding="utf-8")
    print(render_summary(entries, code_sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
