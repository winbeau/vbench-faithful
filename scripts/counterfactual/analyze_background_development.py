"""Background-specific dev analysis, preserving failures and intervention roles."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from background_consistency.algorithms import METHODS
from .analyze_subject_background_auto import cluster_summary
from .analyze_subject_natural import evaluate_population, score
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json, write_jsonl


def checked_rows(directory):
    run = json.loads((directory/"run.json").read_text())
    if not run.get("completed") or sha256_file(directory/"scores.jsonl") != run["scores_sha256"]:
        raise ValueError("incomplete or modified score run")
    rows = read_jsonl(directory/"scores.jsonl")
    if len(rows) != run["video_count"]:
        raise ValueError("run denominator mismatch")
    return rows, run


def region_cases(rows, methods=METHODS):
    cases = []
    for row in rows:
        if row["construction_status"] != "accepted":
            continue
        variants = row["variants"]
        for position in ("full", "start", "middle", "end"):
            for method in methods:
                clean = score(variants.get("clean", {}), method)
                fg = score(variants.get(position+"/subject_corrupt", {}), method)
                bg = score(variants.get(position+"/background_corrupt", {}), method)
                original_clean = score(variants.get("clean", {}), "official")
                original_fg = score(variants.get(position+"/subject_corrupt", {}), "official")
                nuisance = abs(fg-clean) if None not in (clean, fg) else None
                bg_drop = clean-bg if None not in (clean, bg) else None
                improvement = abs(original_fg-original_clean)-nuisance if None not in (original_fg, original_clean, nuisance) else None
                cases.append({"base_id": row["base"]["base_id"], "prompt_id": row["base"]["prompt_id"],
                    "method": method, "position": position, "clean": clean, "subject_blur": fg, "background_blur": bg,
                    "foreground_signed_change": fg-clean if None not in (clean, fg) else None,
                    "foreground_absolute_change": nuisance, "paired_improvement_vs_official": improvement,
                    "background_blur_drop": bg_drop, "background_drop_gt_foreground_abs": float(bg_drop > nuisance) if None not in (bg_drop, nuisance) else None})
    return cases


def summarize_region(rows, methods=METHODS):
    cases = region_cases(rows, methods)
    report = {"cohort": "reused subject-source diagnostic, not official background test",
              "candidate_count": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
              "max_origin_parity_error": max((r.get("parity_absolute_error", 0) for r in rows), default=None),
              "by_position": {}}
    for position in ("full", "start", "middle", "end"):
        report["by_position"][position] = {}
        for method in methods:
            selected = [r for r in cases if r["position"] == position and r["method"] == method]
            groups = {r["base_id"]: r["prompt_id"] for r in selected}
            value = {"construction_accepted": len(selected)}
            for key in ("clean", "subject_blur", "background_blur", "foreground_signed_change", "foreground_absolute_change",
                        "paired_improvement_vs_official", "background_blur_drop", "background_drop_gt_foreground_abs"):
                # Bootstrap only contrast quantities; raw means need no
                # repeated bootstrap for the same summary table.
                values = {r["base_id"]: r[key] for r in selected if r[key] is not None}
                if key in ("foreground_absolute_change", "paired_improvement_vs_official", "background_blur_drop"):
                    value[key] = cluster_summary(values, groups)
                else:
                    value[key] = {"n_bases": len(values), "mean": float(np.mean(list(values.values()))) if values else None}
            report["by_position"][position][method] = value
    return report, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-run", type=Path)
    parser.add_argument("--natural-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = new_output(args.output)
    report = {}
    if args.region_run:
        rows, run = checked_rows(args.region_run)
        report["region"], cases = summarize_region(rows)
        report["region"]["provenance"] = run
        with (output/"region_per_case.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader(); writer.writerows(cases)
    if args.natural_run:
        protocol = json.loads((ROOT/"configs/background-repair/development_protocol_v2.json").read_text())
        rows, runs = [], []
        for directory in sorted(args.natural_run.glob("shard*")):
            if not directory.is_dir():
                continue
            chunk, run = checked_rows(directory)
            if run["protocol_sha256"] != sha256_file(ROOT/"configs/background-repair/development_protocol_v2.json"):
                raise ValueError("natural protocol mismatch")
            rows.extend(chunk); runs.append(run)
        expected = [r for r in read_jsonl(ROOT/"configs/background-repair"/protocol["manifest_file"]) if r["split"] == "dev"]
        videos = {r["video_uid"]: r for r in rows}
        if len(rows) != len(videos) or set(videos) != {r["video_uid"] for r in expected}:
            raise ValueError("natural dev has missing, duplicate or unexpected videos")
        pair_path = ROOT/"data/processed/pairwise_master_split.csv"
        if sha256_file(pair_path) != protocol["pair_labels_sha256"]:
            raise ValueError("background labels changed")
        pairs = [r for r in csv.DictReader(pair_path.open()) if r["dimension"] == "background_consistency" and r["split"] == "dev"]
        if len(pairs) != 1020:
            raise ValueError("expected 1020 background dev preference pairs")
        report["natural_dev"] = evaluate_population(pairs, videos, METHODS, {m: 0 for m in METHODS})
        report["natural_dev"].update(video_count=len(rows), status_counts=dict(Counter(r["status"] for r in rows)),
            max_origin_parity_error=max((r.get("parity_absolute_error", 0) for r in rows), default=None),
            source_runs=runs, media_duration_seconds=sum(r.get("media_duration_seconds", 0) for r in rows),
            raw_means={m: float(np.mean([v for r in rows if (v := score(r, m)) is not None])) for m in METHODS})
        write_jsonl(output/"natural_dev_scores.jsonl", rows)
    write_json(output/"statistics.json", report)
    print(json.dumps({"output": str(output), "sections": list(report)}))


if __name__ == "__main__":
    main()
