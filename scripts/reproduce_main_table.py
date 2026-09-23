#!/usr/bin/env python3
"""Recompute the frozen eight-dimension table without private machine paths or GPUs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

DIMENSIONS = ("scene", "human_action", "object_class", "subject_consistency",
              "background_consistency", "spatial_relationship", "multiple_objects", "color")
TASKS = {"scene": "scene", "human_action": "action", "spatial_relationship": "spatial",
         "multiple_objects": "objects"}
CELLS = ("origin_base", "origin_cf", "repair_base", "repair_cf")


def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def scalar(value):
    return {"score": value.get("score") if value.get("status") == "succeeded" else None,
            "status": value.get("status", "missing")}


def summarize(rows):
    unique(rows, "sample_id")
    for row in rows:
        for key in CELLS:
            value = row["scores"][key]["score"]
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid score: {row['sample_id']} / {key}")
    primary = [row for row in rows if row["primary"]]
    common = [row for row in primary if all(row["scores"][key]["score"] is not None for key in CELLS)]
    return {"planned_primary": len(primary), "complete_pairs": len(common),
            **{key: statistics.mean(r["scores"][key]["score"] for r in common) if common else None
               for key in CELLS}}


def replay_semantic(bundle, code, labels):
    sys.path[:0] = [str(code / "scripts"), str(code / "src")]
    from score_matrix import score_one, native_entity_codec
    from evaluate_spatial_repair import reference_geometry
    from cache_matrix_transforms import evidence_id
    from vbench_prompts_compile.sources import load_k400
    from vbench_prompts_compile.experiments import text_key

    codec = native_entity_codec(json.loads((bundle / "provenance/VBench_full_info.json").read_text()))
    vocabulary = load_k400(labels)
    result = {}
    for dimension, task in TASKS.items():
        folder = bundle / "dimensions" / dimension
        originals = unique(readl(folder / "evidence/original.jsonl"), "relative_path")
        transforms = unique(readl(folder / "evidence/transforms.jsonl"), "evidence_id") if task in {"spatial", "objects"} else {}
        predictions = {r.get("request_id", text_key(r["task"], r["prompt"], r.get("caption"))): r.get("target")
                       for r in readl(folder / "evidence/semantic_predictions.jsonl")}
        rows = []
        for pair in readl(folder / "pairs.jsonl"):
            variant = pair["source_matrix_row"]
            base = {**variant, "prompt": variant["original_prompt"], "official_target": variant["base_official_target"],
                    "transform": "identity", "evidence_kind": "original"}
            original = originals[variant["relative_path"]]
            transformed = original if task in {"scene", "action"} else transforms[evidence_id(variant)]
            row = {"sample_id": pair["sample_id"], "dimension": dimension, "primary": True, "scores": {}}
            for backend, scheme in [("origin", "Origin"), ("repair", "Repair-model")]:
                for side, item, evidence in [("base", base, original), ("cf", variant, transformed)]:
                    score = score_one(item, scheme, evidence, predictions, vocabulary, codec)
                    valid = evidence.get("status") == "ok" and score.get("status") == "ok" and score.get("missing", 0) == 0
                    row["scores"][f"{backend}_{side}"] = {
                        "score": score["score"] if valid else None,
                        "status": score["status"] if valid else "incomplete_" + score["status"],
                        "raw_partial_diagnostic": score["score"] if not valid else None}
            if task == "spatial":
                geometry = [reference_geometry(base["official_target"], frame) for frame in original["frame_detections"]]
                row["primary"] = sum(g["forward"] for g in geometry) > 0 and sum(g["reverse"] for g in geometry) == 0
            rows.append(row)
        result[dimension] = rows
    return result


def replay_visual_scores(bundle, dimension):
    folder = bundle / "dimensions" / dimension
    rows = []
    if dimension in {"object_class", "color"}:
        maps = {backend: unique(json.loads((folder / "historical-scores" / backend / "results.json").read_text()), "query_uid")
                for backend in ["official", "repair"]}
        for pair in readl(folder / "pairs.jsonl"):
            scores = {f"{name}_{side}": scalar(maps[backend][pair[f"{side}_query_uid"]])
                      for name, backend in [("origin", "official"), ("repair", "repair")] for side in ["base", "cf"]}
            rows.append({"sample_id": pair["sample_id"], "dimension": dimension,
                         "primary": pair["primary"] and pair["split"] == "test", "scores": scores})
    else:
        variant = "start/background_corrupt" if dimension == "subject_consistency" else "full/subject_blur"
        repair = "hybrid_exclude" if dimension == "subject_consistency" else "patch_frame_all_pairs_calibrated"
        paths = sorted((folder / "historical-scores").rglob("scores.jsonl"))
        for path in paths:
            for item in readl(path):
                uid = item["base"]["video_uid"]
                accepted = item["status"] == "completed"
                scores = {f"{name}_{side}": scalar(item["variants"][version]["scores"][backend]) if accepted
                          else {"score": None, "status": item["status"]}
                          for name, backend in [("origin", "official"), ("repair", repair)]
                          for side, version in [("base", "clean"), ("cf", variant)]}
                rows.append({"sample_id": uid, "dimension": dimension, "scores": scores,
                             "primary": accepted and (dimension != "subject_consistency" or uid != "v_4f8abc7bb931501fb862")})
    return rows


def compare_rows(rows, expected):
    actual, frozen = unique(rows, "sample_id"), unique(expected, "sample_id")
    if actual.keys() != frozen.keys():
        raise ValueError("Candidate coverage differs from frozen scores")
    checked = 0
    for uid, row in actual.items():
        if row["primary"] != frozen[uid]["primary"]:
            raise ValueError(f"Primary cohort changed: {uid}")
        for key in CELLS:
            for field in ["score", "status", "raw_partial_diagnostic"]:
                if row["scores"][key].get(field) != frozen[uid]["scores"][key].get(field):
                    raise ValueError(f"Frozen result changed: {uid} / {key} / {field}")
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-code", type=Path, required=True)
    parser.add_argument("--k400-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.bundle.resolve() or args.bundle.resolve() in args.output.resolve().parents:
        raise ValueError("Write reproduction output outside the frozen input bundle")
    results = replay_semantic(args.bundle, args.model_code.resolve(), args.k400_labels)
    for dimension in DIMENSIONS:
        if dimension not in results:
            results[dimension] = replay_visual_scores(args.bundle, dimension)
    with (args.bundle / "reports/main-table.csv").open() as f:
        expected_table = {row["dimension"]: row for row in csv.DictReader(f)}
    summaries, checked = [], {}
    for dimension in DIMENSIONS:
        suffix = "replay" if dimension in TASKS else "fresh" if dimension in {"object_class", "color"} else "verified"
        frozen = readl(args.bundle / "dimensions" / dimension / f"scores/paired-{suffix}.jsonl")
        checked[dimension] = compare_rows(results[dimension], frozen)
        summary = {"dimension": dimension, **summarize(results[dimension])}
        for key in ["planned_primary", "complete_pairs", *CELLS]:
            if not math.isclose(float(summary[key]), float(expected_table[dimension][key]), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Main-table mismatch: {dimension} / {key}")
        summaries.append(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "main-table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {"status": "passed", "scope": "frozen_evidence_formula_replay_and_score_reaggregation",
              "fresh_gpu_inference": False, "fresh_training": False, "dimensions": summaries,
              "exact_score_cells_checked": checked, "all_primary_flags_and_statuses_checked": True,
              "input_files": [{"path": p.relative_to(args.bundle).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in sorted(args.bundle.rglob("*")) if p.is_file()]}
    (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "dimensions": len(summaries),
                      "exact_score_cells": sum(checked.values()), "fresh_gpu_inference": False}), flush=True)


if __name__ == "__main__":
    main()
