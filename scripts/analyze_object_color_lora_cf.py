#!/usr/bin/env python3
"""Verify new visual inference and summarize LoRA endpoint pairs; never fill nulls."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.complete_object_color_lora_cf import PINS, digest, save
from scripts.reproduce_main_table import CELLS, scalar, summarize, unique


def analyze(root, historical, vocabulary):
    from vbench_audit_models.labels import LabelVocabulary
    labels = LabelVocabulary.from_file(vocabulary)
    summaries, paired, artifacts = {}, {}, {}
    for dimension in PINS:
        family = root / "families" / dimension
        freeze = json.loads((family / "freeze.json").read_text())
        metadata = family / "metadata.json"
        if digest(metadata) != freeze["metadata_sha256"]:
            raise ValueError("subset changed after freeze")
        compiled = root / "compiled" / f"{dimension}-lora.json"
        if digest(compiled) != PINS[dimension]["compiled"]:
            raise ValueError("compiled LoRA identity mismatch")
        expected = {r["query_uid"] for r in json.loads(metadata.read_text())["videos"]}
        maps = {}
        for backend in ("official", "repair-lora"):
            path = root / "scores" / dimension / backend / "results.json"
            maps[backend] = unique(json.loads(path.read_text()), "query_uid")
            if set(maps[backend]) != expected:
                raise ValueError("incomplete query accounting")
            artifacts[str(path)] = digest(path)
        original = unique(json.loads((historical / dimension / "historical-scores" /
                         "repair-lora-canonical/results.json").read_text()), "query_uid")
        old_official = unique(json.loads((historical / dimension / "historical-scores" /
                             "official/results.json").read_text()), "query_uid")
        for uid, result in maps["official"].items():
            if scalar(result) != scalar(old_official[uid]):
                raise ValueError("fresh Origin differs from frozen historical scores or statuses")
        module = importlib.import_module(dimension + ".algorithms")
        errors = []
        for result in maps["repair-lora"].values():
            if result.get("query_source") != "lora":
                raise ValueError("metadata repair must not be relabelled as LoRA")
            identity = result["provenance"]["compiler"]
            if identity["sha256"] != PINS[dimension]["compiled"]:
                raise ValueError("worker used different text predictions")
            evidence = root / "evidence" / dimension / "repair-lora" / Path(result["evidence"]["path"]).name
            if digest(evidence) != result["evidence"]["sha256"]:
                raise ValueError("raw visual evidence SHA mismatch")
            raw = json.loads(evidence.read_text())
            query = raw["query"]
            if dimension == "object_class":
                replay = module.score_frames(raw["raw_frames"], query["object"], labels, lexical=True)
            else:
                replay = module.score_frames(raw["raw_frames"], query["object"], query["color"], labels, variant="repair")
            if replay["status"] != result["status"] or replay["score"] != result["score"]:
                raise ValueError("raw evidence formula replay differs")
            errors.append(0.0)
        rows, base_errors = [], []
        for pair in freeze["pairs"]:
            scores = {f"{name}_{side}": scalar(maps[backend][pair[f"{side}_query_uid"]])
                      for name, backend in (("origin", "official"), ("repair", "repair-lora"))
                      for side in ("base", "cf")}
            uid = pair["base_query_uid"]
            before, after = scalar(original[uid]), scalar(maps["repair-lora"][uid])
            if before["status"] != after["status"]:
                raise ValueError("canonical LoRA status changed")
            base_errors.append(abs(before["score"] - after["score"]) if before["score"] is not None else None)
            rows.append({**pair, "sample_id": pair["base_id"], "dimension": dimension, "primary": True, "scores": scores})
        common = [r for r in rows if all(r["scores"][key]["score"] is not None for key in CELLS)]
        all_repair = [r for r in rows if all(r["scores"]["repair_" + side]["score"] is not None for side in ("base", "cf"))]
        summaries[dimension] = {
            **summarize(rows), "status_counts": {key: dict(Counter(r["status"] for r in value.values())) for key, value in maps.items()},
            "repair_all_available_pairs": len(all_repair),
            "repair_all_base": statistics.mean(r["scores"]["repair_base"]["score"] for r in all_repair),
            "repair_all_cf": statistics.mean(r["scores"]["repair_cf"]["score"] for r in all_repair),
            "common_repair_exactly_unchanged": sum(r["scores"]["repair_base"]["score"] == r["scores"]["repair_cf"]["score"] for r in common),
            "canonical_parity_max_abs_error": max((e for e in base_errors if e is not None), default=None),
            "raw_evidence_replayed": len(errors), "raw_evidence_max_abs_error": max(errors),
            "origin_historical_parity_rows": len(maps["official"]),
            "origin_historical_parity_exact": True,
        }
        paired[dimension] = rows
    return {"schema": "object-color-lora-cf-completion/1", "summaries": summaries,
            "semantic_execution": "frozen selected LoRA compiled outputs; NOT fresh Qwen inference",
            "visual_execution": "fresh GRiT GPU inference for base and CF, both backends",
            "artifacts_sha256": artifacts, "paired": paired}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--historical-dimensions", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=Path("configs/four_dimension/object_color_vocabulary.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run, args.historical_dimensions, args.vocabulary)
    save(args.output, report)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
