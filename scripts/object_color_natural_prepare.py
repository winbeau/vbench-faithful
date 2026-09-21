"""Freeze all paper-annotation videos and pairs before natural-set scoring."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.upstream import verify_upstream


def write_once(path, payload):
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite a different frozen manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def prepare(args):
    out = args.output
    if "output" not in out.resolve().parts:
        raise ValueError("new cohort files must be under output/")
    verify_upstream(args.upstream)
    full = json.loads((args.upstream / "vbench/VBench_full_info.json").read_text())
    seed = json.loads((args.config / "object_color_seeds.json").read_text())["rows"]
    seedmap = {(r["dimension"], r["prompt"]): r for r in seed}
    rows = list(csv.DictReader(args.manifest.open()))
    pp = list(csv.DictReader(args.pairs.open()))
    allrecords = []
    for dimension, alias in (("object_class", "object"), ("color", "color")):
        selected = [r for r in rows if r["dimension"] == alias]
        relevant = {r["prompt_en"]: r for r in full if dimension in r["dimension"]}
        for r in selected:
            prompt = r["prompt_id"]
            aux = relevant[prompt]["auxiliary_info"][dimension]
            s = seedmap[dimension, prompt]
            allrecords.append({**r, "dimension": dimension, "annotation_dimension": alias,
                "prompt": prompt, "official_auxiliary_info": aux,
                "semantic_training_split": s["split"], "semantic_source_group": s["source_group"]})
        if len(selected) != (1580 if dimension == "object_class" else 1700):
            raise ValueError("the complete frozen official cohort count changed")
        if {r["prompt_id"] for r in selected} != relevant.keys():
            raise ValueError("annotated prompts differ from the pinned official dimension suite")
        dimension_rows = [r for r in allrecords if r["dimension"] == dimension]
        dimension_pairs = [r for r in pp if r["dimension"] == alias]
        uid_rows = {r["video_uid"]: r for r in dimension_rows}
        if len(uid_rows) != len(dimension_rows):
            raise ValueError("duplicate source video UID")
        for p in dimension_pairs:
            for key in ("video_a_uid", "video_b_uid"):
                r = uid_rows[p[key]]
                if r["split"] != p["split"] or r["prompt_id"] != p["prompt_id"]:
                    raise ValueError("source pair/clip split identity mismatch")
        write_once(out / (dimension + "-manifest.json"), {"dimension": dimension, "rows": dimension_rows})
        write_once(out / (dimension + "-pairs.json"), {"rows": dimension_pairs})
    protocol = {
        "schema_version": 1,
        "dataset": "VBench 1.0 paper official per-dimension human preference videos",
        "generation_models": ["cogvideo", "lavie", "modelscope", "videocraft"],
        "upstream_sha": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490",
        "hf_dataset_revision": "8c89bb0218a12323d04821d11ec03c780d0dfb9c",
        "source_manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        "source_pairs": {"path": str(args.pairs), "sha256": sha256_file(args.pairs)},
        "vocabulary_sha256": sha256_file(args.config / "object_color_vocabulary.json"),
        "primary_backends": ["official", "repair_deterministic"],
        "additional_backends": ["repair_base", "repair_lora"],
        "train_new_weights": False,
        "primary_population": "all official annotated videos; deterministic repair uses no learned prompt adapter",
        "paper_alignment": "4-generator win-ratio Pearson; metric exact ties and human ties each 0.5; undefined pairs retained and coverage disclosed",
        "additional_alignment": "frozen prompt-disjoint dev/test pair accuracy, zero margin and dev-only calibrated ties; full expected-pair denominator with undefined counted unsuccessful, plus common-scored-pair comparison",
        "lora_evaluation_populations": ["all descriptive with training overlap disclosed", "official test intersect original semantic test (not train or semantic dev)"],
        "bootstrap": {"resamples": 10000, "seed": 20260920, "cluster": "raw prompt", "paired": True},
        "missing_policy": "retain every expected video and pair; no selective removal or post-score filtering",
        "formula_policy": "use committed afd48ec formulas without tuning",
        "files": {p.name: sha256_file(p) for p in sorted(
            [out / (d + suffix) for d in ("object_class", "color") for suffix in ("-manifest.json", "-pairs.json")])},
    }
    write_once(out / "protocol.json", protocol)
    write_once(out / "official-prompts.json", {d: sorted({r["prompt"] for r in allrecords if r["dimension"] == d})
                                               for d in ("object_class", "color")})
    print(json.dumps({"videos": len(allrecords), "output": str(out),
                      "protocol_sha256": sha256_file(out / "protocol.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/object_color_natural_20260920"))
    parser.add_argument("--upstream", type=Path, default=Path("../VBench"))
    parser.add_argument("--config", type=Path, default=Path("configs/four_dimension"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/e0_scoring_manifest.csv"))
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/pairwise_master_split.csv"))
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
