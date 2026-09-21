"""Generate the reviewable Object/Color tables from frozen measured artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np

from vbench_audit_models.labels import LabelVocabulary, compile_prompt, validate_compilation
from vbench_audit_core.inputs import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def verification_record(root):
    directory = root / "verification"
    log = directory / "pure-tests-final.log"
    match = re.search(r"(\d+) passed, (\d+) skipped in ([\d.]+)s", log.read_text())
    if match is None:
        raise ValueError("missing successful full-suite verification")
    wheels = json.loads((directory / "isolated-wheels.json").read_text())
    if len(wheels) != 2 or any(r["other_metric_installed"] or r["workspace_found"] or r["help_exit"] != 0
                             or r["missing_config_exit"] != 1 or r["null_failure_rows"] != 2 for r in wheels):
        raise ValueError("independent wheel verification incomplete")
    return {"full_test_command": "uv run --no-sync --group test pytest tests metrics -q -rs",
        "passed": int(match[1]), "skipped": int(match[2]), "seconds": float(match[3]),
        "skip_reasons": [line for line in log.read_text().splitlines() if line.startswith("SKIPPED")],
        "log_sha256": sha256_file(log), "isolated_wheels": wheels,
        "help_logs": {p.name: sha256_file(p) for p in sorted(directory.glob("*-help.txt"))},
        "environment_logs": {name: sha256_file(directory / name) for name in ("lock-check.log", "sync.log", "cpu-overlay.log")},
        "built_artifacts": {p.name: sha256_file(p) for p in sorted((root / "dist").iterdir())
                            if p.is_file() and p.name.endswith((".whl", ".tar.gz"))},
        "scope": "local CPU/build/interface checks; real H100 parity is recorded separately"}


def semantic_tables(root, vocabulary):
    records = json.loads((root / "semantics/records.json").read_text())["rows"]
    summary, details = [], []
    for dimension in ("object_class", "color"):
        compiled = {mode: {r["prompt"]: r for r in json.loads((root / "compiled" / f"{dimension}-{mode}.json").read_text())["records"]}
                    for mode in ("base", "lora")}
        for split in ("dev", "test"):
            for quality in ("gold_seed", "silver"):
                selected = [r for r in records if r["dimension"] == dimension and r["split"] == split and r["quality"] == quality]
                for mode in ("deterministic", "base", "lora"):
                    matched, invalid, null = 0, 0, 0
                    by_group = defaultdict(list)
                    for row in selected:
                        prediction = ({"output": compile_prompt(dimension, row["prompt"], vocabulary), "status": "succeeded"}
                                      if mode == "deterministic" else compiled[mode][row["prompt"]])
                        agrees = prediction["output"] == row["target"]
                        matched += agrees
                        invalid += prediction["status"] != "succeeded"
                        null += prediction["output"] is not None and prediction["output"]["object"] is None
                        by_group[row["source_group"]].append(agrees)
                        details.append({"dimension": dimension, "mode": mode, "sample_id": row["sample_id"],
                            "source_group": row["source_group"], "split": split, "quality": quality,
                            "reference_agreement": agrees, "status": prediction["status"],
                            "prediction": prediction["output"], "reference": row["target"], "human_reviewed": False})
                    rates = np.asarray([np.mean(v) for v in by_group.values()])
                    samples = np.random.default_rng(20260920).choice(rates, (10000, len(rates)), replace=True)
                    summary.append({"dimension": dimension, "split": split, "quality": quality, "mode": mode,
                        "n_prompts": len(selected), "n_source_groups": len(by_group), "reference_matches": int(matched),
                        "n_unique_prompts": len({r["prompt"] for r in selected}),
                        "reference_agreement": matched/len(selected), "invalid_outputs": int(invalid), "null_object_outputs": int(null),
                        "group_bootstrap_ci95": np.quantile(samples.mean(1), [.025, .975]).tolist(),
                        "human_review_completed": 0, "claim": "agreement with official seed/silver reference; not human-verified annotation accuracy"})
    return summary, details


def verify_inputs(root, config, vocabulary):
    """Check the frozen data actually used, including contamination and nulls."""
    checked = 0
    for directory, filename in ((config, "object_color_freeze.json"), (root / "semantics", "freeze.json")):
        frozen = json.loads((directory / filename).read_text())
        for name, expected in frozen["files"].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f"frozen artifact changed: {directory / name}")
            checked += 1
    seeds = json.loads((config / "object_color_seeds.json").read_text())["rows"]
    if Counter(r["dimension"] for r in seeds) != {"object_class": 79, "color": 85}:
        raise ValueError("official seed coverage changed")
    records = json.loads((root / "semantics/records.json").read_text())["rows"]
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate semantic record ID")
    source_split = {r["source_group"]: r["split"] for r in seeds}
    training = {}
    conflicting = defaultdict(list)
    for row in records:
        if row["split"] != source_split[row["source_group"]]:
            raise ValueError("expansion crossed its frozen source split")
        validate_compilation(row["dimension"], row["target"], vocabulary)
        conflicting[(row["dimension"], row["prompt"])].append(row)
    for dimension in ("object_class", "color"):
        chosen = [r for r in records if r["dimension"] == dimension and r["training_eligible"]]
        heldout = [r for r in records if r["dimension"] == dimension and r["split"] != "train"]
        norm = lambda r: " ".join(r["prompt"].casefold().split())
        if ({r["source_group"] for r in chosen} & {r["source_group"] for r in heldout}
                or {norm(r) for r in chosen} & {norm(r) for r in heldout}):
            raise ValueError("training contamination in frozen source groups or literal prompts")
        manifest = json.loads((root / "training" / dimension / "training_manifest.json").read_text())
        if manifest["n"] != len(chosen) or manifest["data_sha256"] != sha256_file(root / "semantics/records.json"):
            raise ValueError("trained records do not match frozen data")
        if set(manifest["source_groups"]) != {r["source_group"] for r in chosen}:
            raise ValueError("training source manifest differs")
        training[dimension] = len(chosen)
    observation = json.loads((config / "object_color_observation.json").read_text())
    for row in observation["records"]:
        path = root / "observation" / Path(row["path"]).name
        if sha256_file(path) != row["sha256"] or source_split[row["source_group"]] != "train":
            raise ValueError("GRiT observation trace changed or uses heldout data")
        checked += 1
    expected_objects = {r["gold"]["object"] for r in seeds if r["dimension"] == "object_class"} | set(observation["labels"])
    if set(vocabulary.objects) != expected_objects:
        raise ValueError("object vocabulary is not exactly official union observed")
    review = json.loads((root / "semantics/human_review.pending.json").read_text())["rows"]
    if len(review) > 200 or any(r["reviewed"] for r in review):
        raise ValueError("human review state changed; update the report claims before regenerating")
    conflicts = [{"dimension": key[0], "prompt": key[1],
                  "references": [{k: r[k] for k in ("source_group", "split", "target", "training_eligible")} for r in rows]}
                 for key, rows in conflicting.items() if len({json.dumps(r["target"], sort_keys=True) for r in rows}) > 1]
    return {"verified_frozen_files": checked, "seed_counts": dict(Counter(r["dimension"] for r in seeds)),
        "semantic_records": len(records), "training_records": training, "training_heldout_source_overlap": 0,
        "training_heldout_normalized_prompt_overlap": 0, "observed_training_sources": len(observation["records"]),
        "object_vocabulary_count": len(expected_objects), "human_review_slots": len(review), "human_review_completed": 0,
        "conflicting_reference_prompts_retained": conflicts,
        "scope": "integrity and split checks, not human correctness or successful scientific gates"}


def video_ablation(root):
    rows, detail = [], []
    for dimension in ("object_class", "color"):
        metadata = json.loads((root / "families" / dimension / "metadata.json").read_text())["videos"]
        original = {r["query_uid"] for r in metadata if r["split"] == "test" and
            (r.get("variant") == "canonical" if dimension == "object_class" else
             r["family"] == "color_visibility_denominator" and r["visible_fraction"] == 1)}
        for stage, directory in (("old_string_official", "official"),
                                 ("deterministic_alias_repair", "repair-deterministic-canonical"),
                                 ("unfinetuned_same_base_repair", "repair-base-canonical"),
                                 ("independent_lora_repair", "repair-lora-canonical")):
            path = root / "scores" / dimension / directory / "results.json"
            values = [r for r in json.loads(path.read_text()) if r["query_uid"] in original]
            if len(values) != len(original) or {r["query_uid"] for r in values} != original:
                raise ValueError("head ablation lost original-query coverage")
            scores = [r["score"] for r in values if r["status"] == "succeeded" and r["score"] is not None]
            lookup = {r["query_uid"]: r for r in metadata}
            detail.extend({"dimension": dimension, "stage": stage, "split": "test",
                "base_id": lookup[r["query_uid"]]["base_id"], "query_uid": r["query_uid"],
                "score": r["score"], "status": r["status"]} for r in values)
            rows.append({"dimension": dimension, "stage": stage, "split": "test", "n_inputs": len(values),
                         "n_scored": len(scores), "status_counts": dict(Counter(r["status"] for r in values)),
                         "coverage": len(scores)/len(values),
                         "aggregate": float(np.mean(scores)) if len(scores) == len(values) else None,
                         "observed_subset_mean_diagnostic_only": float(np.mean(scores)) if scores else None,
                         "source_results_sha256": sha256_file(path),
                         "scope": "original queries only; metadata-only label interventions never enter prompt-only model"})
    return rows, detail


def execution_tables(root):
    """Keep measured time, missing results and asset identity next to effects."""
    inventory_path = root / "inventory/execution.json"
    inventory = json.loads(inventory_path.read_text())
    timing, coverage, runs = [], [], {}
    for path in sorted((root / "scores").glob("*/*/run.json")):
        dimension, variant = path.parent.parent.name, path.parent.name
        run = json.loads(path.read_text())
        summary = json.loads(path.with_name("summary.json").read_text())
        results = json.loads(path.with_name("results.json").read_text())
        counts = Counter(r["status"] for r in results)
        coverage.append({"dimension": dimension, "variant": variant, "n_inputs": len(results),
            "status_counts": dict(counts), "n_scored": counts.get("succeeded", 0),
            "score_coverage": counts.get("succeeded", 0)/len(results),
            "denominator_kind": summary["denominator_kind"],
            "n_retained_null": sum(r["score"] is None for r in results)})
        runs[dimension + "/" + variant] = {"path": str(path), "sha256": sha256_file(path),
            "code_sha256": run.get("code_sha256"), "family_freeze_sha256": run.get("family_freeze_sha256"),
            "provenance": run.get("provenance", [])[:1], "measurement": run.get("measurement")}
        if "wall_seconds" in run:
            timing.append({"dimension": dimension, "run": variant, "n_video_requests": len(results),
                "physical_gpu": run["physical_gpu"], "wall_seconds": run["wall_seconds"],
                "four_card_wall_seconds": None, "output": str(path.parent),
                "scope": "one-card GRiT scoring; compiled prompt cache already prepared"})
    parity = {}
    for dimension in ("object_class", "color"):
        path = root / "cli_parity" / dimension / "parity.json"
        parity[dimension] = json.loads(path.read_text())
        timing.append({"dimension": dimension, "run": "cli_four_shards_both_backends",
            "n_video_requests": 4, "physical_gpu": parity[dimension]["physical_gpu_visibility"],
            "wall_seconds": parity[dimension]["four_card_wall_seconds"],
            "four_card_wall_seconds": parity[dimension]["four_card_wall_seconds"],
            "output": str(path.parent), "scope": "four original videos, both backends; not full-cohort four-card timing"})
    training = {d: {name: json.loads((root / "training" / d / f"{name}.json").read_text())
                   for name in ("training_manifest", "training_result")} for d in ("object_class", "color")}
    qwen_runtime = json.loads((root / "inventory/qwen.json").read_text())
    pinned_base = json.loads((root / "inventory/h100-qwen-base.json").read_text())
    for dimension, record in training.items():
        expected = record["training_manifest"]["base_file_sha256"]
        if expected != pinned_base["base_file_sha256"]:
            raise ValueError("training backbone differs from pinned H100 snapshot")
        actual = next(r for r in qwen_runtime["adapters"] if r["dimension"] == dimension)
        if actual["sha256"] != record["training_result"]["adapter_sha256"]:
            raise ValueError("adapter changed after training")
        for mode in ("base", "lora"):
            compiled = json.loads((root / "compiled" / f"{dimension}-{mode}.json").read_text())
            if compiled["input_fields"] != ["prompt"] or compiled["dimension"] != dimension or compiled["mode"] != mode:
                raise ValueError("prompt-only compilation route changed")
            model = compiled["model"]
            if model["declared_revision"] != pinned_base["revision"]:
                raise ValueError("compiled model revision differs")
            if any(expected.get(k) != v for k, v in model["base_file_sha256"].items()):
                raise ValueError("compiled backbone differs from trained backbone")
            if model["adapters"][dimension]["sha256"] != actual["sha256"]:
                raise ValueError("compiled adapter differs from trained adapter")
    for dimension, record in training.items():
        timing.append({"dimension": dimension, "run": "lora_300_steps", "n_video_requests": 0,
            "physical_gpu": "RTX4090:5", "wall_seconds": record["training_result"]["metrics"]["train_runtime"],
            "four_card_wall_seconds": None, "output": "/data1/wenbiao_zhao/vbench-audit-object-color-20260920/output/adapters/"+dimension,
            "scope": "trainer time; excludes model loading, silver generation and compilation"})
    execution = {"inventory": inventory, "inventory_sha256": sha256_file(inventory_path),
        "score_runs": runs, "cli_parity": parity, "training": training,
        "qwen_runtime": qwen_runtime, "pinned_qwen_base": pinned_base,
        "teacher_identity": json.loads((root / "semantics/teacher_identity.json").read_text()),
        "semantic_freeze_sha256": sha256_file(root / "semantics/freeze.json"),
        "construction_models": json.loads((root / "construction/color/protocol.json").read_text()),
        "scope": "real measured runs, with post-run media inventory; no human accuracy or frozen E0 replication claim"}
    execution["frame_diagnostics"] = {}
    correction = root / "verification/summary-denominator-correction.json"
    if correction.exists():
        execution["postprocess_summary_label_corrections"] = json.loads(correction.read_text())
    for dimension in ("object_class", "color"):
        reasons, binding = Counter(), Counter()
        frames, instances = 0, 0
        results = json.loads((root / "scores" / dimension / "repair/results.json").read_text())
        for row in results:
            path = root / "evidence" / dimension / "repair" / Path(row["evidence"]["path"]).name
            if sha256_file(path) != row["evidence"]["sha256"]:
                raise ValueError("evidence changed before delivery")
            trace = json.loads(path.read_text())
            frames += len(trace["raw_frames"])
            reasons.update(trace["scoring"].get("reason_counts", {}))
            for frame in trace["raw_frames"]:
                instances += len(frame.get("objects", []))
                binding.update(p["method"] for p in (frame.get("binding") or {}).get("pairs", []))
        execution["frame_diagnostics"][dimension] = {"frames": frames, "instances": instances,
            "reason_counts": dict(reasons), "binding_methods": dict(binding),
            "scope": "all constructed queries; repeated source frames are not independent observations"}
    return timing, coverage, execution


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("output/object_color_20260920"))
    p.add_argument("--config", type=Path, default=Path("configs/four_dimension"))
    p.add_argument("--output", type=Path, default=Path("docs/counterfactual-reports/object_color_repair_20260920"))
    args = p.parse_args()
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    audit = verify_inputs(args.root, args.config, vocabulary)
    reports = {d: json.loads((args.root / "reports" / d / "main.json").read_text()) for d in ("object_class", "color")}
    details = [r for d in reports for r in json.loads((args.root / "reports" / d / "per_base.json").read_text())["rows"]]
    semantic, semantic_detail = semantic_tables(args.root, vocabulary)
    ablation, ablation_detail = video_ablation(args.root)
    timing, coverage, execution = execution_tables(args.root)
    # The missing Object synonym row has no frozen alias, not a failed score.
    # Keep the original all-accepted denominator and also expose eligibility.
    object_detail = json.loads((args.root / "reports/object_class/per_base.json").read_text())["rows"]
    for row in reports["object_class"]["main"]:
        if row["family"] == "object_label_invariance":
            eligible = sum(r["split"] == row["split"] and r["variant"] == row["contrast"] for r in object_detail)
            row["eligible_bases"] = eligible
            row["defined_pair_coverage_of_eligible"] = row["n"]/eligible if eligible else None
    table(args.output / "main.csv", [r for report in reports.values() for r in report["main"]])
    table(args.output / "per_base.csv", details)
    table(args.output / "prompt_compilation.csv", semantic)
    table(args.output / "video_ablation.csv", ablation)
    table(args.output / "video_ablation_per_base.csv", ablation_detail)
    table(args.output / "coverage.csv", coverage)
    table(args.output / "timing.csv", timing)
    table(args.output / "media.csv", execution["inventory"]["media"])
    write(args.output / "execution_manifest.json", execution)
    write(args.output / "integrity_audit.json", audit)
    write(args.output / "verification.json", verification_record(args.root))
    write(args.output / "prompt_compilation_details.json", semantic_detail)
    write(args.output / "main.json", {"reports": reports, "prompt_compilation": semantic, "video_ablation": ablation,
        "source_files": {str(p): sha256_file(p) for p in [args.root / "reports/object_class/main.json", args.root / "reports/color/main.json",
                                                       args.root / "semantics/freeze.json", args.config / "object_color_vocabulary.json"]}})
    bases = []
    for dimension in reports:
        frozen = json.loads((args.root / "families" / dimension / "freeze.json").read_text())
        source_rows = json.loads((args.root / "localization" / dimension / "manifest.json").read_text())["rows"]
        metadata = json.loads((args.root / "families" / dimension / "metadata.json").read_text())["videos"]
        accepted = {r["base_id"] for r in metadata}
        rejected = {r["base_id"]: r for r in frozen["rejections"]}
        for r in source_rows:
            source = r["source"]; base = source["source_group"]
            bases.append({"dimension": dimension, "base_id": base, "split": source["split"],
                "prompt": source["prompt"], "source_sha256": source["sha256"], "archive_member": source["archive_member"],
                "generator": source["generator"], "seed": source["seed"],
                "status": "accepted" if base in accepted else "rejected",
                "rejection_reason": rejected.get(base, {}).get("reason"),
                "query_uids": [q["query_uid"] for q in metadata if q["base_id"] == base],
                "frozen_family_sha256": sha256_file(args.root / "families" / dimension / "freeze.json")})
    write(args.config / "object_color_family_manifest.json", {"rows": bases, "no_backfill": True,
          "selection_protocol_sha256": sha256_file(args.config / "object_color_protocol.json")})
    print(json.dumps({"output": str(args.output), "bases": len(bases), "video_ablation": ablation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
