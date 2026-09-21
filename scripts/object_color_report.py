"""Base-level paired effects and coverage. No pooled/tie-margin CPA headline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib
import json
from pathlib import Path

import numpy as np

from vbench_audit_models.labels import LabelVocabulary
from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.schemas import VideoResult
from vbench_audit_core.outputs import write_results
from scripts.object_color import save


def paired(values, *, seed=20260920, repeats=10000):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "median_abs_delta": None, "signed_ci95": None, "abs_ci95": None, "zero_effect_count": 0}
    if not np.isfinite(array).all():
        raise ValueError("nonfinite effects must be missing with explicit coverage, not silently dropped")
    indexes = np.random.default_rng(seed).integers(0, len(array), (repeats, len(array)))
    samples = array[indexes]
    return {"n": len(array), "median_delta": float(np.median(array)), "median_abs_delta": float(np.median(abs(array))),
            "signed_ci95": np.quantile(np.median(samples, axis=1), [.025, .975]).tolist(),
            "abs_ci95": np.quantile(np.median(abs(samples), axis=1), [.025, .975]).tolist(),
            "zero_effect_count": int((array == 0).sum())}


def gate(stats, relation):
    if not stats["n"]:
        return None
    if relation == "change":
        return stats["median_abs_delta"] >= .20 and stats["abs_ci95"][0] > .10
    if relation == "ordered_drop":
        return stats["median_delta"] >= .20 and stats["signed_ci95"][0] > .10
    if relation == "invariance":
        return stats["median_abs_delta"] <= .05 and stats["signed_ci95"][0] <= 0 <= stats["signed_ci95"][1]
    raise ValueError("unknown gate")


def load_run(root, dimension, variant, metadata):
    directory = root / "scores" / dimension / variant
    results = json.loads((directory / "results.json").read_text())
    expected = {m["query_uid"] for m in metadata}
    if len(results) != len(expected) or {r["query_uid"] for r in results} != expected:
        raise ValueError("incomplete or duplicate query coverage")
    if len({r["video_uid"] for r in results}) != len(results):
        raise ValueError("duplicate video_uid")
    return {r["query_uid"]: r for r in results}


def object_report(args):
    root = args.output
    family = root / "families/object_class"
    metadata = json.loads((family / "metadata.json").read_text())["videos"]
    frozen = json.loads((family / "freeze.json").read_text())
    if frozen["metadata_sha256"] != sha256_file(family / "metadata.json"):
        raise ValueError("family mutated after freezing")
    official = load_run(root, "object_class", "official", metadata)
    repair = load_run(root, "object_class", "repair", metadata)
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    from object_class.algorithms import score_frames
    lookup = {r["query_uid"]: r for r in metadata}
    detail, replay_errors = [], []
    for query in metadata:
        uid = query["query_uid"]
        o, a = official[uid], repair[uid]
        artifact = root / "evidence/object_class/repair" / Path(a["evidence"]["path"]).name
        if sha256_file(artifact) != a["evidence"]["sha256"]:
            raise ValueError("raw evidence hash mismatch")
        frames = json.loads(artifact.read_text())["raw_frames"]
        legacy = score_frames(frames, query["dimension_metadata"]["object_class"]["object"], vocabulary, lexical=False)
        parity_error = (abs(legacy["score"]-o["score"]) if legacy["score"] is not None and o["score"] is not None else None)
        replay_errors.append(parity_error)
        detail.append({"base_id": query["base_id"], "query_uid": uid, "split": query["split"],
                       "family": query["family"], "variant": query["variant"],
                       "official": o["score"], "repair": a["score"], "official_status": o["status"],
                       "repair_status": a["status"], "raw_trace_exact_rule": legacy["score"], "parity_error": parity_error})
    main = []
    for split in ("dev", "test"):
        bases = defaultdict(dict)
        for r in detail:
            if r["split"] == split:
                bases[r["base_id"]][r["variant"]] = r
        for variant in ("uppercase", "synonym"):
            for backend, relation in (("official", "change"), ("repair", "invariance")):
                deltas = [v["canonical"][backend]-v[variant][backend] for v in bases.values()
                          if variant in v and v["canonical"][backend] is not None and v[variant][backend] is not None]
                stats = paired(deltas)
                main.append({"dimension": "object_class", "family": "object_label_invariance", "contrast": variant,
                    "split": split, "backend": backend, "relation": relation, **stats, "gate_passed": gate(stats, relation),
                    "accepted_bases": len(bases), "paired_coverage": len(deltas)/len(bases) if bases else 0})
        for backend in ("official", "repair"):
            values = [v["absent"][backend] for v in bases.values() if "absent" in v and v["absent"][backend] is not None]
            main.append({"dimension": "object_class", "family": "object_absent_control", "contrast": "automatic_absence_proxy",
                "split": split, "backend": backend, "relation": "both_low", "n": len(values),
                "median_score": float(np.median(values)) if values else None,
                "gate_passed": bool(np.median(values) <= .1) if values else None,
                "absence_human_verified": False})
    summary = {"dimension": "object_class", "main": main, "n_input_bases": frozen["n_input_bases"],
         "accepted_bases": len({r["base_id"] for r in metadata}), "rejected_bases": len(frozen["rejections"]),
         "queries": len(metadata), "official_status_counts": dict(Counter(r["status"] for r in official.values())),
         "repair_status_counts": dict(Counter(r["status"] for r in repair.values())),
         "raw_trace_exact_rule_parity": {"n": sum(e is not None for e in replay_errors),
                                         "max_abs_error": max(e for e in replay_errors if e is not None)},
         "scope": "metadata-only lexical counterfactual; not observed natural synonym frequency, human accuracy, or E0 replication",
         "freeze_sha256": sha256_file(family / "freeze.json")}
    destination = root / "reports/object_class"
    save(destination / "main.json", summary)
    save(destination / "per_base.json", {"rows": detail, "rejections": frozen["rejections"]})
    fields = list(detail[0]) if detail else []
    with (destination / "per_base.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(detail)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def color_report(args):
    from scipy.stats import spearmanr
    family = args.output / "families/color"
    metadata = json.loads((family / "metadata.json").read_text())["videos"]
    frozen = json.loads((family / "freeze.json").read_text())
    if sha256_file(family / "metadata.json") != frozen["metadata_sha256"]:
        raise ValueError("color family changed after freeze")
    variants = ("official", "binding", "binding_lexical", "repair")
    runs = {v: load_run(args.output, "color", v, metadata) for v in variants}
    detail = [{"base_id": r["base_id"], "query_uid": r["query_uid"], "split": r["split"],
               "family": r["family"], "visible_fraction": r["visible_fraction"],
               **{variant: runs[variant][r["query_uid"]]["score"] for variant in variants},
               **{variant+"_status": runs[variant][r["query_uid"]]["status"] for variant in variants}}
              for r in metadata]
    main, monotonic = [], []
    for split in ("dev", "test"):
        bases = defaultdict(dict)
        for row in detail:
            if row["split"] == split:
                key = row["visible_fraction"] if row["family"] == "color_visibility_denominator" else "synonym"
                bases[row["base_id"]][key] = row
        for variant in variants:
            endpoint_deltas, conditional_deltas, control_deltas, rhos, strict = [], [], [], [], []
            for base, rows in bases.items():
                baseline, zero = rows[1][variant], rows[0][variant]
                if baseline is not None and zero is not None:
                    endpoint_deltas.append(baseline-zero)
                available = [baseline-rows[level][variant] for level in (.75, .5, .25, 0)
                             if baseline is not None and rows[level][variant] is not None]
                if available:
                    # One statistic per base: largest absolute departure from
                    # the baseline among defined levels, preserving its sign.
                    conditional_deltas.append(max(available, key=abs))
                if "synonym" in rows and baseline is not None and rows["synonym"][variant] is not None:
                    control_deltas.append(baseline-rows["synonym"][variant])
                values = [rows[level][variant] for level in (0, .25, .5, .75, 1)]
                if all(x is not None for x in values):
                    rho = float(spearmanr([0, .25, .5, .75, 1], values).statistic) if len(set(values)) > 1 else 0.
                    ordered = all(b > a for a, b in zip(values, values[1:]))
                    rhos.append(rho); strict.append(ordered)
                    monotonic.append({"base_id": base, "split": split, "backend": variant,
                                      "spearman": rho, "strictly_monotonic": ordered})
            relation = "ordered_drop" if variant == "repair" else "invariance"
            values = endpoint_deltas if variant == "repair" else conditional_deltas
            stats = paired(values)
            rho_median = float(np.median(rhos)) if rhos else None
            strict_fraction = float(np.mean(strict)) if strict else None
            ordering_gate = (rho_median >= .8 or strict_fraction >= .7) if rhos else None
            main.append({"dimension": "color", "family": "color_visibility_denominator", "split": split,
                "backend": variant, "relation": relation, **stats, "effect_gate_passed": gate(stats, relation),
                "accepted_bases": len(bases), "paired_coverage": len(values)/len(bases) if bases else 0,
                "median_base_spearman": rho_median, "strict_monotonic_base_fraction": strict_fraction,
                "monotonic_n": len(rhos), "ordering_gate_passed": ordering_gate,
                "gate_passed": bool(gate(stats, relation) and ordering_gate) if variant == "repair" and stats["n"] else gate(stats, relation),
                "scope": "100-to-0 endpoints" if variant == "repair" else "maximum absolute departure over defined conditional levels; undefined retained"})
            control = paired(control_deltas)
            main.append({"dimension": "color", "family": "color_synonym_control", "split": split,
                "backend": variant, "relation": "invariance", **control,
                "accepted_bases": len(bases), "paired_coverage": len(control_deltas)/len(bases) if bases else 0,
                "gate_passed": gate(control, "invariance") if len(control_deltas) == len(bases) else None,
                "missing_pairs_prevent_full_control_claim": len(control_deltas) != len(bases)})
    destination = args.output / "reports/color"
    summary = {"dimension": "color", "main": main, "monotonic_per_base": monotonic,
        "n_input_bases": frozen["n_input_bases"], "accepted_bases": frozen["n_accepted_bases"],
        "rejected_bases": len(frozen["rejections"]), "queries": len(metadata),
        "status_counts": {v: dict(Counter(r["status"] for r in run.values())) for v, run in runs.items()},
        "freeze_sha256": sha256_file(family / "freeze.json"),
        "scope": "qualified automatic construction cohort, not human mask/color accuracy or natural preference improvement"}
    save(destination / "main.json", summary)
    save(destination / "per_base.json", {"rows": detail, "rejections": frozen["rejections"]})
    with (destination / "per_base.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(detail[0]) if detail else []); writer.writeheader(); writer.writerows(detail)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def replay_color_ablations(args):
    from color.algorithms import score_frames, legacy_trace_score
    from color.runtime import summarize
    root = args.output
    metadata = json.loads((root / "families/color/metadata.json").read_text())["videos"]
    repair = load_run(root, "color", "repair", metadata)
    official = load_run(root, "color", "official", metadata)
    vocabulary = LabelVocabulary.from_file(args.config / "object_color_vocabulary.json")
    rows = {"binding": [], "binding_lexical": []}
    parity = []
    for query in metadata:
        uid = query["query_uid"]
        source = repair[uid]
        path = root / "evidence/color/repair" / Path(source["evidence"]["path"]).name
        if sha256_file(path) != source["evidence"]["sha256"]:
            raise ValueError("color evidence hash mismatch")
        raw = json.loads(path.read_text())
        color = query["dimension_metadata"]["color"]["color"]
        target = query["prompt"].replace('a ', '').replace('an ', '').replace(color, '').strip()
        original = legacy_trace_score(raw["raw_frames"], query["prompt"], color)
        observed = official[uid]
        parity.append({"query_uid": uid, "official_status": observed["status"], "trace_status": original["status"],
                       "null_agrees": (observed["score"] is None) == (original["score"] is None),
                       "abs_error": abs(observed["score"]-original["score"]) if observed["score"] is not None and original["score"] is not None else None})
        for variant in rows:
            result = score_frames(raw["raw_frames"], target, color, vocabulary, variant=variant)
            fields = {k: v for k, v in source.items() if k not in {"video", "status", "score", "error"}}
            fields.update({k: v for k, v in result.items() if k not in {"frames", "score", "status"}})
            fields.update({"variant": variant, "formula_version": "color-"+variant+"-v1",
                           "measurement": "ablation_replay_on_same_real_grit_evidence",
                           "query_parser": "unchanged_official_prompt_replace"})
            rows[variant].append(VideoResult(source["video"], result["status"], result["score"], fields,
                                             None if result["status"] == "succeeded" else result["status"]))
    for variant, values in rows.items():
        write_results(root / "scores/color" / variant, values, summarize("audit", values),
                      {"measurement": "real GRiT trace replay; not a second GPU run or Official call",
                       "source_repair_results_sha256": sha256_file(root / "scores/color/repair/results.json")})
    save(root / "reports/color/legacy_trace_parity.json", {"rows": parity,
        "max_abs_error": max((r["abs_error"] for r in parity if r["abs_error"] is not None), default=None),
        "null_agreement_count": sum(r["null_agrees"] for r in parity), "n": len(parity)})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/four_dimension"))
    p.add_argument("--dimension", choices=["object_class", "color"], required=True)
    p.add_argument("--replay-ablations", action="store_true", help="color: consume retained GRiT traces through metric formulas")
    args = p.parse_args()
    if args.replay_ablations:
        if args.dimension != "color":
            raise ValueError("ablation replay flag applies only to color")
        replay_color_ablations(args)
    (object_report if args.dimension == "object_class" else color_report)(args)


if __name__ == "__main__":
    main()
