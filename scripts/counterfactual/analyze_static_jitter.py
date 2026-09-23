"""Clustered paired analysis, dev calibration, and explicit joint goal gates.

Origin is a Boolean decision. Repair is short-side lengths/second. They never
share a numeric difference or a purported common score scale.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .static_jitter import digest
from .qualify_static_jitter import qualification


def cluster_interval(values, groups, *, iterations=2000, seed=20260922):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    members = {g: values[np.asarray(groups) == g] for g in sorted(set(groups))}
    if len(members) < 2:
        return None  # A single prompt is not independent evidence for a CI.
    keys = list(members)
    sums = np.asarray([members[k].sum() for k in keys])
    counts = np.asarray([len(members[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), (iterations, len(keys)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def stats(values, groups):
    if not values:
        return {"n": 0, "mean": None, "ci95": None, "p95": None, "max": None}
    return {"n": len(values), "prompt_groups": len(set(groups)), "mean": float(np.mean(values)),
            "ci95": cluster_interval(values, groups), "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values))}


def load_scores(paths):
    combined = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            item = json.loads(line)
            row = combined.setdefault(item["candidate_id"], {"input_sha256": item["input_sha256"]})
            if row["input_sha256"] != item["input_sha256"]:
                raise ValueError("candidate identity collision across different construction runs")
            for backend in ("origin", "repair"):
                if backend in item:
                    if backend in row and row[backend] != item[backend]:
                        raise ValueError(f"duplicate nonidentical result: {item['candidate_id']}/{backend}")
                    row[backend] = item[backend]
    return combined


def get_score(row, scored, backend, variant="full"):
    record = scored.get(row["candidate_id"], {})
    if record and record["input_sha256"] != row.get("sha256"):
        raise ValueError("manifest/score input hash mismatch")
    value = record.get(backend, {})
    if row["status"] != "qualified" or value.get("status") != "succeeded":
        return None
    if backend == "repair" and variant != "full":
        return value.get("ablations", {}).get(variant)
    return value.get("score")


def paired_tables(rows, scored):
    baseline = {(r["base_id"], r["kind"], r["motion"]): r for r in rows if r["family"] == "clean"}
    cells = defaultdict(list)
    for row in rows:
        cells[(row["kind"], row["motion"], row["family"], row["amplitude"])].append(row)
    output = []
    for key, members in sorted(cells.items()):
        item = dict(zip(("kind", "motion", "family", "amplitude"), key))
        item.update(candidates=len(members), qualified=sum(r["status"] == "qualified" for r in members))
        item["construction_rejected_or_failed"] = len(members) - item["qualified"]
        for backend, variant in (("origin", "full"), ("repair", "raw_tracks"),
                                 ("repair", "verified_correspondence"), ("repair", "full")):
            before, after, delta, groups = [], [], [], []
            succeeded = 0
            for row in members:
                score = get_score(row, scored, backend, variant)
                succeeded += score is not None
                base = baseline.get((row["base_id"], row["kind"], row["motion"]))
                reference = get_score(base, scored, backend, variant) if base else None
                if score is not None and reference is not None:
                    before.append(reference)
                    after.append(score)
                    delta.append(score - reference)
                    groups.append(row["prompt_id"])
            result = {
                "scored": succeeded, "paired": len(delta),
                "coverage_qualified": succeeded / item["qualified"] if item["qualified"] else 0,
                "coverage_candidates": succeeded / len(members),
                "base": stats(before, groups), "counterfactual": stats(after, groups),
                "signed_delta": stats(delta, groups), "absolute_delta": stats(list(np.abs(delta)), groups),
                "positive_delta": stats(list(np.maximum(0, delta)), groups),
            }
            if backend == "origin":
                result["false_to_true_rate"] = (float(np.mean((np.asarray(before) == 0) & (np.asarray(after) == 1)))
                                                if delta else None)
                if key[0] == "static":
                    result["static_false_positive_rate"] = float(np.mean(after)) if after else None
            item[f"{backend}:{variant}"] = result
        output.append(item)
    return output


def ladder(rows, scored, *, backend="repair", family="clean", amplitude=0):
    grouped = defaultdict(dict)
    for row in rows:
        if row["kind"] == "translation" and row["family"] == family and row["amplitude"] == amplitude:
            grouped[(row["base_id"], row["seed"])][row["motion"]] = row
    correct, complete, expected, groups = [], 0, 0, []
    eligible_correct, eligible_groups, eligible_complete, eligible_ladders = [], [], 0, 0
    clean = {(r["base_id"], r["motion"]): r for r in rows
             if r["kind"] == "translation" and r["family"] == "clean"}
    for _, values in grouped.items():
        levels = sorted(values)
        eligible = all(r["status"] == "qualified"
                       and clean.get((r["base_id"], r["motion"]), {}).get("status") == "qualified"
                       for r in values.values())
        eligible_ladders += eligible
        for a, b in zip(levels[:-1], levels[1:]):
            expected += 1
            before, after = get_score(values[a], scored, backend), get_score(values[b], scored, backend)
            complete += before is not None and after is not None
            correct.append(float(before is not None and after is not None and after > before))
            groups.append(values[a]["prompt_id"])
            if eligible:
                eligible_correct.append(correct[-1])
                eligible_groups.append(groups[-1])
                eligible_complete += before is not None and after is not None
    return {"expected_pairs": expected, "complete_pairs": complete, "strict_ordering": stats(correct, groups),
            "input_qualified": {"eligible_ladders": eligible_ladders,
                "construction_excluded_ladders": len(grouped) - eligible_ladders,
                "expected_pairs": len(eligible_correct), "complete_pairs": eligible_complete,
                "strict_ordering": stats(eligible_correct, eligible_groups)}}


def calibrate(rows, scored, construction):
    if not rows or {r["split"] for r in rows} != {"dev"}:
        raise ValueError("calibration requires development-only data")
    medium = sorted(construction["motion_pixels_per_frame"])[2]
    refs = [get_score(r, scored, "repair") for r in rows
            if r["kind"] == "translation" and r["family"] == "clean" and r["motion"] == medium]
    refs = [r for r in refs if r is not None]
    order = ladder(rows, scored)["input_qualified"]["strict_ordering"]["mean"]
    if not refs or np.median(refs) <= 0 or order is None or order < 0.9:
        raise ValueError("calibration fails: nonzero medium anchor and >=90% clean ordering required")
    reference = float(np.median(refs))
    return {"R": reference, "epsilon": reference * 0.05, "dynamic_threshold": reference * 0.05,
            "medium_motion": medium, "anchor_count": len(refs), "epsilon_ratio": 0.05,
            "units": "short_side_lengths_per_second", "status": "development_calibration_not_test_acceptance"}


def goal_gates(rows, scored, construction, calibration):
    epsilon, reference = calibration["epsilon"], calibration["R"]
    static = [r for r in rows if r["kind"] == "static"]
    clean = {r["base_id"]: r for r in static if r["family"] == "clean"}
    qualified_sources = {r["base_id"] for r in static if r["family"] == "clean" and r["status"] == "qualified"}
    is_test = {r["split"] for r in rows} == {"test"}
    required_sources = 100 if is_test else construction["dev_prompts"] * 4
    output = {"source_scope": {"qualified_sources": len(qualified_sources), "required_sources": required_sources,
                               "pass": len(qualified_sources) >= required_sources}}
    for family in construction["families"]:
        per_base = defaultdict(list)
        for row in static:
            if row["family"] == family:
                per_base[row["base_id"]].append(row)
        low, invariant, worst_delta, groups, excluded = [], [], [], [], 0
        for base_id, members in per_base.items():
            base_row = clean[base_id]
            if any(r["status"] != "qualified" for r in members + [base_row]):
                excluded += 1
                continue  # input-only eligibility; excluded denominator is retained below
            base_score = get_score(base_row, scored, "repair")
            cf = [get_score(r, scored, "repair") for r in members]
            complete = base_score is not None and all(s is not None for s in cf)
            low.append(float(complete and max([base_score, *cf]) <= epsilon))
            delta = max(0.0, max(cf) - base_score) if complete else None
            invariant.append(float(delta is not None and delta <= epsilon))
            groups.append(base_row["prompt_id"])
            if complete:
                worst_delta.append((delta, base_row["prompt_id"]))
        maximum = stats([x[0] for x in worst_delta], [x[1] for x in worst_delta])
        rate_low, rate_invariant = stats(low, groups), stats(invariant, groups)
        output[f"static/{family}"] = {
            "eligible_bases": len(low), "construction_excluded_bases": excluded,
            "absolute_low_rate": rate_low, "nonincrease_rate": rate_invariant,
            "worst_positive_delta": maximum,
            "pass": bool(rate_low["mean"] is not None and rate_low["mean"] >= 0.95
                         and rate_invariant["mean"] >= 0.95 and maximum["ci95"] is not None
                         and maximum["ci95"][1] <= epsilon),
        }
    control = ladder(rows, scored)
    refs = [get_score(r, scored, "repair") for r in rows
            if r["kind"] == "translation" and r["family"] == "clean" and r["motion"] == calibration["medium_motion"]]
    refs = [x for x in refs if x is not None]
    median = float(np.median(refs)) if refs else None
    output["clean_motion"] = {**control, "medium_median": median,
        "pass": bool(median is not None and median >= 0.8 * reference
                     and (control["input_qualified"]["strict_ordering"]["mean"] or 0) >= 0.9)}
    clean_medium = {r["base_id"]: r for r in rows if r["kind"] == "translation"
                    and r["family"] == "clean" and r["motion"] == calibration["medium_motion"]}
    for family in construction["families"]:
        for amplitude in construction["amplitudes"]:
            order = ladder(rows, scored, family=family, amplitude=amplitude)
            groups = defaultdict(list)
            for row in rows:
                if row["kind"] == "translation" and row["family"] == family and row["amplitude"] == amplitude:
                    groups[(row["base_id"], row["seed"])].append(row)
            clean_levels = {(r["base_id"], r["motion"]): r for r in rows
                            if r["kind"] == "translation" and r["family"] == "clean"}
            eligible = {key for key, members in groups.items()
                        if all(r["status"] == "qualified" for r in members)
                        and all(clean_levels.get((r["base_id"], r["motion"]), {}).get("status") == "qualified"
                                for r in members)}
            before, after = [], []
            for row in rows:
                if row["kind"] == "translation" and row["motion"] == calibration["medium_motion"] and row["family"] == family and row["amplitude"] == amplitude:
                    base = clean_medium.get(row["base_id"])
                    a = get_score(base, scored, "repair") if base else None
                    b = get_score(row, scored, "repair")
                    if (row["base_id"], row["seed"]) in eligible and a is not None and b is not None:
                        before.append(a)
                        after.append(b)
            ratio = float(np.mean(after) / np.mean(before)) if before and np.mean(before) > 0 else None
            output[f"noisy_motion/{family}/{amplitude}"] = {**order, "retention_ratio": ratio,
                "matched_count": len(before), "mean_delta": float(np.mean(np.asarray(after) - before)) if before else None,
                "pass": bool(ratio is not None and ratio >= 0.8
                             and (order["input_qualified"]["strict_ordering"]["mean"] or 0) >= 0.9)}
    return output


def analyze(args):
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate construction candidate IDs")
    scored = load_scores(args.scores)
    if args.origin_scores:
        official = load_scores(args.origin_scores)
        for key, item in official.items():
            if "origin" not in item:
                continue
            target = scored.setdefault(key, {"input_sha256": item["input_sha256"]})
            if target["input_sha256"] != item["input_sha256"]:
                raise ValueError("cannot reuse Origin across different video bytes")
            if "origin" in target and target["origin"] != item["origin"]:
                raise ValueError("conflicting Origin results")
            target["origin"] = item["origin"]
    if set(scored) - {r["candidate_id"] for r in rows}:
        raise ValueError("scores contain candidates outside manifest")
    construction = json.loads(Path(args.construction).read_text())
    if "config" in construction:
        construction = construction["config"]
    eligibility = qualification(rows, construction)
    if args.analysis_manifest:
        frozen = json.loads(Path(args.analysis_manifest).read_text())
        if frozen.get("manifest_sha256") != digest(Path(args.manifest)):
            raise ValueError("analysis manifest belongs to different construction bytes")
        if any(frozen.get(k) != v for k, v in eligibility.items()):
            raise ValueError("input eligibility changed since analysis manifest was frozen")
    elif any(r["split"] == "test" for r in rows):
        raise ValueError("test analysis requires a previously frozen input-only --analysis-manifest")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    calibration = None
    if args.calibrate:
        calibration = calibrate(rows, scored, construction)
        calibration["manifest_sha256"] = digest(Path(args.manifest))
        calibration["scores_sha256"] = {str(p): digest(Path(p)) for p in args.scores}
        (output / "calibration.json").write_text(json.dumps(calibration, indent=2))
    elif args.calibration:
        calibration = json.loads(Path(args.calibration).read_text())
    table = paired_tables(rows, scored)
    gates = goal_gates(rows, scored, construction, calibration) if calibration else None
    coverage_pass = all(cell[f"{backend}:full"]["coverage_qualified"] >= 0.95 for cell in table for backend in ("origin", "repair"))
    summary = {"candidate_count": len(rows), "source_count": len({r["base_id"] for r in rows}),
               "score_files_sha256": {str(p): digest(Path(p)) for p in args.scores + (args.origin_scores or [])},
               "prompt_groups": len({r["prompt_id"] for r in rows}), "split": sorted({r["split"] for r in rows}),
               "construction_qualified": sum(r["status"] == "qualified" for r in rows),
               "construction_rejected_or_failed": sum(r["status"] != "qualified" for r in rows),
               "construction_status_counts": eligibility["construction_status_counts"],
               "analysis_population_rule": eligibility["rule"],
               "analysis_manifest_sha256": digest(Path(args.analysis_manifest)) if args.analysis_manifest else None,
               "analysis_code_sha256": digest(Path(__file__)),
               "calibration": calibration, "gates": gates, "coverage_pass": coverage_pass,
               "joint_pass": bool(gates and coverage_pass and all(g["pass"] for g in gates.values())),
               "manifest_sha256": digest(Path(args.manifest)), "tables": table}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Static-jitter paired analysis", "", f"Sources: {summary['source_count']}; prompt clusters: {summary['prompt_groups']}; candidates: {len(rows)}.",
             "", "Repair: short-side lengths/s. Origin: official Boolean mean. CI: prompt-cluster bootstrap; one cluster cannot give a CI.", "",
             "| Control | Motion | Family | Dose | Qualified / candidates | Origin base → CF | Repair base → CF | Repair positive Δ | Repair absolute Δ | Coverage O / R |",
             "|---|---:|---|---:|---:|---|---|---:|---:|---|"]
    def number(x):
        return "NOT RUN" if x is None else f"{x:.6f}"
    for cell in table:
        orig, repair = cell["origin:full"], cell["repair:full"]
        pair = lambda r: number(r["base"]["mean"]) + " → " + number(r["counterfactual"]["mean"])
        lines.append(f"| {cell['kind']} | {cell['motion']} | {cell['family']} | {cell['amplitude']} | {cell['qualified']}/{cell['candidates']} | {pair(orig)} | {pair(repair)} | {number(repair['positive_delta']['mean'])} | {number(repair['absolute_delta']['mean'])} | {orig['scored']} / {repair['scored']} |")
    lines.extend(["", f"Joint gates: {summary['joint_pass']} (development is not independent-test acceptance).", ""])
    (output / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"tables", "gates"}}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--origin-scores", nargs="+", help="reuse only the official backend from byte-identical prior inputs")
    parser.add_argument("--construction", required=True)
    parser.add_argument("--analysis-manifest", help="input-only qualification JSON, mandatory for test analysis")
    parser.add_argument("--output", required=True)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibrate", action="store_true")
    calibration.add_argument("--calibration")
    analyze(parser.parse_args(argv))


if __name__ == "__main__":
    main()
