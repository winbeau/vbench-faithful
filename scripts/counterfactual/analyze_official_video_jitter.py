"""Original official video -> controlled intervention, never synthetic controls.

Development diagnostic only until the corrected protocol is independently
frozen. Origin means fraction classified dynamic, NOT accuracy or static FPR.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .analyze_static_jitter import get_score, load_scores, stats
from .official_video_jitter import interventions
from .score_static_jitter import select_candidates
from .static_jitter import digest


def validate_population(rows, config):
    ids = [r["candidate_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate official-video construction candidates")
    if any(r.get("protocol") != config["protocol"] or r.get("kind") != "official_native" for r in rows):
        raise ValueError("synthetic/staticized inputs cannot enter the official-video analysis")
    if {r["split"] for r in rows} != {"dev"}:
        raise ValueError("corrected independent-test protocol must be frozen before test analysis")
    expected = {(s["family"], s["amplitude"], s["seed"]) for s in interventions(config)}
    sources = defaultdict(list)
    for row in rows:
        if row["status"] == "qualified":
            is_local = (config["protocol"] == "official-video-local-texture-jitter-v1"
                        and row["family"] == "local_texture_alternating")
            expected_map = "bounded_local_displacement_field" if is_local else "identity"
            if is_local and (row.get("geometry_qualified") is not True or row.get("intensity_noise_added") is not False):
                raise ValueError("local texture jitter lacks geometry verification or adds intensity noise")
            if (row.get("frame_map") != "identity_all_source_frames" or row.get("coordinate_map") != expected_map
                    or not row.get("native_timeline_preserved") or not row.get("pixel_exact_to_intended")
                    or row.get("decoded_shape") != row.get("source_shape")):
                raise ValueError("qualified input lacks verified original-video geometry/timing/pixels")
            if row["family"] == "original" and row.get("source_format_adaptation") == "none":
                if row["video"] != row["source_video"] or row["sha256"] != row["source_sha256"]:
                    raise ValueError("original MP4 baseline was replaced instead of scored directly")
        sources[row["base_id"]].append(row)
    for base_id, members in sources.items():
        observed = [(r["family"], r["amplitude"], r["seed"]) for r in members]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError(f"incomplete official-video candidate ledger: {base_id}")
        if len({r["prompt_id"] for r in members}) != 1:
            raise ValueError("source has inconsistent prompt identity")
    return sources


def diagnostic_score(row, scored, backend, variant="full", *, stress_test=False):
    if not stress_test or row["status"] != "rejected":
        return get_score(row, scored, backend, variant)
    # Validation is explicit; neither the ledger nor its qualification is changed.
    select_candidates([row], stress_test=True)
    record = scored.get(row["candidate_id"], {})
    if record and record["input_sha256"] != row.get("sha256"):
        raise ValueError("manifest/stress score input hash mismatch")
    value = record.get(backend, {})
    if value.get("status") != "succeeded":
        return None
    return value.get("ablations", {}).get(variant) if backend == "repair" and variant != "full" else value.get("score")


def tables(rows, scored, *, stress_test=False):
    baseline = {r["base_id"]: r for r in rows if r["family"] == "original"}
    cells = defaultdict(list)
    for row in rows:
        cells[(row["family"], row["amplitude"])].append(row)
    output = []
    for (family, amplitude), members in sorted(cells.items()):
        cell = {"family": family, "amplitude": amplitude, "candidate_count": len(members),
                "construction_status_counts": dict(Counter(r["status"] for r in members)),
                "qualified": sum(r["status"] == "qualified" for r in members)}
        eligible = select_candidates(members, stress_test=stress_test)
        cell.update(scoring_population=len(eligible), includes_quality_rejections=stress_test)
        for backend, variant in (("origin", "full"), ("repair", "full"),
                                 ("repair", "raw_tracks"), ("repair", "verified_correspondence")):
            before, after, groups, cases, statuses = [], [], [], [], Counter()
            for row in members:
                if row in eligible:
                    statuses[scored.get(row["candidate_id"], {}).get(backend, {}).get("status", "NOT RUN")] += 1
                a = diagnostic_score(baseline[row["base_id"]], scored, backend, variant, stress_test=stress_test)
                b = diagnostic_score(row, scored, backend, variant, stress_test=stress_test)
                if a is None or b is None:
                    continue
                before.append(a)
                after.append(b)
                groups.append(row["prompt_id"])
                cases.append({"base_id": row["base_id"] , "candidate_id": row["candidate_id"],
                              "base": a, "counterfactual": b, "delta": b - a})
            delta = np.asarray(after) - np.asarray(before)
            value = {"paired_count": len(before), "scoring_status_counts": dict(statuses),
                     "coverage_qualified": (sum(r["status"] == "qualified" and
                         scored.get(r["candidate_id"], {}).get(backend, {}).get("status") == "succeeded" for r in members)
                         / cell["qualified"] if cell["qualified"] else 0),
                     "coverage_scoring_population": statuses["succeeded"] / len(eligible) if eligible else 0,
                     "base": stats(before, groups), "counterfactual": stats(after, groups),
                     "signed_delta": stats(delta.tolist(), groups),
                     "absolute_delta": stats(np.abs(delta).tolist(), groups),
                     "positive_delta": stats(np.maximum(delta, 0).tolist(), groups),
                     "cases": cases}
            if backend == "origin":
                value.update(zero_to_one=sum(a == 0 and b == 1 for a, b in zip(before, after)),
                             one_to_zero=sum(a == 1 and b == 0 for a, b in zip(before, after)),
                             decision_change_rate=stats((delta != 0).astype(float).tolist(), groups))
            cell[f"{backend}:{variant}"] = value
        output.append(cell)
    return output


def case_table(rows, scored, *, stress_test=False):
    """All cases, including missing values; no winner selection or replacement bases."""
    baseline = {r["base_id"]: r for r in rows if r["family"] == "original"}
    output = []
    for row in rows:
        base = baseline[row["base_id"]]
        item = {key: row.get(key) for key in ("candidate_id", "base_id", "prompt_id", "generator",
                                              "family", "amplitude", "seed", "status", "reason")}
        item.update(source_fps=row.get("source_fps"), source_shape=row.get("source_shape"),
                    source_duration_seconds=row.get("source_duration_seconds"),
                    minimum_warp_jacobian=row.get("minimum_warp_jacobian"),
                    geometry_qualified=row.get("geometry_qualified"),
                    min_structure_correlation=row.get("min_structure_correlation"))
        for backend in ("origin", "repair"):
            a = diagnostic_score(base, scored, backend, stress_test=stress_test)
            b = diagnostic_score(row, scored, backend, stress_test=stress_test)
            before = scored.get(base["candidate_id"], {}).get(backend, {})
            after = scored.get(row["candidate_id"], {}).get(backend, {})
            item.update({f"{backend}_base": a, f"{backend}_cf": b,
                         f"{backend}_delta": b - a if a is not None and b is not None else None,
                         f"{backend}_base_status": before.get("status", "NOT RUN"),
                         f"{backend}_cf_status": after.get("status", "NOT RUN")})
            if backend == "repair":
                item.update(repair_base_coverage=before.get("coverage"), repair_cf_coverage=after.get("coverage"))
        phase = row.get("temporal_phase")
        if phase is not None:
            o = scored.get(row["candidate_id"], {}).get("origin", {}).get("diagnostics", {})
            r = scored.get(row["candidate_id"], {}).get("repair", {})
            for backend, indices in (("origin", o.get("sampling", {}).get("sampled_source_frame_indices")),
                                     ("repair", r.get("sampling", {}).get("source_indices"))):
                if indices:
                    sampled = np.asarray(phase)[indices]
                    item[f"{backend}_sampled_jitter_phase"] = sampled.tolist()
                    item[f"{backend}_sampled_phase_changes"] = int(np.count_nonzero(np.diff(sampled)))
        output.append(item)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stress-test", action="store_true", help="show DEV quality-rejected scores separately; not invariance acceptance")
    parser.add_argument("--review", help="exact, hash-bound development review used by the scorer")
    args = parser.parse_args(argv)
    if args.review and (args.stress_test or len(args.manifests) != 1):
        parser.error("review analysis requires one bound manifest and no stress-test flag")
    config = json.loads(Path(args.config).read_text())
    rows = [json.loads(line) for p in args.manifests for line in Path(p).read_text().splitlines()]
    sources = validate_population(rows, config)
    select_candidates(rows, stress_test=args.stress_test)
    reviewed_rows = None
    if args.review:
        from .review_selection import select_reviewed_candidates
        reviewed_rows = select_reviewed_candidates(rows, Path(args.review), Path(args.manifests[0]),
                                                   Path(__file__).resolve().parents[2])
    scored = load_scores(args.scores)
    if set(scored) - {r["candidate_id"] for r in rows}:
        raise ValueError("score contains candidates outside the native-video ledger")
    identities = []
    for path in args.scores:
        provenance = json.loads(Path(path).with_name("provenance.json").read_text())
        if args.stress_test and not provenance.get("stress_test_includes_quality_rejections"):
            raise ValueError("stress analysis requires explicitly labeled stress scoring provenance")
        if args.review and provenance.get("review_sha256") != digest(Path(args.review)):
            raise ValueError("analysis review differs from the scoring review")
        identities.append({k: provenance[k] for k in ("config_sha256", "tracker_weight_sha256", "raft_weight_sha256",
                                                      "upstream_sha", "backend", "code_files")})
    if any(identity != identities[0] for identity in identities):
        raise ValueError("cannot pool different method/model/source identities")
    table = tables(rows, scored)
    stress_table = tables(rows, scored, stress_test=True) if args.stress_test else None
    review_table = tables(reviewed_rows, scored, stress_test=True) if args.review else None
    cases = case_table(reviewed_rows if args.review else rows, scored, stress_test=args.stress_test or bool(args.review))
    if args.review:
        for case in cases:
            case["user_review_selected"] = True
    summary = {"status": "development_stress_including_quality_rejections" if args.stress_test else "development_diagnostic_only",
               "protocol": config["protocol"], "stress_test": args.stress_test, "config_sha256": digest(Path(args.config)),
               "source_count": len(sources), "prompt_groups": len({r["prompt_id"] for r in rows}),
               "candidate_count": len(rows), "construction_status_counts": dict(Counter(r["status"] for r in rows)),
               "manifest_sha256": {p: digest(Path(p)) for p in args.manifests},
               "scores_sha256": {p: digest(Path(p)) for p in args.scores},
               "analysis_sha256": digest(Path(__file__)), "method_identity": identities[0],
               "analysis_dependency_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                    ("analyze_static_jitter.py", "score_static_jitter.py", "static_jitter.py", "official_video_jitter.py")},
               "epsilon": None, "joint_acceptance": "NOT EVALUATED: corrected protocol not yet frozen",
               "tables": table, "stress_tables": stress_table, "cases": cases}
    if args.review:
        summary.update(status="user_reviewed_development_cohort_not_repair_acceptance",
                       review_sha256=digest(Path(args.review)), reviewed_tables=review_table)
        summary["analysis_dependency_sha256"]["review_selection.py"] = digest(Path(__file__).with_name("review_selection.py"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    columns = list(dict.fromkeys(key for case in cases for key in case))
    with (output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cases)
    def number(x):
        return "—" if x is None else f"{x:.6f}"
    lines = ["# Official native-video jitter: development diagnostic", "",
             "Base = original official MP4 (or an explicitly qualified lossless GIF format adapter).",
             "Origin is a Boolean decision, repair is short-side lengths/s; their scales differ.",
             f"Sources: {summary['source_count']}; prompts: {summary['prompt_groups']}; candidates: {len(rows)}.",
             "Development only. CIs resample prompts, not same-source doses/seeds; five prompts remain a small pilot.", "",
             "A dash means no valid paired score, not zero or necessarily NOT RUN; exact failure/insufficient/missing counts are in summary.json.", "",
             ("STRESS TEST: tables below include quality-rejected (nonfolding, verified) warps. Rejection statuses and gates are unchanged; "
              "qualified-only statistics are in summary.json tables, all-stress statistics in stress_tables. This is NOT invariance acceptance."
              if args.stress_test else "USER-REVIEWED DEV COHORT: all listed accepted interventions enter the denominator. Automatic flags remain historical. This is not independent validation or proof of Repair success."
              if args.review else "Tables below include only construction-qualified pairs."), "",
             "| Family | Dose | Automatic-qualified / candidates | Origin base → CF | Origin flips 0→1 / 1→0 | Repair base → CF | Repair mean absolute Δ | Paired O / R |",
             "|---|---:|---:|---|---|---|---:|---|"]
    for cell in (review_table if args.review else stress_table if args.stress_test else table):
        origin, repair = cell["origin:full"], cell["repair:full"]
        pair = lambda x: number(x["base"]["mean"]) + " → " + number(x["counterfactual"]["mean"])
        lines.append(f"| {cell['family']} | {cell['amplitude']} | {cell['qualified']}/{cell['candidate_count']} | "
                     f"{pair(origin)} | {origin['zero_to_one']} / {origin['one_to_zero']} | {pair(repair)} | "
                     f"{number(repair['absolute_delta']['mean'])} | {origin['paired_count']} / {repair['paired_count']} |")
    if "primary_amplitude" in config and "primary_seed" in config:
        lines += ["", f"## Predeclared primary: {config['primary_amplitude']} native px, seed {config['primary_seed']}", "",
                  "| Official source prompt | Generator | Origin base → CF | Repair base → CF | Repair CF status | Construction status |",
                  "|---|---|---|---|---|---|"]
        for case in cases:
            if case["amplitude"] != config["primary_amplitude"] or case["seed"] != config["primary_seed"]:
                continue
            lines.append(f"| {case['prompt_id']} | {case['generator']} | "
                         f"{number(case['origin_base'])} → {number(case['origin_cf'])} | "
                         f"{number(case['repair_base'])} → {number(case['repair_cf'])} | {case['repair_cf_status']} | {case['status']} |")
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"tables", "stress_tables", "reviewed_tables", "cases", "method_identity"}}, indent=2))


if __name__ == "__main__":
    main()
