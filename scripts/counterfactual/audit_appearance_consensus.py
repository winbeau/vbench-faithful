"""Independent vote/peak/cohort checks of regional appearance proposals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .static_jitter import digest


def votes_at(entries, candidates, *, intersection, power, radius, fold):
    """Direct max-per-feature evaluation, independent of rasterized Hough code."""
    grouped = defaultdict(list)
    for entry in entries:
        grouped[tuple(map(float, entry["xy"]))].append(entry["appearance"])
    positions = np.array(sorted(grouped), float).reshape(-1, 2)
    values = np.zeros((len(candidates), len(positions)))
    for index, xy in enumerate(map(tuple, positions)):
        if fold != "all" and index % 2 != (fold == "odd_location"):
            continue
        scales = grouped[xy]
        for record in scales:
            alternative = next((a["hypotheses"] for a in record["self_alternatives"] if a["maximum_support_intersection"] == intersection), [])
            if not alternative or not record["hypotheses"]:
                continue
            displacement = np.array([p["displacement_pixels"] for p in record["hypotheses"]])
            margins = np.maximum(np.array([p["correlation"] for p in record["hypotheses"]]) - alternative[0]["correlation"], 0) ** power
            close = np.max(abs(np.asarray(candidates)[:, None] - displacement[None]), axis=-1) <= radius
            values[:, index] += np.max(np.where(close, margins, 0), axis=1) / len(scales)
    return values, positions


def verify_consensus(result, entries):
    for fold in result["folds"]:
        for peak in fold["peaks"]:
            center = np.array(peak["displacement_pixels"])
            yy, xx = np.mgrid[-1:2, -1:2]
            neighbors = center + np.c_[xx.ravel(), yy.ravel()]
            values, positions = votes_at(entries, neighbors, intersection=result["intersection"], power=result["power"],
                                         radius=result["integer_localization_radius"], fold=fold["fold"])
            votes = values.sum(axis=1)
            points = positions[values[4] > 0]
            rank = int(np.linalg.matrix_rank(np.c_[np.ones(len(points)), points - points.mean(axis=0)])) if len(points) else 0
            if (not np.isclose(votes[4], peak["vote_weight"], rtol=1e-12, atol=1e-12)
                    or votes[4] < votes.max() - 1e-11 or points.tolist() != peak["source_positions"]
                    or len(points) != peak["supporting_locations"] or rank != peak["spatial_design_rank"]):
                raise ValueError("reported consensus peak, supporters or weight differs")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "appearance-run", "execution-root", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args(argv)
    source, parent, execution, output = map(Path, (args.source_run, args.appearance_run, args.execution_root, args.output))
    if output.exists():
        raise FileExistsError("fresh analysis required")
    identity = json.loads((source / "provenance.json").read_text())
    runtime = json.loads((source / "runtime.json").read_text())
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    if (runtime["status"] != "finished" or runtime["failed"] or runtime["completed"] != runtime["expected"]
            or len(rows) != runtime["expected"] or sorted(r["candidate_id"] for r in rows) != sorted(identity["selected_cohort"])
            or digest(source / "diagnostics.jsonl") != identity["diagnostics_sha256"]):
        raise ValueError("incomplete or changed consensus run")
    for name, sha in {**identity["code_files"], "scripts/counterfactual/probe_appearance_consensus.py": identity["script_sha256"]}.items():
        if digest(execution / name) != sha:
            raise ValueError("execution snapshot changed")
    source_rows = {}
    for shard, sha in identity["source_provenance_sha256"].items():
        if (digest(parent / shard / "provenance.json") != sha
                or digest(parent / shard / "diagnostics.jsonl") != identity["source_diagnostics_sha256"][shard]):
            raise ValueError("appearance source changed")
        for row in map(json.loads, (parent / shard / "diagnostics.jsonl").read_text().splitlines()):
            if row["candidate_id"] in source_rows:
                raise ValueError("duplicate appearance candidate")
            source_rows[row["candidate_id"]] = row
    originals, controls, stats = {}, [], []
    for row in rows:
        key = row["candidate_id"]
        original = source_rows[key]
        wanted = [(s, l) for s in identity["starts"] for l in identity["lags"]]
        if (row["status"] != "diagnostic_only" or row["score"] is not None
                or [(p["start"], p["lag"]) for p in row["pairs"]] != wanted
                or [(p["start"], p["lag"]) for p in original["pairs"]] != wanted):
            raise ValueError("phase or null-score contract changed")
        count = 0
        for measured, inputs in zip(row["pairs"], original["pairs"]):
            grouped = defaultdict(list)
            for point in inputs["source_points"]:
                for ref in point["regional_supports"]:
                    grouped[ref["region"], ref["factor"], ref["whole_frame_control"]].append(
                        {"xy": point["xy"], "appearance": inputs["supports"][ref["support_sha256"]]})
            if [(r["region"], r["factor"], r["whole_frame_control"]) for r in measured["regions"]] != sorted(grouped):
                raise ValueError("missing or added region-scale group")
            for region in measured["regions"]:
                entries = grouped[region["region"], region["factor"], region["whole_frame_control"]]
                if region["source_point_scale_groups"] != len(entries) or [a["power"] for a in region["ablations"]] != [1, 2]:
                    raise ValueError("point or ablation count changed")
                for ablation in region["ablations"]:
                    if ablation["score"] is not None:
                        raise ValueError("proposal promoted to a score")
                    verify_consensus(ablation, entries)
                    count += 1
        if row["family"] == "original":
            originals[row["base_id"]] = row["pairs"]
        elif row["family"] == "encoding_control":
            equal = originals[row["base_id"]] == row["pairs"]
            if not equal:
                raise ValueError("encoding control differs")
            controls.append({"candidate_id": key, "all_pair_fields_exact": equal})
        stats.append({"candidate_id": key, "pairs": len(row["pairs"]), "region_ablation_groups_checked": count})
    output.mkdir(parents=True)
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "runtime": runtime,
              "source_provenance_sha256": digest(source / "provenance.json"), "source_diagnostics_sha256": identity["diagnostics_sha256"],
              "script_sha256": digest(Path(__file__)), "videos": stats, "encoding_controls": controls,
              "warning": "vote arithmetic, local peaks and phase coverage verified; physical correspondence remains unverified"}
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"videos": len(rows), "pairs": sum(s["pairs"] for s in stats), "controls": controls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
