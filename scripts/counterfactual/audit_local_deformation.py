"""Completeness and conditional statistics for the local deformation ablation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .static_jitter import digest


def point_displacement(fit, xy):
    if fit["status"] != "diagnostic_only":
        return None
    xy = np.asarray(xy, float)
    return (np.asarray(fit["affine_matrix"]) @ np.r_[xy, 1] - xy).tolist()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "proposal-run", "execution-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--case-candidate")
    p.add_argument("--case-source-keys", nargs="+", type=int, default=[])
    args = p.parse_args(argv)
    if bool(args.case_candidate) != bool(args.case_source_keys):
        p.error("posthoc case candidate and source keys must be supplied together")
    source, prior, execution, output = map(Path, (args.source_run, args.proposal_run, args.execution_root, args.output))
    if output.exists():
        raise FileExistsError("fresh analysis required")
    proposals, proposal_hashes, proposal_ids = {}, {}, {}
    for shard in sorted(prior.glob("shard-*")):
        proposal_hashes[shard.name] = digest(shard / "diagnostics.jsonl")
        proposal_ids[shard.name] = digest(shard / "provenance.json")
        for row in map(json.loads, (shard / "diagnostics.jsonl").read_text().splitlines()):
            if row["candidate_id"] in proposals:
                raise ValueError("duplicate proposal candidate")
            proposals[row["candidate_id"]] = row
    stats, runtimes, controls, cases, originals, seen, identities = [], [], [], [], {}, [], []
    for shard in sorted(source.glob("shard-*")):
        identity = json.loads((shard / "provenance.json").read_text())
        runtime = json.loads((shard / "runtime.json").read_text())
        if (runtime["status"] != "finished" or runtime["completed"] != runtime["expected"]
                or identity["source_diagnostics_sha256"] != proposal_hashes
                or identity["source_provenance_sha256"] != proposal_ids
                or identity["diagnostics_sha256"] != digest(shard / "diagnostics.jsonl")):
            raise ValueError("partial/changed input or source run")
        for name, sha in {**identity["code_files"], **identity["helper_sha256"],
                          "scripts/counterfactual/probe_local_deformation.py": identity["script_sha256"]}.items():
            if digest(execution / name) != sha:
                raise ValueError("execution snapshot changed")
        rows = list(map(json.loads, (shard / "diagnostics.jsonl").read_text().splitlines()))
        if ([r["candidate_id"] for r in rows] != identity["sharding"]["assigned"]
                or sum(r["status"] == "failed" for r in rows) != runtime["failed"]):
            raise ValueError("completed records disagree with runtime")
        identities.append(identity); runtimes.append(runtime)
        for row in rows:
            key = row["candidate_id"]
            seen.append(key)
            proposal = proposals[key]
            if row["score"] is not None:
                raise ValueError("diagnostic relabeled as score")
            old_pairs = {(r["start"], r["lag"]): r for r in proposal["pairs"]}
            if row["status"] != "failed" and [(r["start"], r["lag"]) for r in row["pairs"]] != list(old_pairs):
                raise ValueError("missing phase")
            fits, movements = defaultdict(list), defaultdict(list)
            counts = Counter()
            for pair in row["pairs"]:
                old = old_pairs[(pair["start"], pair["lag"])]
                expected = defaultdict(list)
                for index, point in enumerate(old["source_points"]):
                    for ref in point["regional_supports"]:
                        if ref["factor"] in identity["factors"]:
                            expected[ref["support_sha256"]].append({"point": index, **ref})
                if set(pair["supports"]) != set(expected):
                    raise ValueError("missing support in declared factors")
                counts["supports"] += len(expected)
                for skey, support in pair["supports"].items():
                    if support["references"] != expected[skey] or len(support["hypotheses"]) != len(old["supports"][skey]["hypotheses"]):
                        raise ValueError("missing source reference or hypothesis")
                    for candidate, before in zip(support["hypotheses"], old["supports"][skey]["hypotheses"]):
                        if candidate["initial"] != before["displacement_pixels"] or set(candidate["refinements"]) != set(identity["models"]):
                            raise ValueError("proposal or model set changed")
                        for model, fit in candidate["refinements"].items():
                            if fit["score"] is not None:
                                raise ValueError("refinement relabeled as score")
                            fits[model].append(fit)
                    for ref in support["references"]:
                        point = old["source_points"][ref["point"]]
                        if ref["whole_frame_control"] and support["hypotheses"]:
                            # The same ORIGINAL rank-zero proposal in both models;
                            # do not compare fitted losses across differing visible
                            # subsets to cherry-pick a new winning hypothesis.
                            for model, fit in support["hypotheses"][0]["refinements"].items():
                                d = point_displacement(fit, point["xy"])
                                movements[model].append(None if d is None else float(np.linalg.norm(d)))
                        if key == args.case_candidate and set(point["source_keys"]) & set(args.case_source_keys):
                            cases.append({"candidate_id": key, "start": pair["start"], "lag": pair["lag"],
                                          "xy": point["xy"], "source_keys": point["source_keys"], "reference": ref,
                                          "hypotheses": [{"initial": h["initial"], "refinements": {
                                              model: fit | {"query_displacement_pixels": point_displacement(fit, point["xy"])}
                                              for model, fit in h["refinements"].items()}} for h in support["hypotheses"]]})
            models = {}
            for model, model_fits in fits.items():
                observed = [f for f in model_fits if f["status"] == "diagnostic_only"]
                measured = [d for d in movements[model] if d is not None]
                models[model] = {"attempted": len(model_fits), "status_counts": dict(Counter(f["status"] for f in model_fits)),
                                 "converged": sum(bool(f.get("converged")) for f in model_fits),
                                 "rank_deficient": sum(f["jacobian_rank"] != f["parameters"] for f in observed),
                                 "mean_correlation_gain": float(np.mean([f["correlation"] - f["initial_correlation"] for f in observed])) if observed else None,
                                 "whole_frame_top1_points": len(movements[model]), "whole_frame_top1_missing": len(movements[model]) - len(measured),
                                 "whole_frame_top1_conditional_mean_length_pixels": float(np.mean(measured)) if measured else None}
            stats.append({k: row[k] for k in ("candidate_id", "family", "seed", "status")} | {"counts": dict(counts), "models": models})
            signature = hashlib.sha256(json.dumps(row["pairs"], sort_keys=True, allow_nan=False).encode()).hexdigest()
            if row["family"] == "original":
                originals[row["base_id"]] = signature
            elif row["family"] == "encoding_control":
                controls.append({"candidate_id": key, "all_pair_fields_exact": originals.get(row["base_id"]) == signature})
    if not identities:
        raise ValueError("no complete shards")
    shared = {k: v for k, v in identities[0].items() if k not in {"sharding", "diagnostics_sha256"}}
    if (any({k: v for k, v in i.items() if k not in {"sharding", "diagnostics_sha256"}} != shared for i in identities)
            or sorted(i["sharding"]["shard"] for i in identities) != list(range(identities[0]["sharding"]["shards"]))
            or sorted(seen) != sorted(proposals)):
        raise ValueError("incomplete or incompatible cohort")
    if args.case_candidate and not cases:
        raise ValueError("requested posthoc case missing")
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "videos": stats,
              "runtimes": runtimes, "encoding_controls": controls, "posthoc_cases": cases,
              "script_sha256": digest(Path(__file__)),
              "source_provenance_sha256": {s.name: digest(s / "provenance.json") for s in source.glob("shard-*")},
              "source_diagnostics_sha256": {s.name: digest(s / "diagnostics.jsonl") for s in source.glob("shard-*")},
              "scope": "one DEV frame pair and declared scales; conditional adaptive-point statistics are not valid scores"}
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"videos": stats, "controls": controls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
