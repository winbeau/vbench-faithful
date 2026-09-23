"""All-region translation hypotheses from completed self-ambiguity diagnostics.

This ablation changes candidate selection, not the underlying image evidence.
Both linear/squared source-ambiguity margins are retained, with disjoint queried
location folds. No final Dynamic score or physical-identity label is produced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.local_appearance import consensus_hypotheses
from .score_static_jitter import source_hashes
from .static_jitter import digest


def inspect_pair(pair, shape):
    grouped = defaultdict(list)
    for point in pair["source_points"]:
        for ref in point["regional_supports"]:
            appearance = pair["supports"][ref["support_sha256"]]
            if "self_alternatives" not in appearance or appearance.get("reverse_check") != "NOT RUN":
                raise ValueError("distinct self-ambiguity evidence required; do not transfer reverse certification")
            grouped[ref["region"], ref["factor"], ref["whole_frame_control"]].append({"xy": point["xy"], "appearance": appearance})
    result = []
    for (region, factor, whole), entries in sorted(grouped.items()):
        result.append({"region": region, "factor": factor, "whole_frame_control": whole,
                       "source_point_scale_groups": len(entries),
                       "ablations": [consensus_hypotheses(entries, shape, intersection=.5, power=power, radius=1) for power in (1, 2)]})
    return {"start": pair["start"], "lag": pair["lag"], "seconds": pair["seconds"], "regions": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "source-execution-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    source, execution, output = map(Path, (args.source_run, args.source_execution_root, args.output))
    if output.exists():
        raise FileExistsError("fresh consensus run required")
    shards = sorted(source.glob("shard-*"))
    identities, rows = [], []
    for shard in shards:
        identity = json.loads((shard / "provenance.json").read_text())
        runtime = json.loads((shard / "runtime.json").read_text())
        if (runtime["status"] != "finished" or runtime["failed"] or runtime["expected"] != runtime["completed"]
                or identity.get("search") != "ambiguity" or digest(shard / "diagnostics.jsonl") != identity["diagnostics_sha256"]):
            raise ValueError("complete hash-bound ambiguity run required")
        for name, sha in {**identity["code_files"], **identity["helper_sha256"],
                          "scripts/counterfactual/probe_local_appearance.py": identity["script_sha256"]}.items():
            if digest(execution / name) != sha:
                raise ValueError("source execution snapshot changed")
        records = list(map(json.loads, (shard / "diagnostics.jsonl").read_text().splitlines()))
        if ([r["candidate_id"] for r in records] != identity["sharding"]["assigned"]
                or len(records) != runtime["completed"] or sum(len(r["pairs"]) for r in records) != runtime["pairs_completed"]):
            raise ValueError("source phase coverage differs")
        identities.append(identity); rows.extend(records)
    if not identities:
        raise ValueError("no source shards")
    reference = identities[0]
    varying = {"sharding", "diagnostics_sha256", "resolved_media"}
    common = {k: v for k, v in reference.items() if k not in varying}
    if (any({k: v for k, v in i.items() if k not in varying} != common for i in identities)
            or sorted(i["sharding"]["shard"] for i in identities) != list(range(reference["sharding"]["shards"]))
            or sorted(r["candidate_id"] for r in rows) != sorted(reference["sharding"]["selected_cohort"])):
        raise ValueError("source shard identities/cohort differ")
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    provenance = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                  "source_provenance_sha256": {s.name: digest(s / "provenance.json") for s in shards},
                  "source_diagnostics_sha256": {s.name: digest(s / "diagnostics.jsonl") for s in shards},
                  "input_sha256": reference["input_sha256"], "selected_cohort": reference["sharding"]["selected_cohort"],
                  "starts": reference["starts"], "lags": reference["lags"], "support_factors": reference["support_factors"],
                  "config": {"self_intersection": .5, "powers": [1, 2], "integer_localization_radius": 1,
                             "hypotheses": 3, "folds": "all/even/odd sorted distinct source locations; patches may overlap"},
                  "script_sha256": digest(Path(__file__)), "code_files": source_hashes(Path(__file__).resolve().parents[2]),
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu"}}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0,
               "failed": 0, "pairs_completed": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def update(**fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    update()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for source_row in sorted(rows, key=lambda r: r["candidate_id"]):
            key = source_row["candidate_id"]
            record = {k: source_row[k] for k in ("candidate_id", "base_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, pairs=[])
            try:
                wanted = [(s, l) for s in reference["starts"] for l in reference["lags"]]
                if source_row["status"] != "diagnostic_only" or source_row["score"] is not None or [(p["start"], p["lag"]) for p in source_row["pairs"]] != wanted:
                    raise ValueError("incomplete source video")
                for pair in source_row["pairs"]:
                    result = inspect_pair(pair, source_row["sampling"]["sampled_shape"][:2])
                    record["pairs"].append(result)
                    update(candidate_id=key, start=pair["start"], pairs_completed=runtime["pairs_completed"] + 1)
                    print(json.dumps({"candidate_id": key, "start": pair["start"], "region_scale_groups": len(result["regions"])}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            update(completed=runtime["completed"] + 1)
    update(status="finished", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    provenance["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
