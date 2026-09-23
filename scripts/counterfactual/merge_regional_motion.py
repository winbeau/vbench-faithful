"""Strict merge of complete regional diagnostic shards; not score acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .static_jitter import digest


def merge_runs(inputs):
    common = None
    rows, provenance = {}, []
    seen_shards = set()
    for folder in map(Path, inputs):
        identity = json.loads((folder / "provenance.json").read_text())
        runtime = json.loads((folder / "runtime.json").read_text())
        sharding = identity["sharding"]
        invariant = {k: identity[k] for k in ("config_sha256", "code_files", "script_sha256",
            "feature_provenance_sha256", "region_provenance_sha256", "input_sha256")}
        invariant.update(full_cohort=sharding["full_cohort"], shards=sharding["shards"])
        if common is None:
            common = invariant
        if invariant != common or sharding["shard"] in seen_shards:
            raise ValueError("duplicate or incompatible shard identity")
        seen_shards.add(sharding["shard"])
        if runtime["status"] != "finished" or runtime["completed"] != runtime["expected"]:
            raise ValueError("incomplete shard; never merge a running/stale output as complete")
        records = [json.loads(s) for s in (folder / "diagnostics.jsonl").read_text().splitlines()]
        if ([r["candidate_id"] for r in records] != sharding["assigned"]
                or sharding["assigned"] != sharding["full_cohort"][sharding["shard"]::sharding["shards"]]
                or len(records) != runtime["expected"]
                or sum(r["status"] == "failed" for r in records) != runtime["failed"]):
            raise ValueError("shard row assignment/count/status mismatch")
        for row in records:
            key = row["candidate_id"]
            if key in rows or row["score"] is not None or row["status"] not in {"diagnostic_only", "failed"}:
                raise ValueError("duplicate row or invalid diagnostic status/score")
            if row["status"] != "failed":
                n = len(row["sampling"]["timestamps"])
                expected = [(start, lag) for lag in identity["config"]["lags"] for start in range(n - lag)]
                if [(p["start"], p["lag"]) for p in row["pairs"]] != expected:
                    raise ValueError("missing/duplicate/reordered lag phase")
                for pair in row["pairs"]:
                    items = pair["regions"]
                    if (not items or [x["region"] for x in items] != list(range(len(items)))
                            or [x["whole_frame_control"] for x in items] != [False] * (len(items) - 1) + [True]):
                        raise ValueError("region identities or whole-frame control missing")
            rows[key] = row
        provenance.append({"path": str(folder), "provenance_sha256": digest(folder / "provenance.json"),
                           "runtime_sha256": digest(folder / "runtime.json"),
                           "diagnostics_sha256": digest(folder / "diagnostics.jsonl"), "runtime": runtime})
    if not common or seen_shards != set(range(common["shards"])) or set(rows) != set(common["full_cohort"]):
        raise ValueError("all assigned shards and every cohort member required")
    ordered = [rows[key] for key in common["full_cohort"]]
    return ordered, {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                     "common_identity": common, "shards": provenance, "completed": len(ordered),
                     "failed": sum(r["status"] == "failed" for r in ordered)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs='+', required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows, provenance = merge_runs(args.inputs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostics.jsonl").write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    provenance.update(script_sha256=digest(Path(__file__)), diagnostics_sha256=digest(output / "diagnostics.jsonl"))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"completed": len(rows), "failed": provenance["failed"], "status": "diagnostic_only"}))
    return 1 if provenance["failed"] else 0


if __name__ == '__main__':
    raise SystemExit(main())
