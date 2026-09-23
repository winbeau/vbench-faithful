"""Freeze input-only analysis eligibility; never open a score or model output.

The full construction ledger is retained. A ladder is eligible only when *all*
its levels and clean references pass construction checks. Missing scores on an
eligible ladder are failures, not a reason to change this analysis population.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from .static_jitter import digest, variants


FIELDS = ("kind", "motion", "family", "amplitude", "seed")


def qualification(rows, construction):
    expected = {tuple(v[k] for k in FIELDS) for v in variants(construction)}
    by_source = defaultdict(dict)
    identities = set()
    for row in rows:
        if row["candidate_id"] in identities:
            raise ValueError("duplicate candidate identity")
        identities.add(row["candidate_id"])
        if row["status"] not in {"qualified", "rejected", "construction_failed"}:
            raise ValueError("unknown construction status")
        key = tuple(row[k] for k in FIELDS)
        source = by_source[row["base_id"]]
        if key in source:
            raise ValueError("duplicate intervention within source")
        source[key] = row
    if not by_source:
        raise ValueError("empty candidate ledger")
    groups = []
    for base_id, source in sorted(by_source.items()):
        if set(source) != expected:
            raise ValueError(f"incomplete or unexpected construction grid: {base_id}")
        if len({(r["split"], r["prompt_id"]) for r in source.values()}) != 1:
            raise ValueError("source crosses split/prompt identities")
        first = next(iter(source.values()))

        def add(kind, family, amplitude, seed, members, references):
            combined = {r["candidate_id"]: r for r in members + references}
            exclusions = [{"candidate_id": r["candidate_id"], "status": r["status"],
                           "reason": r.get("reason")} for r in combined.values()
                          if r["status"] != "qualified"]
            groups.append({"base_id": base_id, "prompt_id": first["prompt_id"], "split": first["split"],
                           "kind": kind, "family": family, "amplitude": amplitude, "seed": seed,
                           "candidate_ids": [r["candidate_id"] for r in members],
                           "reference_ids": [r["candidate_id"] for r in references],
                           "eligible": not exclusions, "input_exclusions": exclusions})

        for family in construction["families"]:
            members = [r for r in source.values() if r["kind"] == "static" and r["family"] == family]
            add("static", family, None, None, members, [source[("static", 0, "clean", 0, 0)]])
        for kind in ("translation", "oscillation"):
            ladders = defaultdict(list)
            for row in source.values():
                if row["kind"] == kind:
                    ladders[(row["family"], row["amplitude"], row["seed"])].append(row)
            for (family, amplitude, seed), members in sorted(ladders.items()):
                members.sort(key=lambda r: r["motion"])
                refs = [source[(kind, r["motion"], "clean", 0, 0)] for r in members]
                add(kind, family, amplitude, seed, members, refs)
    return {"schema": "static-jitter-input-qualification-v1", "score_blind": True,
            "rule": "complete input-qualified intervention family or motion ladder and its clean references",
            "candidate_count": len(rows), "source_count": len(by_source),
            "construction_status_counts": dict(Counter(r["status"] for r in rows)),
            "qualified_candidate_ids": sorted(r["candidate_id"] for r in rows if r["status"] == "qualified"),
            "groups": groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--construction", required=True)
    parser.add_argument("--output", required=True, help="new JSON analysis manifest; never overwrites")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    construction = json.loads(Path(args.construction).read_text())
    construction = construction.get("config", construction)
    result = qualification(rows, construction)
    result.update(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  manifest_sha256=digest(Path(args.manifest)),
                  construction_file_sha256=digest(Path(args.construction)),
                  qualifier_sha256=digest(Path(__file__)))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in {"groups", "qualified_candidate_ids"}}))


if __name__ == "__main__":
    main()
