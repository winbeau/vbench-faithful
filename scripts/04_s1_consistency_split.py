#!/usr/bin/env python3
"""S1 global audit and deterministic prompt-disjoint split for real annotations."""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path

LABELS = {"0", "0.0", "0.5", "1", "1.0"}
PAIR_FIELDS = None

def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=Path("data/processed/pairwise_master.csv"))
    ap.add_argument("--audit-out", type=Path, default=Path("data/processed/audit/global_consistency.csv"))
    ap.add_argument("--main-out", type=Path, default=Path("data/processed/audit/main_dimension_coverage.csv"))
    ap.add_argument("--split-out", type=Path, default=Path("splits/e0_prompt_split.csv"))
    ap.add_argument("--split-master-out", type=Path, default=Path("data/processed/pairwise_master_split.csv"))
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--dev-fraction", type=float, default=.4)
    ns = ap.parse_args(); rows = read_rows(ns.master)
    groups = defaultdict(list); prompts = defaultdict(set)
    for r in rows:
        groups[(r["dimension"], r["instance_id"])].append(r)
        prompts[r["dimension"]].add(r["prompt_id"])

    audit = []
    for (dim, gid), rs in sorted(groups.items()):
        pairs = {(r["model_a"], r["model_b"]) if r["model_a"] < r["model_b"] else (r["model_b"], r["model_a"]) for r in rs}
        model_ids = {r["model_a"] for r in rs} | {r["model_b"] for r in rs}
        bad_self = sum(r["model_a"] == r["model_b"] for r in rs)
        bad_label = sum(r["human_label"] not in LABELS for r in rs)
        ok = len(model_ids) == 4 and len(pairs) == 6 and bad_self == 0 and bad_label == 0 and len(rs) == 6
        audit.append({"scope": "group", "dimension": dim, "group_id": gid, "rows": len(rs), "models": len(model_ids),
                      "unique_pairs": len(pairs), "self_pairs": bad_self, "invalid_labels": bad_label,
                      "pass": int(ok)})
    assert all(x["pass"] for x in audit), "one or more groups failed consistency checks"

    dims = sorted({r["dimension"] for r in rows}); coverage = []
    for dim in dims:
        dr = [r for r in rows if r["dimension"] == dim]
        dg = [x for x in audit if x["dimension"] == dim]
        path_ids = {r["video_a_path"] for r in dr} | {r["video_b_path"] for r in dr}
        video_ids = {r["video_a_id"] for r in dr} | {r["video_b_id"] for r in dr}
        n_groups, n_pairs = len(dg), len(dr)
        assert n_groups * 6 == n_pairs
        coverage.append({"dimension": dim, "groups": n_groups, "pairs": n_pairs,
                         "prompts": len(prompts[dim]), "unique_video_paths": len(path_ids),
                         "unique_video_ids": len(video_ids), "all_groups_four_models": int(all(x["models"] == 4 for x in dg)),
                         "groups_times_6_equals_pairs": int(n_groups * 6 == n_pairs)})
    total_groups, total_pairs = len(audit), len(rows)
    assert total_groups == sum(x["groups"] for x in coverage)
    assert total_groups == 6930, total_groups
    assert total_groups * 6 == total_pairs == 41580
    audit_fields = list(audit[0])
    audit.append({"scope": "global", "dimension": "__ALL__", "group_id": "__ALL__", "rows": total_pairs,
                  "models": 4, "unique_pairs": total_pairs, "self_pairs": sum(x["self_pairs"] for x in audit),
                  "invalid_labels": sum(x["invalid_labels"] for x in audit), "pass": 1})
    write_csv(ns.audit_out, audit, audit_fields)
    write_csv(ns.main_out, [x for x in coverage if x["dimension"] in {"dynamics_degree", "subject_consistency", "human_action", "spatial_relationship"}], list(coverage[0]))

    split_rows = []
    for dim in dims:
        ps = sorted(prompts[dim]); random.Random(ns.seed ^ hash(dim)).shuffle(ps)
        # Avoid Python hash randomization: redo with a stable character-sum seed.
        ps = sorted(prompts[dim]); random.Random(ns.seed + sum((i + 1) * ord(c) for i, c in enumerate(dim))).shuffle(ps)
        n_dev = round(len(ps) * ns.dev_fraction)
        dev = set(ps[:n_dev])
        split_rows.extend({"dimension": dim, "prompt_id": p, "split": "dev" if p in dev else "test", "seed": ns.seed} for p in ps)
    write_csv(ns.split_out, split_rows, ["dimension", "prompt_id", "split", "seed"])
    split_map = {(x["dimension"], x["prompt_id"]): x["split"] for x in split_rows}
    out_rows = []
    for r in rows:
        q = dict(r); q["split"] = split_map[(r["dimension"], r["prompt_id"])]
        out_rows.append(q)
    assert len(out_rows) == len(rows)
    write_csv(ns.split_master_out, out_rows, list(out_rows[0]))
    for dim in dims:
        a = {x["prompt_id"] for x in split_rows if x["dimension"] == dim and x["split"] == "dev"}
        b = {x["prompt_id"] for x in split_rows if x["dimension"] == dim and x["split"] == "test"}
        assert a.isdisjoint(b)
    print(f"PASS groups={total_groups} pairs={total_pairs} dimensions={len(dims)} seed={ns.seed}")

if __name__ == "__main__": main()
