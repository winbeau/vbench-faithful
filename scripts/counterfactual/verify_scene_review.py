"""Re-verification of the `scene` (environment_coverage) conclusions.

CPU only, on the frozen trees.  Recomputes every number the review and the paper
material quote, without importing `run_dimension.py` for the statistics:

  * the official score grid: is every clip on the `k/16` lattice (upstream
    `num_frames=16` mean-of-booleans)?  what is the `k/16` histogram?
  * the reference positive: what does the official metric score on
    `coverage_100`, which is the untouched base video (re-encoded)?
  * the decision decomposition that decides the row: how many test pairs the
    official metric resolves, how many it ties, the accuracy on the resolved
    pairs, and whether the two backends ever disagree in opposite directions;
  * CPA, per-rank-gap census, per-base strict order and the per-base slope for
    both backends, plus the paired `base_id` cluster bootstrap of the delta;
  * the transformation's own census: quadrant prefix, `target_fraction`, donor
    determinism and provenance completeness.

Nothing is written into the frozen trees.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# The probe is often copied to the scoring host and run from /tmp, so the repo
# root is resolved defensively: walk up until `scripts/counterfactual` is found,
# and let `--repo` override it.
def _find_repo() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts" / "counterfactual").is_dir():
            return candidate
    return Path("/root/wenbiao_zhao/vbench-audit")


ROOT = _find_repo()
sys.path.insert(0, str(ROOT / "scripts"))

LEVELS = ["coverage_000", "coverage_025", "coverage_050", "coverage_075", "coverage_100"]
RANK = {level: index for index, level in enumerate(LEVELS)}
QUADRANTS = ["top_left", "top_right", "bottom_left", "bottom_right"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench"))
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    scores_root = args.scores or (args.root / "scores")

    rows = [row for row in read_jsonl(args.root / "manifest.jsonl") if row["dimension"] == "scene"]
    tables = {
        backend: {row["derived_id"]: row for row in read_jsonl(scores_root / f"scene__{backend}.jsonl")}
        for backend in ("official", "repair")
    }
    by_base: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_base[row["base_id"]][row["level"]] = row

    print(f"manifest rows: {len(rows)}  bases: {len(by_base)}  "
          f"official: {len(tables['official'])}  repair: {len(tables['repair'])}")
    failed = [row["derived_id"] for row in rows
              if tables["official"][row["derived_id"]]["status"] != "succeeded"
              or tables["repair"][row["derived_id"]]["status"] != "succeeded"]
    print(f"failed clips: {len(failed)}")

    # ---------------------------------------------------------------- the lattice
    print("\n[1] official score lattice")
    off_grid = [row["derived_id"] for row in rows
                if abs(tables["official"][row["derived_id"]]["score"] * 16
                       - round(tables["official"][row["derived_id"]]["score"] * 16)) > 1e-9]
    histogram: Counter[int] = Counter(
        round(tables["official"][row["derived_id"]]["score"] * 16) for row in rows
    )
    print(f"    clips off the k/16 lattice: {len(off_grid)}/{len(rows)}")
    print("    histogram: " + ", ".join(f"{k}/16={histogram[k]}" for k in sorted(histogram)))

    # ---------------------------------------------------------------- reference positive
    print("\n[2] the reference positive (coverage_100 == the base video, re-encoded)")
    for split in ("dev", "test", "all"):
        subset = [row for row in rows if row["level"] == "coverage_100"
                  and (split == "all" or row["split"] == split)]
        values = [tables["official"][row["derived_id"]]["score"] for row in subset]
        print(f"    official {split:>4}: n={len(values)} mean={statistics.fmean(values):.4f} "
              f"median={statistics.median(values):.4f} "
              f"exactly-0={sum(1 for v in values if v == 0.0)} "
              f"perfect-16/16={sum(1 for v in values if v >= 1.0)}")
    zeros = [row["derived_id"] for row in rows
             if row["level"] == "coverage_000"
             and tables["official"][row["derived_id"]]["score"] != 0.0]
    print(f"    official on coverage_000 above zero: {len(zeros)}/{sum(1 for r in rows if r['level'] == 'coverage_000')}")
    for backend in ("official", "repair"):
        values = [tables[backend][row["derived_id"]]["score"] for row in rows]
        print(f"    {backend} span over all 125 clips: {min(values):.4f} .. {max(values):.4f} "
              f"(range {max(values) - min(values):.4f})")

    # ---------------------------------------------------------------- decision decomposition
    print("\n[3] decision decomposition (test, zero margin)")
    dec: dict[str, dict[str, int]] = {}
    for backend in ("official", "repair"):
        ties = up = down = 0
        for base_id, levels in by_base.items():
            if levels["coverage_000"]["split"] != "test":
                continue
            for low, high in itertools.combinations(LEVELS, 2):
                delta = (tables[backend][levels[high]["derived_id"]]["score"]
                         - tables[backend][levels[low]["derived_id"]]["score"])
                if delta > 0:
                    up += 1
                elif delta < 0:
                    down += 1
                else:
                    ties += 1
        dec[backend] = {"up": up, "down": down, "tie": ties,
                        "total": up + down + ties}
        decided = up + down
        print(f"    {backend}: decided={decided}/{up + down + ties} ties={ties} "
              f"correct={up} wrong={down} CPA={up / (up + down + ties):.4f} "
              f"accuracy-on-decided={up / decided:.4f} "
              f"tie-credited={(up + ties) / (up + down + ties):.4f}")

    print("    per rank gap (test):")
    for backend in ("official", "repair"):
        gaps: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        for base_id, levels in by_base.items():
            if levels["coverage_000"]["split"] != "test":
                continue
            for low, high in itertools.combinations(LEVELS, 2):
                delta = (tables[backend][levels[high]["derived_id"]]["score"]
                         - tables[backend][levels[low]["derived_id"]]["score"])
                gap = RANK[high] - RANK[low]
                gaps[gap][0] += 1
                gaps[gap][1] += 1 if delta == 0 else 0
                gaps[gap][2] += 1 if delta > 0 else 0
        for gap in sorted(gaps):
            total, tied, correct = gaps[gap]
            decided = total - tied
            print(f"      {backend} gap {gap}: n={total} tie_rate={tied / total:.3f} "
                  f"decided={decided} correct={correct} "
                  f"acc-on-decided={correct / decided if decided else float('nan'):.4f}")

    print("    paired comparison on the pairs official resolves:")
    both_right = both_wrong = off_only = rep_only = 0
    for base_id, levels in by_base.items():
        if levels["coverage_000"]["split"] != "test":
            continue
        for low, high in itertools.combinations(LEVELS, 2):
            do = (tables["official"][levels[high]["derived_id"]]["score"]
                  - tables["official"][levels[low]["derived_id"]]["score"])
            if do == 0:
                continue
            dr = (tables["repair"][levels[high]["derived_id"]]["score"]
                  - tables["repair"][levels[low]["derived_id"]]["score"])
            if do > 0 and dr > 0:
                both_right += 1
            elif do > 0 and dr <= 0:
                off_only += 1
            elif do < 0 and dr > 0:
                rep_only += 1
            else:
                both_wrong += 1
    print(f"      both correct={both_right} both wrong={both_wrong} "
          f"official-only={off_only} repair-only={rep_only}")

    # ---------------------------------------------------------------- per-base order
    print("\n[4] per-base order statistics")
    for backend in ("official", "repair"):
        for split in ("dev", "test"):
            bases = [b for b, levels in by_base.items() if levels["coverage_000"]["split"] == split]
            constant = [b for b in bases
                        if len({tables[backend][by_base[b][level]["derived_id"]]["score"]
                                for level in LEVELS}) == 1]
            strict = 0
            violations = 0
            for base_id in bases:
                bad = [(low, high) for low, high in itertools.combinations(LEVELS, 2)
                       if (tables[backend][by_base[base_id][high]["derived_id"]]["score"]
                           <= tables[backend][by_base[base_id][low]["derived_id"]]["score"])]
                violations += len(bad)
                strict += 0 if bad else 1
            slopes = [tables[backend][by_base[b]["coverage_100"]["derived_id"]]["score"]
                      - tables[backend][by_base[b]["coverage_000"]["derived_id"]]["score"]
                      for b in bases]
            print(f"    {backend} {split:>4}: n={len(bases)} constant-profile bases={len(constant)} "
                  f"strict-order={strict}/{len(bases)} violating-pairs={violations} "
                  f"mean-0->100 slope={statistics.fmean(slopes):+.4f} "
                  f"min={min(slopes):+.4f} negatives={sum(1 for s in slopes if s < 0)}")

    # ---------------------------------------------------------------- paired delta
    print("\n[5] paired delta over base_id clusters (test)")
    rng = np.random.default_rng(args.seed)

    def per_base(backend: str) -> dict[str, tuple[int, int]]:
        out = {}
        for base_id, levels in by_base.items():
            if levels["coverage_000"]["split"] != "test":
                continue
            correct = total = 0
            for low, high in itertools.combinations(LEVELS, 2):
                delta = (tables[backend][levels[high]["derived_id"]]["score"]
                         - tables[backend][levels[low]["derived_id"]]["score"])
                total += 1
                correct += 1 if delta > 0 else 0
            out[base_id] = (correct, total)
        return out

    official_pairs, repair_pairs = per_base("official"), per_base("repair")
    keys = sorted(official_pairs)
    deltas = []
    for _ in range(args.iterations):
        pick = rng.choice(keys, size=len(keys), replace=True)
        official_cpa = sum(official_pairs[k][0] for k in pick) / sum(official_pairs[k][1] for k in pick)
        repair_cpa = sum(repair_pairs[k][0] for k in pick) / sum(repair_pairs[k][1] for k in pick)
        deltas.append(repair_cpa - official_cpa)
    deltas_array = np.array(deltas)
    point = (sum(v[0] for v in repair_pairs.values()) / sum(v[1] for v in repair_pairs.values())
             - sum(v[0] for v in official_pairs.values()) / sum(v[1] for v in official_pairs.values()))
    print(f"    delta={point:+.4f} 95% CI=[{np.quantile(deltas_array, 0.025):+.4f}, "
          f"{np.quantile(deltas_array, 0.975):+.4f}]  clusters={len(keys)}")

    # ---------------------------------------------------------------- transformation census
    print("\n[6] transformation census and provenance")
    problems = 0
    donors: dict[str, list[str]] = defaultdict(list)
    for base_id, levels in by_base.items():
        donor = levels["coverage_000"]["transformation_parameters"]["donor"]
        donors[donor["video_uid"]].append(base_id)
        for key in ("video_uid", "prompt_id", "relative_video_path", "generator"):
            if not donor.get(key):
                problems += 1
        if list(levels["coverage_000"]["transformation_parameters"]["quadrant_order"]) != QUADRANTS:
            problems += 1
        for index, level in enumerate(LEVELS):
            parameters = levels[level]["transformation_parameters"]
            if parameters["target_quadrants"] != QUADRANTS[:index]:
                problems += 1
            if abs(parameters["target_fraction"] - index / 4.0) != 0:
                problems += 1
            if levels[level]["expected_rank"] != index:
                problems += 1
    print(f"    provenance/quadrant problems: {problems}")
    print(f"    distinct donors: {len(donors)} over {len(by_base)} bases; "
          f"reused: {sum(1 for v in donors.values() if len(v) > 1)}")
    print(f"    rows recording a donor content hash: "
          f"{sum(1 for row in rows if 'sha256' in json.dumps(row['transformation_parameters'].get('donor', {})))}")
    reencoded = sum(1 for row in rows if row["level"] == "coverage_100"
                    and row["output_sha256"] != row["source_video_sha256"])
    print(f"    coverage_100 rows that are a fresh encode (not a byte copy): {reencoded}/25")
    print(f"    reference level for expected_relation: coverage_100 "
          f"(FAMILY_REFERENCE, build.py:56)")

    # ---------------------------------------------------------------- generator fixes
    print("\n[7] review generator defects: fixed?  (code, not the commit message)")
    source = (ROOT / "scripts" / "counterfactual" / "run_dimension.py").read_text(encoding="utf-8")
    report = (ROOT / "docs" / "counterfactual-reports" / "scene.md").read_text(encoding="utf-8")

    def line_of(needle: str) -> int:
        for index, text in enumerate(source.splitlines(), start=1):
            if needle in text:
                return index
        return -1

    checks = [
        ("contract_split_cpa receives test_rows only",
         'for split, split_rows in (("dev", dev_rows), ("test", test_rows))' in source,
         line_of("split: contract_split_cpa(")),
        ("score_profile is computed per split",
         '"dev": score_profile(dev_rows, backend_scores),' in source,
         line_of('"dev": score_profile(dev_rows, backend_scores),')),
        ("mixed-contract template gated on a shared rank",
         'mixed_family = len(ranks) > 1 and bool(dispersion_rows)' in source,
         line_of("mixed_family = len(ranks) > 1 and bool(dispersion_rows)")),
        ("the scene report has no contract-half section",
         "CPA by contract half" not in report and "Contract decomposition" not in report,
         0),
        ("the scene report has no borrowed tie prose",
         not any(token in report for token in ("relocated", "must tie", "widening dev margin")),
         0),
    ]
    for label, ok, line in checks:
        location = f"run_dimension.py:{line}" if line else "scene.md"
        print(f"    [{'ok  ' if ok else 'FAIL'}] {label}  ({location})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
