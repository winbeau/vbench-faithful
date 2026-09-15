#!/usr/bin/env python3
"""Position profile for the Subject Consistency `temporal_relocation` family.

The family corrupts one fixed region at the start, middle and end of a clip, so
the three corrupted variants are declared same-rank and only *when* the
corruption applies may differ.  The original construction tracked the subject
per frame, which made the three placements differ in area and content as well as
in position, so a position gap was not interpretable.  The confirmatory
construction broadcasts one median box to every frame instead.

This script answers the two questions that comparison raises, separately, as the
review requires:

* **sensitivity** -- does `clean` outrank each corrupted position, per base?
* **position profile** -- how far apart are the three corrupted scores, measured
  per base as `max - min`, the coefficient of variation, and the relative range
  `(max - min) / mean`, summarised by median and IQR, plus the share of bases
  inside a relative tolerance and a paired bootstrap interval for the change in
  median relative range between two constructions.

Arms are passed as `name=directory`, where each directory holds
`subject_consistency__<backend>.jsonl` score files.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

POSITIONS = ("corrupt_start", "corrupt_middle", "corrupt_end")
BACKENDS = ("official", "repair")
TOLERANCES = (0.05, 0.10, 0.20)


def load_arm(directory: Path, backend: str) -> dict[str, dict[str, float]]:
    """Per-base level scores for one backend, keeping fully scored bases only."""
    path = directory / f"subject_consistency__{backend}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing score file: {path}")
    scores: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores.setdefault(row["base_id"], {})[row["level"]] = row["score"]
    required = ("clean",) + POSITIONS
    return {base: levels for base, levels in scores.items()
            if all(levels.get(level) is not None for level in required)}


def quantile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def relative_ranges(table: dict[str, dict[str, float]], bases: list[str]) -> list[float]:
    spreads = []
    for base in bases:
        values = [table[base][position] for position in POSITIONS]
        spreads.append((max(values) - min(values)) / statistics.mean(values))
    return spreads


def profile(table: dict[str, dict[str, float]], bases: list[str]) -> dict[str, object]:
    spreads, coefficients = [], []
    for base in bases:
        values = [table[base][position] for position in POSITIONS]
        mean = statistics.mean(values)
        spreads.append(max(values) - min(values))
        coefficients.append(statistics.pstdev(values) / mean)
    relative = relative_ranges(table, bases)
    record = {
        "bases": len(bases),
        "sensitivity": {position: sum(1 for base in bases
                                      if table[base]["clean"] > table[base][position]) / len(bases)
                        for position in POSITIONS},
        "max_minus_min": {"median": statistics.median(spreads),
                          "iqr": [quantile(spreads, 0.25), quantile(spreads, 0.75)]},
        "coefficient_of_variation": {"median": statistics.median(coefficients)},
        "relative_range": {"median": statistics.median(relative),
                           "iqr": [quantile(relative, 0.25), quantile(relative, 0.75)]},
        "within_tolerance": {f"{tolerance:.0%}": sum(1 for value in relative if value <= tolerance) / len(relative)
                             for tolerance in TOLERANCES},
    }
    return record


def paired_delta(reference: dict[str, dict[str, float]], other: dict[str, dict[str, float]],
                 bases: list[str], iterations: int, seed: int) -> dict[str, object]:
    """Bootstrap the change in median relative range between two constructions."""
    reference_values = relative_ranges(reference, bases)
    other_values = relative_ranges(other, bases)
    observed = statistics.median(reference_values) - statistics.median(other_values)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sample = [bases[rng.randrange(len(bases))] for _ in bases]
        deltas.append(statistics.median(relative_ranges(reference, sample))
                      - statistics.median(relative_ranges(other, sample)))
    deltas.sort()
    return {
        "reference_median": statistics.median(reference_values),
        "other_median": statistics.median(other_values),
        "delta": observed,
        "delta_ci95": [quantile(deltas, 0.025), quantile(deltas, 0.975)],
        "significant": quantile(deltas, 0.025) > 0.0 or quantile(deltas, 0.975) < 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="Score directory for one construction; repeatable. "
                             "The first arm is the reference for the paired delta.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    arms = {}
    for specification in args.arm:
        name, _, directory = specification.partition("=")
        if not directory:
            parser.error(f"--arm needs NAME=DIR, got {specification!r}")
        arms[name] = {backend: load_arm(Path(directory), backend) for backend in BACKENDS}

    common = set.intersection(*[set(table) for arm in arms.values() for table in arm.values()])
    bases = sorted(common)
    if not bases:
        raise SystemExit("the arms share no fully scored base")

    reference_name = next(iter(arms))
    report = {"bases": bases, "reference_arm": reference_name, "arms": {}, "paired": {}}
    for name, arm in arms.items():
        report["arms"][name] = {backend: profile(arm[backend], bases) for backend in BACKENDS}
    for name, arm in arms.items():
        if name == reference_name:
            continue
        # `delta` is reference - other, so a negative value means the reference
        # construction has the tighter (better) position profile.
        report["paired"][name] = {
            backend: paired_delta(arms[reference_name][backend], arm[backend], bases,
                                  args.bootstrap_iterations, args.seed)
            for backend in BACKENDS
        }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
