"""Compute Counterfactual Pair Accuracy (CPA) from scored counterfactual clips.

Plan section 5.2: for a pair (base, counterfactual),
`delta = score_counterfactual - score_base`, and the prediction is the sign of
the delta (zero-margin) or sign against a development-calibrated tie margin.
`expected_relation` comes from the manifest's `expected_rank` ordering: a higher
rank should score higher, equal ranks should tie, and every family is expanded
into all ordered pairs (section 5.2).  Uncertainty is a cluster bootstrap over
`base_id` (section 5.4).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import read_jsonl, write_json

BUDGET = {
    "dynamics_degree": (10, 30),
    "subject_consistency": (5, 20),
    "human_action": (5, 20),
    "spatial_relationship": (10, 30),
    "scene": (5, 20),
    "multiplt_object": (5, 20),
    "motion_smoothness": (5, 20),
}


def expected_relation(rank_a: int, rank_b: int) -> int:
    if rank_a > rank_b:
        return 1
    if rank_a < rank_b:
        return -1
    return 0


def pair_delta(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """score(a) - score(b); callers pass (high, low) so a positive delta is correct."""
    if a["score"] is None or b["score"] is None:
        return None
    return float(a["score"]) - float(b["score"])


def prediction(delta: float, margin: float) -> int:
    if delta > margin:
        return 1
    if delta < -margin:
        return -1
    return 0


def family_pairs(base_levels: dict[str, dict[str, Any]]) -> list[tuple[int, int, float]]:
    """All ordered (rank_a, rank_b) level pairs for one base, with expected sign."""
    levels = sorted(base_levels.values(), key=lambda row: row["expected_rank"])
    pairs: list[tuple[int, int, float]] = []
    for a, b in combinations(range(len(levels)), 2):
        if levels[a]["expected_rank"] == levels[b]["expected_rank"]:
            delta = pair_delta(levels[a], levels[b])
            if delta is not None:
                pairs.append((0, 0, delta))
        else:
            high, low = (a, b) if levels[a]["expected_rank"] > levels[b]["expected_rank"] else (b, a)
            delta = pair_delta(levels[high], levels[low])
            if delta is not None:
                pairs.append((1, levels[high]["expected_rank"] - levels[low]["expected_rank"], delta))
    return pairs


def cpa_at_margin(pairs: list[tuple[int, int, float]], margin: float) -> float:
    if not pairs:
        return float("nan")
    correct = sum(
        1 for expected, _rank_gap, delta in pairs if prediction(delta, margin) == expected
    )
    return correct / len(pairs)


def rank_gap_groups(
    base_pairs: dict[str, list[tuple[int, int, float]]],
) -> dict[int, list[tuple[int, int, float]]]:
    """Split the pair census by expected rank gap.

    A mixed-rank family (`expected_rank` takes more than one value) is a
    conjunction of contracts: rank gap > 0 pairs test sensitivity, while rank gap
    0 pairs test the invariance of the levels that were declared equal.  A
    single composite CPA over both is dominated by whichever contract is easier
    and hides the other, so the report has to show them separately.
    """
    groups: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for pairs in base_pairs.values():
        for pair in pairs:
            groups[pair[1]].append(pair)
    return dict(groups)


def group_stats(
    groups: dict[int, list[tuple[int, int, float]]], margin: float
) -> dict[int, dict[str, Any]]:
    """Per-rank-gap pair count, tie rate and match rate at one margin."""
    stats: dict[int, dict[str, Any]] = {}
    for rank_gap, pairs in sorted(groups.items()):
        matched = sum(
            1 for expected, _gap, delta in pairs if prediction(delta, margin) == expected
        )
        tied = sum(1 for _e, _gap, delta in pairs if prediction(delta, margin) == 0)
        stats[rank_gap] = {
            "n_pairs": len(pairs),
            "matched": matched,
            "match_rate": matched / len(pairs) if pairs else float("nan"),
            "tie_rate": tied / len(pairs) if pairs else float("nan"),
        }
    return stats


def calibrate_margin(dev_pairs: list[tuple[int, int, float]]) -> float:
    """Pick the zero-margin neighbourhood that maximises dev tie-aware CPA."""
    deltas = np.array([abs(delta) for _, _, delta in dev_pairs if delta != 0])
    if deltas.size == 0:
        return 0.0
    candidates = [0.0] + list(np.quantile(deltas, np.linspace(0.1, 0.9, 9)))
    best, best_margin = -1.0, 0.0
    for margin in candidates:
        value = cpa_at_margin(dev_pairs, margin)
        if value > best:
            best, best_margin = value, margin
    return float(best_margin)


def bootstrap_ci(
    base_pairs: dict[str, list[tuple[int, int, float]]],
    margin: float,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys = list(base_pairs)
    estimates = []
    for _ in range(iterations):
        sample = [base_pairs[key] for key in rng.choice(keys, size=len(keys), replace=True)]
        flattened = [pair for group in sample for pair in group]
        estimates.append(cpa_at_margin(flattened, margin))
    estimates = np.array(estimates)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with average ties, so Spearman stays defined when scores repeat."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def _spearman(expected: np.ndarray, scores: np.ndarray) -> float | None:
    if expected.size < 2 or np.all(expected == expected[0]) or np.all(scores == scores[0]):
        return None
    left = _average_ranks(expected)
    right = _average_ranks(scores)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def order_statistics(
    rows: list[dict[str, Any]], scores: dict[str, float | None]
) -> dict[str, Any]:
    """Sequence-level order statistics for an ordered family (plan 5.2, 13.3).

    CPA counts pairs and therefore weights small rank gaps more heavily than
    large ones.  The per-base rank correlation and the strict-order rate answer
    the same question without that weighting: does one clip's levels come out in
    the declared order at all?  A base whose levels are missing a score is left
    out of both, never counted as a violation.
    """
    by_base: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        value = scores.get(row["derived_id"])
        if value is not None:
            by_base[row["base_id"]].append((int(row["expected_rank"]), float(value)))
    correlations: list[float] = []
    strict = 0
    n_bases = 0
    for entries in by_base.values():
        if len(entries) < 2:
            continue
        n_bases += 1
        expected = np.array([entry[0] for entry in entries], dtype=np.float64)
        values = np.array([entry[1] for entry in entries], dtype=np.float64)
        rho = _spearman(expected, values)
        if rho is not None:
            correlations.append(rho)
        ordered = sorted(entries, key=lambda entry: -entry[0])
        # Only pairs whose declared ranks differ are required to be strictly
        # ordered; levels that the family declares equal must not be.
        if all(
            high[1] > low[1]
            for index, high in enumerate(ordered)
            for low in ordered[index + 1 :]
            if high[0] > low[0]
        ):
            strict += 1
    return {
        "n_bases": n_bases,
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "median_spearman": float(np.median(correlations)) if correlations else None,
        "strict_order_bases": strict,
        "strict_order_rate": (strict / n_bases) if n_bases else None,
    }


def paired_bootstrap_ci(
    official_base_pairs: dict[str, list[tuple[int, int, float]]],
    repair_base_pairs: dict[str, list[tuple[int, int, float]]],
    official_margin: float,
    repair_margin: float,
    iterations: int,
    seed: int,
) -> dict[str, float | None]:
    """Cluster bootstrap over `base_id` of `CPA_repair - CPA_official`.

    The two marginal intervals in the report overlap, so a delta quoted from two
    rounded point estimates carries no uncertainty.  Resampling the *same*
    bases for both backends keeps the pairing (both backends scored the same
    clips) and gives the interval the delta claim needs (plan 5.4).
    """
    keys = sorted(set(official_base_pairs) & set(repair_base_pairs))
    if not keys:
        return {"delta": None, "ci_low": None, "ci_high": None, "n_bases": 0}
    official_flat = [pair for key in keys for pair in official_base_pairs[key]]
    repair_flat = [pair for key in keys for pair in repair_base_pairs[key]]
    point = cpa_at_margin(repair_flat, repair_margin) - cpa_at_margin(official_flat, official_margin)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sample = [keys[index] for index in rng.choice(len(keys), size=len(keys), replace=True)]
        official_sample = [pair for key in sample for pair in official_base_pairs[key]]
        repair_sample = [pair for key in sample for pair in repair_base_pairs[key]]
        estimates.append(
            cpa_at_margin(repair_sample, repair_margin) - cpa_at_margin(official_sample, official_margin)
        )
    estimates = np.array(estimates)
    return {
        "delta": float(point),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "n_bases": len(keys),
    }


def evaluate_method(
    rows: list[dict[str, Any]],
    scores: dict[str, float | None],
    margin: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["base_id"], row["family"])
        score = scores.get(row["derived_id"])
        by_base[key][row["level"]] = {"score": score, "expected_rank": row["expected_rank"]}

    family_pairs_map: dict[str, dict[str, list[tuple[int, int, float]]]] = defaultdict(dict)
    for (base_id, family), levels in by_base.items():
        family_pairs_map[family][base_id] = family_pairs(levels)

    out: dict[str, Any] = {}
    for family, bases in sorted(family_pairs_map.items()):
        all_pairs = [pair for group in bases.values() for pair in group]
        n_bases = len(bases)
        n_pairs = len(all_pairs)
        cpa = cpa_at_margin(all_pairs, margin) if all_pairs else float("nan")
        low, high = bootstrap_ci(bases, margin, iterations, seed) if bases else (float("nan"), float("nan"))
        out[family] = {
            "n_bases": n_bases,
            "n_pairs": n_pairs,
            "cpa": None if np.isnan(cpa) else round(cpa, 4),
            "ci_low": None if np.isnan(low) else round(low, 4),
            "ci_high": None if np.isnan(high) else round(high, 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    scores: dict[tuple[str, str], float | None] = {}
    for score_file in args.scores_root.glob("*.jsonl"):
        for line in score_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["status"] == "succeeded" and entry["score"] is not None:
                scores[(entry["derived_id"], entry["backend"])] = float(entry["score"])

    report: dict[str, Any] = {}
    for backend in ("official", "repair"):
        method: dict[str, Any] = {}
        for dimension in BUDGET:
            dim_rows = [row for row in rows if row["dimension"] == dimension]
            dim_scores = {
                row["derived_id"]: scores.get((row["derived_id"], backend)) for row in dim_rows
            }
            dev_rows = [row for row in dim_rows if row["split"] == "dev"]
            test_rows = [row for row in dim_rows if row["split"] == "test"]
            margin = calibrate_margin(
                [pair for group in _base_pairs(dev_rows, dim_scores).values() for pair in group]
            )
            method[dimension] = {
                "dev_margin": round(margin, 6),
                "dev": evaluate_method(dev_rows, dim_scores, 0.0, args.iterations, args.seed),
                "test_zero_margin": evaluate_method(test_rows, dim_scores, 0.0, args.iterations, args.seed),
                "test_tie_aware": evaluate_method(test_rows, dim_scores, margin, args.iterations, args.seed),
            }
        report[backend] = method

    write_json(args.output, report)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "backend": list(report)}, indent=2))
    return 0


def _base_pairs(rows: list[dict[str, Any]], scores: dict[str, float | None]) -> dict[str, list[tuple[int, int, float]]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["base_id"], row["family"])
        by_base[key][row["level"]] = {"score": scores.get(row["derived_id"]), "expected_rank": row["expected_rank"]}
    return {f"{base_id}::{family}": family_pairs(levels) for (base_id, family), levels in by_base.items()}


if __name__ == "__main__":
    raise SystemExit(main())
