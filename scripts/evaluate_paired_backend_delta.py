#!/usr/bin/env python3
"""Paired bootstrap CI for the Official-vs-Repair natural-set accuracy gap.

`evaluate_pairwise_statistics.py` reports one marginal interval per backend,
which answers "how precise is this backend's accuracy".  The decision the audit
actually has to make -- is Repair different from Official on the same human
labels -- needs the *paired* difference instead, and a paired interval is
tighter because both backends score the same pairs.

Both backends keep their own dev-calibrated tie margin, exactly as in the
marginal report, so the comparison is between the two shipped decision rules
rather than between two arbitrary threshold choices.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results/e0/raw_official_scores"


def _evaluator():
    """Load `evaluate_pairwise_statistics` by path (scripts/ is not a package)."""
    path = ROOT / "scripts" / "evaluate_pairwise_statistics.py"
    spec = importlib.util.spec_from_file_location("evaluate_pairwise_statistics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _index(valid):
    """Key scored pairs by their video pair so the two backends can be joined."""
    return {(pair["video_a_uid"], pair["video_b_uid"]): (pair, difference)
            for pair, difference in valid}


def _percentile(values, fraction):
    return values[int(fraction * (len(values) - 1))]


def compare(alias, predictions_root, scores_root, iterations, seed):
    evaluator = _evaluator()
    _, official = evaluator.build(alias, predictions_root=None, scores_root=scores_root)
    _, repair = evaluator.build(alias, predictions_root=predictions_root, scores_root=scores_root)
    officials, repairs = _index(official), _index(repair)

    common = set(officials) & set(repairs)
    if not common:
        raise SystemExit(f"{alias}: the two backends share no scored pair")
    dev_keys = sorted(key for key in common if officials[key][0]["split"] == "dev")
    test_keys = sorted(key for key in common if officials[key][0]["split"] == "test")
    if not dev_keys or not test_keys:
        raise SystemExit(f"{alias}: need both dev and test pairs, got {len(dev_keys)}/{len(test_keys)}")

    deltas = {}
    for name, table in (("official", officials), ("repair", repairs)):
        deltas[name] = evaluator.calibrate([table[key] for key in dev_keys])

    accuracies = {
        name: evaluator.accuracy([table[key] for key in test_keys], deltas[name])
        for name, table in (("official", officials), ("repair", repairs))
    }

    rng, count = random.Random(seed), len(test_keys)
    gaps = []
    for _ in range(iterations):
        sample = [test_keys[rng.randrange(count)] for _ in range(count)]
        repair_accuracy = evaluator.accuracy([repairs[key] for key in sample], deltas["repair"])
        official_accuracy = evaluator.accuracy([officials[key] for key in sample], deltas["official"])
        gaps.append(repair_accuracy - official_accuracy)
    gaps.sort()

    return {
        "dimension": alias,
        "test_pairs": count,
        "official_tie_margin": deltas["official"],
        "repair_tie_margin": deltas["repair"],
        "official_accuracy": accuracies["official"],
        "repair_accuracy": accuracies["repair"],
        "delta": accuracies["repair"] - accuracies["official"],
        "delta_ci95": [_percentile(gaps, 0.025), _percentile(gaps, 0.975)],
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True,
                        choices=("dynamic_degree", "subject_consistency", "human_action",
                                 "spatial_relationship", "motion_smoothness"))
    parser.add_argument("--predictions-root", required=True, type=Path,
                        help="Root containing <dimension>/predictions.csv (the repair scores).")
    parser.add_argument("--official-scores-root", required=True, type=Path,
                        help="Root containing <official_dimension>/results.csv (the official scores).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    record = compare(args.dimension, args.predictions_root, args.official_scores_root,
                     args.bootstrap_iterations, args.seed)
    low, high = record["delta_ci95"]
    record["significant"] = low > 0.0 or high < 0.0
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("\n| dimension | Official | Repair | Delta (paired) | 95% CI | n test |")
    print("|---|---:|---:|---:|---|---:|")
    print("| {} | {:.4f} | {:.4f} | {:+.4f} | [{:+.4f}, {:+.4f}] | {} |".format(
        record["dimension"], record["official_accuracy"], record["repair_accuracy"],
        record["delta"], low, high, record["test_pairs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
