"""Adversarial re-verification of the `dynamics_degree` conclusions.

Recomputes, on CPU only, every number the review quotes, from the frozen trees:

  * in-sample level profiles, fitted exponents (dev / test / pooled) and the
    per-base exponent distribution for Official, the archived v1 repair and the
    shipped v2 repair;
  * the split-aware invariance statistics (CV, relative range, per-base
    `fps2/fps8` median / IQR / fraction within +/-20% plus a bootstrap CI of the
    median);
  * the exponent identities that decide whether `v2/fixed1` is a behaviour-
    preserving control and whether the FPS ladder is a pure rescaling;
  * the paired CPA interval with the repository's own `cpa.paired_bootstrap_ci`,
    for both the zero-margin and the tie-aware statistic, to check which one the
    generated report actually prints;
  * the P1.3 holdout (`validation_alpha.json`) re-derived from its per-clip
    scores, including the algebraic identity `ratio(alpha) = ratio(0) * 4**-alpha`
    and a paired comparison of the absolute log deviation from 1;
  * whether the holdout bases and the 40 in-use bases are exchangeable
    (score level, generator mix, frame count, source fps);
  * the P1.1 natural-preference paired delta, by invoking
    `scripts/evaluate_paired_backend_delta.py` on the frozen natural tree.

Nothing is written into the frozen trees; the only output is a JSON report.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.counterfactual.cpa import (  # noqa: E402
    _base_pairs,
    calibrate_margin,
    paired_bootstrap_ci,
)

LEVELS = ("fps8", "fps6", "fps4", "fps2")
DT = {"fps8": 0.125, "fps6": 1.0 / 6.0, "fps4": 0.25, "fps2": 0.5}
BACKENDS = {
    "official": ("scores/dynamics_degree__official.jsonl",),
    "v1_archived": ("scores/archive/dynamics_degree__repair_v1_archived.jsonl",),
    "v2_shipped": ("scores/dynamics_degree__repair.jsonl",),
    "v2_fixed1": ("v2/fixed1__shard0.jsonl", "v2/fixed1__shard1.jsonl", "v2/fixed1__shard2.jsonl",
                  "v2/fixed1__shard3.jsonl", "v2/fixed1__shard4.jsonl"),
    "v2_fixed05": ("v2/fixed05__shard0.jsonl", "v2/fixed05__shard1.jsonl", "v2/fixed05__shard2.jsonl",
                   "v2/fixed05__shard3.jsonl", "v2/fixed05__shard4.jsonl"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_scores(root: Path, patterns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for pattern in patterns:
        for entry in read_jsonl(root / pattern):
            if entry.get("status") not in ("succeeded", None) or entry.get("score") is None:
                continue
            out.setdefault(entry["derived_id"], {})[entry["level"]] = float(entry["score"])
    return out


def complete_bases(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = scores.get(row["derived_id"], {}).get(row["level"])
        if value is not None:
            grouped[row["base_id"]][row["level"]] = value
    return {base: levels for base, levels in grouped.items() if all(level in levels for level in LEVELS)}


def level_means(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, float]:
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = scores.get(row["derived_id"], {}).get(row["level"])
        if value is not None:
            by_level[row["level"]].append(value)
    return {level: float(np.mean(values)) for level, values in by_level.items() if values}


def fitted_exponent(means: dict[str, float]) -> float | None:
    levels = [level for level in LEVELS if level in means and means[level] > 0]
    if len(levels) < 2:
        return None
    x = np.log([DT[level] for level in levels])
    y = np.log([means[level] for level in levels])
    return float(np.polyfit(x, y, 1)[0])


def per_base_exponents(bases: dict[str, dict[str, float]]) -> list[float]:
    out = []
    for levels in bases.values():
        value = fitted_exponent(levels)
        if value is not None:
            out.append(value)
    return out


def dispersion(bases: dict[str, dict[str, float]]) -> dict[str, float]:
    cvs, ranges = [], []
    for levels in bases.values():
        values = np.array([levels[level] for level in LEVELS], dtype=float)
        scale = float(np.abs(values).mean())
        if scale <= 0:
            continue
        cvs.append(float(values.std() / scale))
        ranges.append(float((values.max() - values.min()) / scale))
    return {"bases": len(cvs), "mean_cv": float(np.mean(cvs)) if cvs else None,
            "mean_relative_range": float(np.mean(ranges)) if ranges else None}


def ratio_stats(bases: dict[str, dict[str, float]], iterations: int = 2000, seed: int = 2026) -> dict[str, Any]:
    ratios = np.array([levels["fps2"] / levels["fps8"] for levels in bases.values()
                       if levels["fps8"] > 0], dtype=float)
    if ratios.size == 0:
        return {"n_bases": 0}
    rng = np.random.default_rng(seed)
    medians = [float(np.median(rng.choice(ratios, size=ratios.size, replace=True))) for _ in range(iterations)]
    return {
        "n_bases": int(ratios.size),
        "median": float(np.median(ratios)),
        "median_ci95": [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))],
        "q25": float(np.quantile(ratios, 0.25)),
        "q75": float(np.quantile(ratios, 0.75)),
        "relative_iqr_q75_over_q25": float(np.quantile(ratios, 0.75) / np.quantile(ratios, 0.25)),
        "within_pm20pct": float(np.mean(np.abs(ratios - 1.0) <= 0.20)),
        "fraction_below_1": float(np.mean(ratios < 1.0)),
    }


# --------------------------------------------------------------------------
# Independent re-implementation of the CPA statistics.
#
# The review must not let a bug in `cpa.py` reproduce itself, so the margin, the
# tie-aware CPA and the paired cluster bootstrap are re-derived here from their
# definitions.  Only the definitions are shared with the repository; the code
# path is not.  `cross_check` below compares the two on the same inputs.
# --------------------------------------------------------------------------
def independent_pairs(rows: list[dict[str, Any]], scores: dict[str, float | None]) -> dict[str, list[float]]:
    """Per-base list of deltas; every level pair expects a tie in this family."""
    by_base: dict[str, dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        by_base[row["base_id"]][row["level"]] = scores.get(row["derived_id"])
    out: dict[str, list[float]] = {}
    for base_id, levels in by_base.items():
        present = [levels[level] for level in LEVELS if levels.get(level) is not None]
        if len(present) < 2:
            continue
        out[base_id] = [present[i] - present[j] for i in range(len(present)) for j in range(i + 1, len(present))]
    return out


def independent_cpa(deltas: list[float], margin: float) -> float:
    if not deltas:
        return float("nan")
    return float(np.mean([abs(delta) <= margin for delta in deltas]))


def independent_margin(dev_pairs: dict[str, list[float]]) -> float:
    deltas = np.array([abs(delta) for group in dev_pairs.values() for delta in group if delta != 0.0])
    if deltas.size == 0:
        return 0.0
    candidates = [0.0] + list(np.quantile(deltas, np.linspace(0.1, 0.9, 9)))
    best, chosen = -1.0, 0.0
    flat = [delta for group in dev_pairs.values() for delta in group]
    for margin in candidates:
        value = independent_cpa(flat, float(margin))
        if value > best:
            best, chosen = value, float(margin)
    return chosen


def independent_paired_ci(official: dict[str, list[float]], repair: dict[str, list[float]],
                          official_margin: float, repair_margin: float,
                          iterations: int, seed: int) -> dict[str, Any]:
    keys = sorted(set(official) & set(repair))
    flat_official = [delta for key in keys for delta in official[key]]
    flat_repair = [delta for key in keys for delta in repair[key]]
    point = independent_cpa(flat_repair, repair_margin) - independent_cpa(flat_official, official_margin)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sample = [keys[index] for index in rng.integers(0, len(keys), size=len(keys))]
        official_sample = [delta for key in sample for delta in official[key]]
        repair_sample = [delta for key in sample for delta in repair[key]]
        estimates.append(independent_cpa(repair_sample, repair_margin)
                         - independent_cpa(official_sample, official_margin))
    array = np.array(estimates)
    return {"delta": point, "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)), "n_bases": len(keys)}


def cross_check_cpa(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float | None],
                    iterations: int, seed: int) -> dict[str, Any]:
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    result: dict[str, Any] = {}
    independent: dict[str, dict[str, list[float]]] = {}
    for backend in ("official", "repair"):
        independent[backend] = independent_pairs(test, {row["derived_id"]: scores.get((row["derived_id"], backend))
                                                        for row in test})
        dev_pairs = independent_pairs(dev, {row["derived_id"]: scores.get((row["derived_id"], backend))
                                            for row in dev})
        result[f"{backend}_margin"] = independent_margin(dev_pairs)
    for label, margin_key in (("zero_margin", None), ("tie_aware", "margin")):
        margins = {backend: (0.0 if margin_key is None else result[f"{backend}_margin"])
                   for backend in ("official", "repair")}
        result[label] = independent_paired_ci(independent["official"], independent["repair"],
                                              margins["official"], margins["repair"], iterations, seed)
        result[f"{label}_marginal"] = {
            backend: independent_cpa([delta for group in independent[backend].values() for delta in group],
                                     margins[backend])
            for backend in ("official", "repair")}
    result["independent_seed"] = seed
    return result


def bootstrap_exponent(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]],
                       split: str, iterations: int, seed: int) -> dict[str, Any]:
    """Cluster bootstrap over `base_id` of the fitted level-profile exponent."""
    subset = rows if split == "all" else [row for row in rows if row["split"] == split]
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_base[row["base_id"]].append(row)
    bases = sorted(by_base)
    point = fitted_exponent(level_means(subset, scores))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        picked = rng.choice(bases, size=len(bases), replace=True)
        sample = [row for base in picked for row in by_base[base]]
        value = fitted_exponent(level_means(sample, scores))
        if value is not None:
            estimates.append(value)
    array = np.array(estimates)
    return {"point": point, "mean": float(array.mean()),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "fraction_negative": float(np.mean(array < 0)), "iterations": len(estimates)}


def paired_deviation_test(ratios_a: np.ndarray, ratios_b: np.ndarray, iterations: int = 20000,
                          seed: int = 11) -> dict[str, Any]:
    """Sign test + paired bootstrap on |log ratio| (invariance target is 1.0)."""
    dev_a = np.abs(np.log(ratios_a))
    dev_b = np.abs(np.log(ratios_b))
    improved = int(np.sum(dev_b < dev_a))
    n = int(dev_a.size)
    # exact two-sided sign test under p = 0.5
    from math import comb
    tail = sum(comb(n, k) for k in range(improved, n + 1)) / 2**n
    sign_p = float(min(1.0, 2 * tail))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        index = rng.integers(0, n, size=n)
        differences.append(float(dev_b[index].mean() - dev_a[index].mean()))
    differences = np.array(differences)
    return {
        "n": n, "improved_bases": improved, "sign_test_p_two_sided": sign_p,
        "mean_abs_log_deviation_a": float(dev_a.mean()), "mean_abs_log_deviation_b": float(dev_b.mean()),
        "paired_mean_difference": float(dev_b.mean() - dev_a.mean()),
        "paired_mean_difference_ci95": [float(np.quantile(differences, 0.025)),
                                        float(np.quantile(differences, 0.975))],
    }


def permutation_p(values_a: np.ndarray, values_b: np.ndarray, iterations: int = 20000, seed: int = 7) -> float:
    """Two-sided permutation test on the difference of medians."""
    observed = abs(float(np.median(values_a)) - float(np.median(values_b)))
    pooled = np.concatenate([values_a, values_b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iterations):
        rng.shuffle(pooled)
        if abs(float(np.median(pooled[: values_a.size])) - float(np.median(pooled[values_a.size:])) ) >= observed:
            count += 1
    return (count + 1) / (iterations + 1)


def paired_ci(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float | None],
              iterations: int, seed: int) -> dict[str, Any]:
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    margins, test_pairs = {}, {}
    for backend in ("official", "repair"):
        backend_scores = {row["derived_id"]: scores.get((row["derived_id"], backend)) for row in dev}
        dev_pairs = [pair for group in _base_pairs(dev, backend_scores).values() for pair in group]
        margins[backend] = calibrate_margin(dev_pairs)
        test_backend = {row["derived_id"]: scores.get((row["derived_id"], backend)) for row in test}
        test_pairs[backend] = _base_pairs(test, test_backend)
    return {
        "dev_margins": margins,
        "zero_margin": paired_bootstrap_ci(test_pairs["official"], test_pairs["repair"], 0.0, 0.0, iterations, seed),
        "tie_aware": paired_bootstrap_ci(test_pairs["official"], test_pairs["repair"],
                                         margins["official"], margins["repair"], iterations, seed),
    }


def holdout_section(validation: Path, iterations: int, seed: int) -> dict[str, Any]:
    payload = json.loads(validation.read_text(encoding="utf-8"))
    rows = read_jsonl(validation.parent / "manifest.jsonl")
    methods = ("official", "alpha0", "alpha0.5", "alpha1")
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        entry = payload["scores"].get(row["derived_id"], {})
        for method in methods:
            if entry.get(method) is not None:
                grouped[method][row["base_id"]][row["level"]] = float(entry[method])

    out: dict[str, Any] = {"n_clips": len(rows), "n_bases": len(grouped["official"])}
    # algebraic identity: score(alpha) = displacement / dt**alpha, so the per-base
    # ratio must be ratio(0) * 4**-alpha exactly.
    identity_error = 0.0
    for base_id, levels in grouped["alpha0"].items():
        if not all(level in levels for level in ("fps8", "fps2")):
            continue
        base_ratio = levels["fps2"] / levels["fps8"]
        for alpha, key in ((0.5, "alpha0.5"), (1.0, "alpha1")):
            other = grouped[key].get(base_id, {})
            if not all(level in other for level in ("fps8", "fps2")):
                continue
            identity_error = max(identity_error, abs(other["fps2"] / other["fps8"] - base_ratio * 4.0**-alpha))
    out["ratio_identity_max_abs_error"] = identity_error

    for method in methods:
        close = {base: levels for base, levels in grouped[method].items()
                 if all(level in levels for level in ("fps8", "fps2"))}
        stats = ratio_stats(close, iterations, seed)
        out.setdefault("ratio", {})[method] = stats

    # paired per-base comparison of |log ratio| against the target 1.0
    paired: dict[str, Any] = {}
    base_ratios = {}
    for method in methods:
        base_ratios[method] = {base: levels["fps2"] / levels["fps8"]
                               for base, levels in grouped[method].items()
                               if all(level in levels for level in ("fps8", "fps2"))}
    common = set.intersection(*[set(values) for values in base_ratios.values()])
    ordered = sorted(common)
    for method in methods:
        deviations = np.array([abs(math.log(base_ratios[method][base])) for base in ordered])
        paired[method] = {"mean_abs_log_deviation": float(deviations.mean()),
                          "median_abs_log_deviation": float(np.median(deviations))}
    ratio_matrix = {method: np.array([base_ratios[method][base] for base in ordered]) for method in methods}
    out["paired_deviation_alpha0_to_alpha0.5"] = paired_deviation_test(
        ratio_matrix["alpha0"], ratio_matrix["alpha0.5"])
    out["paired_deviation_alpha1_to_alpha0.5"] = paired_deviation_test(
        ratio_matrix["alpha1"], ratio_matrix["alpha0.5"])
    out["paired_deviation_official_to_alpha0.5"] = paired_deviation_test(
        ratio_matrix["official"], ratio_matrix["alpha0.5"])
    out["paired_abs_log_deviation"] = paired
    return out


def exchangeability_section(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]],
                            validation: Path) -> dict[str, Any]:
    """Are the holdout bases exchangeable with the 40 in-use bases?

    The holdout's own official scores live in `validation_alpha.json` (a separate
    scoring run), so they are read from there rather than from the counterfactual
    score files, which are keyed by different `video_uid`s.
    """
    payload = json.loads(validation.read_text(encoding="utf-8"))
    holdout_rows = read_jsonl(validation.parent / "manifest.jsonl")
    base_rows = {row["base_id"]: row for row in rows}
    holdout_bases = {row["base_id"]: row for row in holdout_rows}

    def fps8(row: dict[str, Any]) -> float | None:
        return scores.get(row["derived_id"], {}).get("fps8")

    def holdout_fps8(row: dict[str, Any]) -> float | None:
        return payload["scores"].get(row["derived_id"], {}).get("official")

    in_use = np.array([fps8(row) for row in base_rows.values() if fps8(row) is not None], dtype=float)
    hold_score = np.array([holdout_fps8(row) for row in holdout_bases.values()
                           if holdout_fps8(row) is not None], dtype=float)
    out = {
        "in_use_bases": len(base_rows), "holdout_bases": len(holdout_bases),
        "in_use_official_fps8_median": float(np.median(in_use)),
        "holdout_official_fps8_median": float(np.median(hold_score)),
        "official_fps8_median_permutation_p": permutation_p(hold_score, in_use),
        "in_use_generators": dict(Counter(row["generator"] for row in base_rows.values())),
        "holdout_generators": dict(Counter(row["generator"] for row in holdout_bases.values())),
        "in_use_source_fps": dict(Counter(row.get("source_fps") for row in base_rows.values())),
        "holdout_source_fps": dict(Counter(row.get("source_fps") for row in holdout_bases.values())),
        "in_use_frame_counts": dict(Counter(row.get("source_frame_count") for row in base_rows.values())),
        "holdout_frame_counts": dict(Counter(row.get("source_frame_count") for row in holdout_bases.values())),
        "in_use_splits": dict(Counter(row.get("split") for row in base_rows.values())),
        "holdout_splits": dict(Counter(row.get("split") for row in holdout_bases.values())),
    }
    return out


def pool_section(bases_path: Path) -> dict[str, Any]:
    """Reproduce the holdout's own selection to check the 'largest honest' claim."""
    from scripts.counterfactual.select_bases import SEED, load_manifest, load_prompts, ranked_pool  # noqa: PLC0415
    used = {row["video_uid"] for row in read_jsonl(bases_path) if row["dimension"] == "dynamics_degree"}
    pool = ranked_pool("dynamics_degree", load_prompts(), load_manifest(), SEED)
    holdout = [base for base in pool if base["video_uid"] not in used]
    return {
        "pool_size": len(pool),
        "in_use": len(used),
        "unused_candidates": len(holdout),
        "used_prompts": len({row["prompt_id"] for row in read_jsonl(bases_path)
                             if row["dimension"] == "dynamics_degree"}),
        "holdout_generator_mix": dict(Counter(base["generator"] for base in holdout)),
        "holdout_taken": 30,
    }


def natural_section(natural_root: Path, repo: Path, official_scores_root: Path) -> dict[str, Any]:
    command = [
        sys.executable, str(repo / "scripts/evaluate_paired_backend_delta.py"),
        "--dimension", "dynamic_degree",
        "--predictions-root", str(natural_root),
        "--official-scores-root", str(official_scores_root),
        "--bootstrap-iterations", "2000", "--seed", "2026",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, cwd=repo)
    except OSError as error:
        return {"error": f"{type(error).__name__}: {error}"}
    payload = None
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout.split("\n\n")[0])
        except json.JSONDecodeError:
            payload = None
    return {"returncode": completed.returncode, "record": payload,
            "stderr_tail": completed.stderr.strip().splitlines()[-3:] if completed.stderr else []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench"))
    parser.add_argument("--natural-root", type=Path,
                        default=Path("/root/wenbiao_zhao/datasets/natural-preference-runs"))
    parser.add_argument("--bases", type=Path,
                        default=Path("/root/wenbiao_zhao/vbench-audit/configs/counterfactual/bases_published.jsonl"))
    parser.add_argument("--official-scores-root", type=Path, default=None,
                        help="frozen E0 official scores root (default: <repo>/results/e0/raw_official_scores)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = args.dataset_root
    rows = [row for row in read_jsonl(root / "manifest.jsonl") if row["dimension"] == "dynamics_degree"]
    scores = {name: load_scores(root, patterns) for name, patterns in BACKENDS.items()}

    report: dict[str, Any] = {"dataset_root": str(root), "n_clips": len(rows),
                              "n_bases": len({row["base_id"] for row in rows}), "backends": {}}
    for name, backend_scores in scores.items():
        entry: dict[str, Any] = {"scored_clips": len(backend_scores)}
        for split in ("dev", "test", "all"):
            subset = rows if split == "all" else [row for row in rows if row["split"] == split]
            means = level_means(subset, backend_scores)
            bases = complete_bases(subset, backend_scores)
            entry[split] = {
                "level_means": means,
                "exponent_bootstrap": bootstrap_exponent(rows, backend_scores, split, 1000, args.seed),
                "fitted_exponent": fitted_exponent(means),
                "profile_vs_fps8": {level: means[level] / means["fps8"] for level in LEVELS if level in means},
                "per_base_exponent_mean": float(np.mean(per_base_exponents(bases))) if bases else None,
                "per_base_exponent_sd": float(np.std(per_base_exponents(bases))) if bases else None,
                "dispersion": dispersion(bases),
                "fps2_over_fps8": ratio_stats(bases, args.iterations, args.seed),
            }
        report["backends"][name] = entry

    # exponent identities
    identical = 0
    max_relative_difference = 0.0
    for derived_id, levels in scores["v1_archived"].items():
        other = scores["v2_fixed1"].get(derived_id, {})
        for level, value in levels.items():
            if level in other and value:
                max_relative_difference = max(max_relative_difference, abs(other[level] - value) / abs(value))
                identical += 1
    report["identity_checks"] = {
        "v1_vs_v2_fixed1_compared_values": identical,
        "v1_vs_v2_fixed1_max_relative_difference": max_relative_difference,
        "fixed05_over_fixed1_ratio_median": float(np.median([
            scores["v2_fixed05"][d]["fps8"] / scores["v2_fixed1"][d]["fps8"]
            for d in scores["v2_fixed1"] if d in scores["v2_fixed05"] and scores["v2_fixed1"][d].get("fps8")
        ])),
    }

    combined = {(row["derived_id"], "official"): scores["official"].get(row["derived_id"], {}).get(row["level"])
                for row in rows}
    combined.update({(row["derived_id"], "repair"): scores["v2_shipped"].get(row["derived_id"], {}).get(row["level"])
                     for row in rows})
    report["paired_ci"] = paired_ci(rows, combined, args.iterations, args.seed)
    report["paired_ci_independent"] = cross_check_cpa(rows, combined, args.iterations, args.seed)
    report["paired_ci_independent_seed_alt"] = cross_check_cpa(rows, combined, args.iterations, 20260915)

    report["holdout_p1_3"] = holdout_section(root / "validation_dynamic/validation_alpha.json",
                                             args.iterations, args.seed)
    report["holdout_exchangeability"] = exchangeability_section(rows, scores["official"],
                                                                root / "validation_dynamic/validation_alpha.json")
    if args.bases.is_file():
        report["holdout_pool"] = pool_section(args.bases)
    official_root = args.official_scores_root or (ROOT / "results/e0/raw_official_scores")
    report["natural_p1_1"] = natural_section(args.natural_root, ROOT, official_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
