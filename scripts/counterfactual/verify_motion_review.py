"""Adversarial re-verification of the `motion_smoothness` conclusions.

CPU only, on the frozen trees.  Recomputes every number the review quotes:

  * revision provenance: the report's `code SHA` vs the revision the frozen
    scores actually came from, validated by replaying the `feeb770` estimator
    from a cached component pass (must reproduce all 125 scores) and then
    replaying the shipped `4d53fa2` defaults;
  * counterfactual: CPA, dev margin, paired `base_id`-clustered interval,
    per-adjacent-pair counts, strict-order rate and a between/within-base
    variance decomposition;
  * natural preference set: dev-calibrated margins, tie-aware accuracy,
    Kendall tau-b, model-level Pearson, per-video Official-vs-Repair
    correlation and the marginal/paired intervals under BOTH the published
    pair-level unit and the plan 5.4 `prompt_id` cluster unit;
  * the lower-MAD CPU baseline (needs `--compute-mad` or a cache), which is the
    control that makes the natural result interpretable;
  * the normalisation inversion: Spearman between the clip motion scale and the
    discontinuity, from the cached components.

Nothing is written into the frozen trees.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import random
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
    cpa_at_margin,
    paired_bootstrap_ci,
)

LEVELS = ["jerk_0_original", "jerk_1_duplicate", "jerk_2_duplicate_skip",
          "jerk_3_local_reverse", "jerk_4_multiple"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[int(fraction * (len(values) - 1))]


def boot_indices(items, seed, iterations, unit):
    rng = random.Random(seed)
    if unit == "pair":
        count = len(items)
        return [[rng.randrange(count) for _ in range(count)] for _ in range(iterations)]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[item[0]["prompt_id"]].append(index)
    keys = sorted(groups)
    draws = []
    for _ in range(iterations):
        sample: list[int] = []
        for _ in range(len(keys)):
            sample.extend(groups[keys[rng.randrange(len(keys))]])
        draws.append(sample)
    return draws


# ---------------------------------------------------------------------------
# estimator replay from cached components
# ---------------------------------------------------------------------------

def cached_clips(pattern: str) -> dict[str, dict[str, Any]]:
    clips: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(pattern)):
        for row in read_jsonl(Path(path)):
            if row.get("status") == "succeeded":
                clips[row["derived_id"]] = row
    return clips


def aggregate(values, mode: str, top_k: int = 3, tail_quantile: float = 0.9,
              tail_weight: float = 0.25) -> float:
    if not values:
        return 0.0
    data = np.sort(np.asarray(values, dtype=float))
    if mode == "topk":
        return float(data[-min(top_k, data.size):].mean())
    count = max(1, math.ceil((1.0 - tail_quantile) * data.size))
    return float((1.0 - tail_weight) * data.mean() + tail_weight * data[-count:].mean())


def replay_score(clip: dict[str, Any], magnitude_weight: float, direction_weight: float,
                 direction_component: str, mode: str) -> float:
    magnitude = [entry[0] for entry in clip["components"]["relative"]]
    direction = [entry[0] for entry in clip["components"][direction_component]]
    return math.exp(-aggregate([magnitude_weight * m + direction_weight * d
                                for m, d in zip(magnitude, direction)], mode))


def revision_section(args) -> dict[str, Any]:
    result: dict[str, Any] = {}
    manifest = [row for row in read_jsonl(args.dataset_root / "manifest.jsonl")
                if row["dimension"] == "motion_smoothness"]
    frozen = {}
    for backend in ("official", "repair"):
        path = args.dataset_root / "scores" / f"motion_smoothness__{backend}.jsonl"
        entries = read_jsonl(path)
        result[f"{backend}_score_rows"] = len(entries)
        result[f"{backend}_rows_carry_code_sha"] = any("code_sha" in row for row in entries)
        result[f"{backend}_mtime"] = path.stat().st_mtime
        if backend == "repair":
            frozen = {row["derived_id"]: row["score"] for row in entries if row.get("score") is not None}
    result["manifest_clip_count"] = len(manifest)
    report = args.dataset_root / "reports" / "motion_smoothness.md"
    if report.is_file():
        for line in report.read_text().splitlines():
            if line.startswith("- code SHA:"):
                result["report_code_sha"] = line.split("`")[1]
                break
    clips = cached_clips(args.components)
    result["component_cache_clips"] = len(clips)
    if clips and frozen:
        old = {d: replay_score(c, 0.7, 0.3, "direction", "mean_tail") for d, c in clips.items()}
        shared = sorted(set(old) & set(frozen))
        diffs = [abs(old[d] - frozen[d]) for d in shared]
        result["replay_feeb770_vs_frozen"] = {
            "n": len(shared), "max_abs_diff": max(diffs) if diffs else None,
            "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
        }
        result["cache_reproduces_frozen"] = bool(diffs) and max(diffs) < 1e-12
        new = {d: replay_score(c, 0.5, 0.5, "raw_direction", "topk") for d, c in clips.items()}
        official = {}
        for row in read_jsonl(args.dataset_root / "scores" / "motion_smoothness__official.jsonl"):
            if row.get("score") is not None:
                official[row["derived_id"]] = float(row["score"])
        for label, table in (("feeb770", old), ("4d53fa2_replay", new)):
            stats = evaluate_counterfactual(manifest, table, official)
            result[f"replay_{label}"] = stats
    return result


# ---------------------------------------------------------------------------
# counterfactual
# ---------------------------------------------------------------------------

def evaluate_counterfactual(manifest, scores, official) -> dict[str, Any]:
    test_rows = [row for row in manifest if row["split"] == "test"]
    dev_rows = [row for row in manifest if row["split"] == "dev"]
    margin = calibrate_margin([p for g in _base_pairs(dev_rows, scores).values() for p in g])
    test_pairs = _base_pairs(test_rows, scores)
    flat = [p for g in test_pairs.values() for p in g]
    by_base: dict[str, dict[str, float]] = defaultdict(dict)
    for row in test_rows:
        if scores.get(row["derived_id"]) is not None:
            by_base[row["base_id"]][row["level"]] = float(scores[row["derived_id"]])
    adjacent = {
        f"{LEVELS[i]}>{LEVELS[i+1]}": sum(1 for v in by_base.values() if v[LEVELS[i]] > v[LEVELS[i+1]])
        for i in range(4)
    }
    adjacent["L2>L3"] = sum(1 for v in by_base.values() if v[LEVELS[2]] > v[LEVELS[3]])
    strict = sum(1 for v in by_base.values()
                 if all(v[LEVELS[i]] > v[LEVELS[i + 1]] for i in range(4)))
    patterns = Counter(
        "".join("1" if v[LEVELS[i]] > v[LEVELS[i + 1]] else "0" for i in range(4))
        for v in by_base.values()
    )
    return {
        "dev_margin": margin,
        "pairs": len(flat),
        "cpa": cpa_at_margin(flat, 0.0),
        "cpa_at_dev_margin": cpa_at_margin(flat, margin),
        "adjacent_correct": adjacent,
        "strict_order": strict,
        "bases": len(by_base),
        "pattern_counts": dict(patterns),
        **({
            "paired_vs_official": paired_bootstrap_ci(
                _base_pairs(test_rows, official), test_pairs, 0.0, margin, 2000, 2026
            )
        } if official is not None else {}),
    }


def variance_section(manifest, scores) -> dict[str, Any]:
    values, bases, levels = [], [], []
    for row in manifest:
        if row["split"] != "test" or scores.get(row["derived_id"]) is None:
            continue
        values.append(float(scores[row["derived_id"]]))
        bases.append(row["base_id"])
        levels.append(row["level"])
    values = np.asarray(values)
    base_means = {b: values[[i for i, x in enumerate(bases) if x == b]].mean() for b in set(bases)}
    total = float(((values - values.mean()) ** 2).sum())
    between = float(sum(len([1 for x in bases if x == b]) * (m - values.mean()) ** 2
                         for b, m in base_means.items()))
    deviation = defaultdict(list)
    for i, value in enumerate(values):
        deviation[levels[i]].append(value - base_means[bases[i]])
    return {
        "between_share": between / total,
        "within_share": 1.0 - between / total,
        "level_mean_deviation": {level: float(np.mean(deviation[level])) for level in LEVELS},
    }


# ---------------------------------------------------------------------------
# natural preference set
# ---------------------------------------------------------------------------

def natural_section(args, evaluator, delta_module) -> dict[str, Any]:
    nat = args.natural_root
    official_root = nat / "official_scores"
    _, official = evaluator.build("motion_smoothness", predictions_root=None, scores_root=official_root)
    _, repair = evaluator.build("motion_smoothness", predictions_root=nat, scores_root=official_root)
    result: dict[str, Any] = {
        "official_pairs": len(official), "repair_pairs": len(repair),
        "test_prompts": len({p["prompt_id"] for p, _ in repair if p["split"] == "test"}),
        "pairs_per_prompt": dict(Counter(Counter(p["prompt_id"] for p, _ in repair).values())),
    }
    detailed = {}
    for name, valid in (("official", official), ("repair", repair)):
        dev = [i for i in valid if i[0]["split"] == "dev"]
        test = [i for i in valid if i[0]["split"] == "test"]
        margin = evaluator.calibrate(dev)
        detailed[name] = {
            "dev_acc0": evaluator.accuracy(dev, 0),
            "test_acc0": evaluator.accuracy(test, 0),
            "tie_margin": margin,
            "tie_aware": evaluator.accuracy(test, margin),
            "kendall_tau_b": float(evaluator.kendalltau(
                [float(p["human_label"]) for p, _ in test], [d for _, d in test]).statistic),
            "tie_aware_ci_pair": [percentile(boot_accuracy(test, margin, 2027, args.iterations, "pair"), f)
                                  for f in (0.025, 0.975)],
            "tie_aware_ci_prompt": [percentile(boot_accuracy(test, margin, 2027, args.iterations, "prompt"), f)
                                    for f in (0.025, 0.975)],
        }
    result["backends"] = detailed

    official_index = {(p["video_a_uid"], p["video_b_uid"]): (p, d) for p, d in official}
    repair_index = {(p["video_a_uid"], p["video_b_uid"]): (p, d) for p, d in repair}
    common = sorted(set(official_index) & set(repair_index))
    dev_keys = [k for k in common if official_index[k][0]["split"] == "dev"]
    test_keys = [k for k in common if official_index[k][0]["split"] == "test"]
    margins = {name: evaluator.calibrate([table[k] for k in dev_keys])
               for name, table in (("official", official_index), ("repair", repair_index))}
    result["paired"] = {"margins": margins, "test_pairs": len(test_keys)}
    for unit in ("pair", "prompt"):
        draws = boot_paired(official_index, repair_index, test_keys, margins, args.iterations, unit)
        point = (evaluator.accuracy([repair_index[k] for k in test_keys], margins["repair"])
                 - evaluator.accuracy([official_index[k] for k in test_keys], margins["official"]))
        result["paired"][f"delta_ci_{unit}"] = [point, percentile(draws, 0.025), percentile(draws, 0.975)]

    official_scores = {row["video_uid"]: evaluator._finite_score(row, False)
                       for row in rows(official_root / "motion_smoothness" / "results.csv")}
    official_scores = {k: v for k, v in official_scores.items() if v is not None}
    repair_scores = {row["video_uid"]: evaluator._finite_score(row, True)
                     for row in rows(nat / "motion_smoothness" / "predictions.csv")}
    repair_scores = {k: v for k, v in repair_scores.items() if v is not None}
    shared = sorted(set(official_scores) & set(repair_scores))
    from scipy.stats import pearsonr, spearmanr
    result["per_video"] = {
        "n": len(shared),
        "spearman_official_repair": float(spearmanr([official_scores[u] for u in shared],
                                                    [repair_scores[u] for u in shared]).statistic),
        "pearson_official_repair": float(pearsonr([official_scores[u] for u in shared],
                                                  [repair_scores[u] for u in shared]).statistic),
        "official_range": [min(official_scores.values()), max(official_scores.values())],
        "repair_range": [min(repair_scores.values()), max(repair_scores.values())],
    }
    audit = {json.loads(line)["video_uid"]: json.loads(line)
             for line in (nat / "motion_smoothness" / "repair_results.jsonl").read_text().splitlines()
             if line.strip() and json.loads(line).get("repair_variant") == "audit"}
    mismatches = [u for u, row in audit.items()
                  if repair_scores.get(u) is None or abs(float(row["repair_score"]) - repair_scores[u]) > 0]
    result["predictions_consistency"] = {"audit_rows": len(audit), "mismatches": len(mismatches)}
    # Reproduce the published paired delta with the repository's own function, so
    # the pair-unit number is verified against the script that produced it.
    published = delta_module.compare("motion_smoothness", nat, official_root, args.iterations, args.seed)
    result["published_paired_pair_unit"] = {
        "delta": published["delta"], "ci": published["delta_ci95"],
        "official_margin": published["official_tie_margin"],
        "repair_margin": published["repair_tie_margin"],
        "test_pairs": published["test_pairs"],
    }
    return result, official_scores, repair_scores


def boot_accuracy(items, delta, seed, iterations, unit):
    draws = boot_indices(items, seed, iterations, unit)

    def acc(index_list):
        return sum(outcome_of(items[i], delta) for i in index_list) / len(index_list)

    return [acc(sample) for sample in draws]


def outcome_of(item, delta):
    pair, difference = item
    if abs(difference) <= delta:
        prediction = 0.5
    else:
        prediction = 1.0 if difference > 0 else 0.0
    return float(prediction == float(pair["human_label"]))


def boot_paired(official_index, repair_index, test_keys, margins, iterations, unit):
    items = [(official_index[k][0], 0.0) for k in test_keys]
    draws = boot_indices(items, 2026, iterations, unit)
    out = []
    for sample in draws:
        keys = [test_keys[i] for i in sample]
        repair_acc = sum(outcome_of(repair_index[k], margins["repair"]) for k in keys) / len(keys)
        official_acc = sum(outcome_of(official_index[k], margins["official"]) for k in keys) / len(keys)
        out.append(repair_acc - official_acc)
    return out


# ---------------------------------------------------------------------------
# MAD baseline and normalisation inversion
# ---------------------------------------------------------------------------

def mad_scores(path: Path, compute: bool) -> dict[str, float] | None:
    if not path.is_file():
        if not compute:
            return None
        raise SystemExit(f"{path} missing and --compute-mad not given")
    payload = json.loads(path.read_text())
    return {uid: -info["mad"] for uid, info in payload.items()}


def mad_section(args, evaluator, official_scores, repair_scores) -> dict[str, Any] | None:
    mad = mad_scores(args.mad_cache, args.compute_mad)
    if mad is None:
        return None
    pairs = [row for row in rows(ROOT / "data/processed/pairwise_master_split.csv")
             if row["dimension"] == "motion_smoothness"]
    items = [(row, mad[row["video_a_uid"]] - mad[row["video_b_uid"]])
             for row in pairs if row["video_a_uid"] in mad and row["video_b_uid"] in mad]
    dev = [i for i in items if i[0]["split"] == "dev"]
    test = [i for i in items if i[0]["split"] == "test"]
    margin = evaluator.calibrate(dev)
    from scipy.stats import spearmanr
    shared = sorted(set(official_scores) & set(repair_scores) & set(mad))
    result = {
        "definition": "mean absolute grayscale frame difference over <=12 frames; higher = smoother (lower MAD)",
        "tie_margin": margin,
        "test_acc0": evaluator.accuracy(test, 0),
        "tie_aware": evaluator.accuracy(test, margin),
        "tie_aware_ci_pair": [percentile(boot_accuracy(test, margin, 2027, args.iterations, "pair"), f)
                              for f in (0.025, 0.975)],
        "tie_aware_ci_prompt": [percentile(boot_accuracy(test, margin, 2027, args.iterations, "prompt"), f)
                                for f in (0.025, 0.975)],
        "spearman_repair_mad": float(spearmanr([repair_scores[u] for u in shared],
                                               [-mad[u] for u in shared]).statistic),
        "spearman_official_mad": float(spearmanr([official_scores[u] for u in shared],
                                                 [-mad[u] for u in shared]).statistic),
    }
    # non-tie direction accuracy
    for name, table in (("lower_mad", mad), ("official", official_scores), ("repair", repair_scores)):
        correct = total = 0
        for row in pairs:
            label = float(row["human_label"])
            if label == 0.5:
                continue
            a, b = row["video_a_uid"], row["video_b_uid"]
            if a not in table or b not in table:
                continue
            total += 1
            prediction = 1.0 if table[a] > table[b] else (0.0 if table[a] < table[b] else 0.5)
            correct += prediction == label
        result[f"non_tie_direction_accuracy_{name}"] = correct / total if total else None
    return result


def normalisation_section(args) -> dict[str, Any] | None:
    clips = cached_clips(args.components)
    if not clips:
        return None
    from scipy.stats import spearmanr
    speeds, scores, ds = [], [], []
    for clip in clips.values():
        scale = float(clip["components"]["speed_scale"])
        magnitude = [e[0] for e in clip["components"]["relative"]]
        direction = [e[0] for e in clip["components"]["direction"]]
        d_t = [0.7 * m + 0.3 * d for m, d in zip(magnitude, direction)]
        data = np.sort(np.asarray(d_t))
        count = max(1, math.ceil(0.1 * data.size))
        d_video = 0.75 * data.mean() + 0.25 * data[-count:].mean()
        speeds.append(scale)
        ds.append(d_video)
        scores.append(math.exp(-d_video))
    return {
        "n": len(speeds),
        "spearman_speed_discontinuity": float(spearmanr(speeds, ds).statistic),
        "spearman_speed_score": float(spearmanr(speeds, scores).statistic),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench"))
    parser.add_argument("--natural-root", type=Path,
                        default=Path("/root/wenbiao_zhao/datasets/natural-preference-runs"))
    parser.add_argument("--components", default="/root/wenbiao_zhao/tmp/ms_components4/shard*.jsonl")
    parser.add_argument("--mad-cache", type=Path,
                        default=Path("/root/wenbiao_zhao/tmp/motion_natural_proxy.json"))
    parser.add_argument("--compute-mad", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    evaluator = load_module("evaluate_pairwise_statistics", ROOT / "scripts/evaluate_pairwise_statistics.py")
    delta_module = load_module("evaluate_paired_backend_delta", ROOT / "scripts/evaluate_paired_backend_delta.py")

    report: dict[str, Any] = {}
    report["revision"] = revision_section(args)

    manifest = [row for row in read_jsonl(args.dataset_root / "manifest.jsonl")
                if row["dimension"] == "motion_smoothness"]
    official_cf = {}
    for row in read_jsonl(args.dataset_root / "scores" / "motion_smoothness__official.jsonl"):
        if row.get("score") is not None:
            official_cf[row["derived_id"]] = float(row["score"])
    repair_cf = {}
    for row in read_jsonl(args.dataset_root / "scores" / "motion_smoothness__repair.jsonl"):
        if row.get("score") is not None:
            repair_cf[row["derived_id"]] = float(row["score"])
    report["counterfactual"] = {
        "official": evaluate_counterfactual(manifest, official_cf, None),
        "repair": evaluate_counterfactual(manifest, repair_cf, official_cf),
        "variance_official": variance_section(manifest, official_cf),
        "variance_repair": variance_section(manifest, repair_cf),
    }
    natural, official_nat, repair_nat = natural_section(args, evaluator, delta_module)
    report["natural"] = natural
    report["mad_baseline"] = mad_section(args, evaluator, official_nat, repair_nat)
    report["normalisation"] = normalisation_section(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
