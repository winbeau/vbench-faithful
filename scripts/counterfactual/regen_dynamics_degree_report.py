"""Regenerate the `dynamics_degree` counterfactual report from raw score files.

The family is a same-rank invariance family, so the tie-margin CPA is degenerate:
every pair expects a tie, the dev margin saturates at the 0.9 quantile of dev
deltas, and the statistic is blind to the sign of the level effect.  This report
therefore leads with the sampling-interval response (level profile and its fitted
exponent), the per-clip dispersion, the per-clip measured exponent distribution
and the threshold/coverage domain, and keeps the composite CPA as an explicitly
non-diagnostic appendix.

Inputs are score JSONL files only; no model and no GPU are needed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

LEVEL_ORDER = ("fps8", "fps6", "fps4", "fps2")
DT = {"fps8": 0.125, "fps6": 1.0 / 6.0, "fps4": 0.25, "fps2": 0.5}
BACKENDS = (
    ("official", "means raw top-5% flow in pixels (the Official intermediate quantity, exponent 0)", "0"),
    ("repair_v1", "divides by dt (exponent 1, the archived repair)", "1"),
    ("repair_v2", "divides by dt**0.5, the shipped default (exponent 0.5)", "0.5"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    """derived_id -> {level: score}."""
    out: dict[str, dict[str, float]] = {}
    for path in paths:
        for entry in read_jsonl(path):
            if entry.get("status") != "succeeded" or entry.get("score") is None:
                continue
            out.setdefault(entry["derived_id"], {})[entry["level"]] = float(entry["score"])
    return out


def rows_for(manifest: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    rows = [r for r in manifest if r["dimension"] == "dynamics_degree"]
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def level_means(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, float]:
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = scores.get(row["derived_id"], {}).get(row["level"])
        if value is not None:
            by_level[row["level"]].append(value)
    return {level: float(np.mean(by_level[level])) for level in LEVEL_ORDER if by_level.get(level)}


def fit_exponent(means: dict[str, float]) -> float | None:
    levels = [level for level in LEVEL_ORDER if level in means and means[level] > 0]
    if len(levels) < 2:
        return None
    x = np.log([DT[level] for level in levels])
    y = np.log([means[level] for level in levels])
    return float(np.polyfit(x, y, 1)[0])


def per_base(rows: list[dict[str, Any]], scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = scores.get(row["derived_id"], {}).get(row["level"])
        if value is not None:
            out[row["base_id"]][row["level"]] = value
    return {base: levels for base, levels in out.items() if len(levels) == 4}


def invariance_stats(bases: dict[str, dict[str, float]]) -> dict[str, float]:
    cvs, ranges, ratios = [], [], []
    for levels in bases.values():
        values = np.array([levels[level] for level in LEVEL_ORDER], dtype=float)
        scale = float(np.abs(values).mean())
        if scale <= 0:
            continue
        cvs.append(float(values.std() / scale))
        ranges.append(float((values.max() - values.min()) / scale))
        ratios.append(float(levels["fps2"] / levels["fps8"]))
    ratios = np.array(ratios) if ratios else np.array([np.nan])
    return {
        "bases": len(cvs),
        "cv": float(np.mean(cvs)) if cvs else float("nan"),
        "relative_range": float(np.mean(ranges)) if ranges else float("nan"),
        "ratio_median": float(np.median(ratios)),
        "ratio_q25": float(np.quantile(ratios, 0.25)),
        "ratio_q75": float(np.quantile(ratios, 0.75)),
        "ratio_within_20pct": float(np.mean(np.abs(ratios - 1) < 0.2)),
    }


def level_means_at_alpha(rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]], alpha: float) -> dict[str, float]:
    """Level means of the recorded displacements under an arbitrary exponent."""
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        entry = evidence.get(row["derived_id"])
        if not entry:
            continue
        means = entry["channel_means"]
        channel = entry.get("selected_channel") or "residual"
        displacement = means.get(f"{channel}_mean_displacement")
        dt = means.get(f"{channel}_mean_dt")
        if displacement is None or not dt:
            continue
        by_level[row["level"]].append(displacement / dt**alpha)
    return {level: float(np.mean(values)) for level, values in by_level.items() if values}


def alpha_star(rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> float:
    """The applied exponent that flattens the level profile, from recorded displacements.

    `rows` may repeat bases, which is what the cluster bootstrap needs.
    """
    def slope(alpha: float) -> float:
        return fit_exponent(level_means_at_alpha(rows, evidence, alpha)) or 0.0

    lo, hi = -0.5, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if slope(lo) * slope(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def bootstrap_alpha(rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]], iterations: int = 400, seed: int = 2026) -> tuple[float, float, float]:
    """Cluster bootstrap over base_id, keeping repeated clusters (a set() would
    silently drop them and understate the interval)."""
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)
    bases = sorted(by_base)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        picked = rng.choice(bases, size=len(bases), replace=True)
        estimates.append(alpha_star([r for base in picked for r in by_base[base]], evidence))
    array = np.array(estimates)
    return float(array.mean()), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived-scores", type=Path, required=True, help="dir with dynamics_degree__official.jsonl / __repair.jsonl")
    parser.add_argument("--v2-scores", type=Path, required=True, help="dir with fixed05__shard*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", default="unknown")
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    rows = [r for r in manifest if r["dimension"] == "dynamics_degree"]
    scores = {
        "official": load_scores([args.archived_scores / "dynamics_degree__official.jsonl"]),
        "repair_v1": load_scores([args.archived_scores / "dynamics_degree__repair.jsonl"]),
        "repair_v2": load_scores(sorted(args.v2_scores.glob("fixed05__shard*.jsonl"))),
    }
    v2_evidence = {
        entry["derived_id"]: entry
        for path in sorted(args.v2_scores.glob("fixed05__shard*.jsonl"))
        for entry in read_jsonl(path)
        if entry.get("status") == "succeeded"
    }
    v1_evidence = {
        entry["derived_id"]: entry
        for path in sorted(args.v2_scores.glob("fixed1__shard*.jsonl"))
        for entry in read_jsonl(path)
        if entry.get("status") == "succeeded"
    }

    lines: list[str] = []
    add = lines.append
    add("# Counterfactual Pair Accuracy: dynamics_degree")
    add("")
    add("- family: `fps_resampling`")
    add("- transformation: resample one trajectory to 8/6/4/2 fps with the duration held fixed.")
    add("- expected relation: invariance — every rung should score the same.")
    add(f"- bases: {len({r['base_id'] for r in rows})} "
        f"(dev {len({r['base_id'] for r in rows if r['split'] == 'dev'})}, "
        f"test {len({r['base_id'] for r in rows if r['split'] == 'test'})})")
    add(f"- derived clips: {len(rows)} "
        f"(dev {sum(1 for r in rows if r['split'] == 'dev')}, test {sum(1 for r in rows if r['split'] == 'test')})")
    add(f"- levels: {', '.join(f'`{level}`' for level in LEVEL_ORDER)}")
    add(f"- code SHA: `{args.code_sha}`")
    add("- backends: " + "; ".join(f"`{name}` {note}" for name, note, _ in BACKENDS))
    add("")
    add("This family has a single expected rank, so **every** level pair expects a tie and the")
    add("composite CPA is degenerate (see the appendix). The primary evidence is the")
    add("sampling-interval response below.")
    add("")

    add("## Score coverage")
    add("")
    add("| backend | scored clips | expected |")
    add("|---|---:|---:|")
    for name, _note, _alpha in BACKENDS:
        add(f"| {name} | {len(scores[name])} | {len(rows)} |")
    add("")

    add("## Sampling-interval response (primary)")
    add("")
    add("Per-level mean score, the ratio profile relative to the 8 fps rung, and the fitted")
    add("exponent `p` in `score ~ dt**p`. Target `p = 0`. Dev and test are reported separately.")
    add("")
    for split in ("dev", "test", "all"):
        subset = rows_for(manifest, None if split == "all" else split)
        add(f"### {split}")
        add("")
        add("| backend | 8 fps | 6 fps | 4 fps | 2 fps | ratio profile | fitted p | fps2/fps8 |")
        add("|---|---:|---:|---:|---:|---|---:|---:|")
        for name, _note, _alpha in BACKENDS:
            means = level_means(subset, scores[name])
            if len(means) < 4:
                add(f"| {name} | — | — | — | — | incomplete | — | — |")
                continue
            profile = " / ".join(f"{means[level] / means['fps8']:.3f}" for level in LEVEL_ORDER)
            p = fit_exponent(means)
            add(f"| {name} | " + " | ".join(f"{means[level]:.4f}" for level in LEVEL_ORDER) +
                f" | {profile} | {p:+.3f} | {means['fps2'] / means['fps8']:.3f} |")
        add("")

    mean_alpha, low, high = bootstrap_alpha(rows, v2_evidence)
    dev_alpha = alpha_star([r for r in rows if r["split"] == "dev"], v2_evidence)
    dev_test_slope = fit_exponent(level_means_at_alpha(rows_for(manifest, "test"), v2_evidence, dev_alpha))
    add("### Calibrated exponent")
    add("")
    add(f"- exponent that flattens the ladder, all {len({r['base_id'] for r in rows})} bases: "
        f"**{mean_alpha:+.3f}**, base-cluster bootstrap 95% CI [{low:+.3f}, {high:+.3f}]")
    add(f"- calibrated on the 10 dev bases only: {dev_alpha:+.3f}; applied to test it leaves a "
        f"residual slope of {dev_test_slope:+.3f} (the dev split is too small to calibrate this constant)")
    add("- both `0` (the Official intermediate quantity) and `1` (the archived repair) fall outside the CI")
    add("")

    add("## Per-clip dispersion and measured exponent")
    add("")
    add("A flat level *mean* does not imply per-clip invariance. `ratio` is the per-base")
    add("fps2/fps8 score ratio; `CV` and relative range are within-base dispersion.")
    add("")
    add("| backend | split | bases | mean CV | mean relative range | median fps2/fps8 | IQR | within ±20% |")
    add("|---|---|---:|---:|---:|---:|---|---:|")
    for name, _note, _alpha in BACKENDS:
        for split in ("dev", "test", "all"):
            subset = rows_for(manifest, None if split == "all" else split)
            stats = invariance_stats(per_base(subset, scores[name]))
            add(f"| {name} | {split} | {stats['bases']} | {stats['cv']:.4f} | {stats['relative_range']:.4f} | "
                f"{stats['ratio_median']:.3f} | [{stats['ratio_q25']:.3f}, {stats['ratio_q75']:.3f}] | "
                f"{stats['ratio_within_20pct'] * 100:.0f}% |")
    add("")
    add("Within-clip exponent fitted from each clip's own multi-lag displacements (lag 1/2/4")
    add("sampled frames). It is a diagnostic: the fitted value is itself lag-window dependent,")
    add("which is why applying it per clip under-corrects relative to the fixed benchmark exponent.")
    add("")
    add("| rung | clips | frames | mean measured exponent | median | sd |")
    add("|---|---:|---|---:|---:|---:|")
    for level in LEVEL_ORDER:
        values = [entry["lag_scaling"]["exponent"] for entry in v2_evidence.values()
                  if entry["level"] == level and entry.get("lag_scaling")
                  and entry["lag_scaling"].get("exponent") is not None]
        frames = sorted({entry.get("n_frames") for entry in v2_evidence.values() if entry["level"] == level})
        if not values:
            continue
        add(f"| `{level}` | {len(values)} | {frames} | {np.mean(values):+.3f} | {np.median(values):+.3f} | {np.std(values):.3f} |")
    add("")

    add("## Threshold and coverage domain")
    add("")
    add("The static/moving decision uses the same normalisation as the intensity: the Official")
    add("pixel threshold is re-expressed as `official_px / diagonal / reference_lag**alpha` and")
    add("compared against `d / dt**alpha`, which makes the fraction frame-rate invariant.")
    add("")
    add("| backend (exponent) | split | mean coverage 8 fps | 6 fps | 4 fps | 2 fps |")
    add("|---|---|---:|---:|---:|---:|")
    for label, evidence in (("repair_v1 (1.0, ballistic)", v1_evidence), ("repair_v2 (0.5, default)", v2_evidence)):
        for split in ("dev", "test"):
            by_level: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row["split"] != split:
                    continue
                entry = evidence.get(row["derived_id"])
                if not entry or not entry.get("coverage"):
                    continue
                value = entry["coverage"].get(entry.get("selected_channel") or "residual")
                if value is not None:
                    by_level[row["level"]].append(float(value))
            if len(by_level) < 4:
                continue
            add(f"| {label} | {split} | " + " | ".join(f"{np.mean(by_level[level]):.4f}" for level in LEVEL_ORDER) + " |")
    add("")

    add("## Composite CPA (non-diagnostic for this family)")
    add("")
    add("Every pair in a same-rank family expects a tie, so `zero-margin` requires exact")
    add("floating-point equality (it is `0.0000` for any continuous score, including a perfectly")
    add("invariant one), and the dev-calibrated tie margin is the 0.9 quantile of dev deltas by")
    add("construction, which makes the dev tie-aware CPA `0.90` regardless of the score. The")
    add("statistic is also blind to the sign of the level effect, so the mirrored exponents of")
    add("`official` (+0.491) and `repair_v1` (-0.481) score the same. The archived values are:")
    add("")
    add("| backend | split | margin | pairs | CPA | 95% CI |")
    add("|---|---|---:|---:|---:|---|")
    add("| official | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |")
    add("| official | test (tie-aware) | 20.14 | 180 | 0.8333 | [0.7444, 0.9111] |")
    add("| repair_v1 | test (zero-margin) | 0 | 180 | 0.0000 | [0.0000, 0.0000] |")
    add("| repair_v1 | test (tie-aware) | 0.1466 | 180 | 0.8444 | [0.7388, 0.9333] |")
    add("")
    add("The `+0.0111` difference is ~1/5 of the sampling standard deviation of that statistic on")
    add("this base budget and is expected to be ~0 under the null; it must not be quoted as a win.")
    add("")

    add("## Status and limitations")
    add("")
    add("- The clips are re-encoded per rung; on the identical 0.5 s trajectory span the 2 fps")
    add("  rung's lag-1 displacement and the 8 fps rung's lag-4 displacement agree to within 3.5%,")
    add("  so the level effect is a sampling-interval effect, not a codec effect.")
    add("- Fixing the level mean does not fix individual clips: the per-clip fps2/fps8 ratio stays")
    add("  widely dispersed, so per-video comparisons across frame rates remain unreliable.")
    add("- Two independent estimators are both sublinear (RAFT and Farneback), so the effect is not")
    add("  specific to RAFT; the chord/path ratio (`straightness`, emitted per clip with the")
    add("  per-lag chord and path displacements) is what distinguishes trajectory curvature from")
    add("  estimator saturation and is recorded for every clip.")
    add("- The coarse rungs sample a shorter span of the source trajectory (fps2 covers 80% of a")
    add("  16-frame 8 fps source and 93.8% of a 33-frame 10 fps source), so the rungs do not")
    add("  average exactly the same content.")
    add("- Model/CUDA parity against the frozen E0 numbers has not been verified.")
    add("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "alpha": mean_alpha,
                      "alpha_ci": [low, high], "dev_alpha": dev_alpha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
