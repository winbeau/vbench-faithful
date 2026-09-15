"""Run one dimension's counterfactual evaluation end to end and write its report.

For a single dimension this:
  1. scores every clip with the Official and Repair backends, fanning the clips
     out over the given physical GPUs (one shard per GPU, logical cuda:0 inside);
  2. retries any shard whose worker died or left the shard incomplete;
  3. merges the shards into one file per backend;
  4. computes zero-margin and dev-calibrated tie-aware CPA with a cluster
     bootstrap over base_id;
  5. writes a Markdown report under `reports/`.

Running one dimension at a time keeps each result reviewable before the next
starts (AGENTS.md multi-card rule) and keeps the expensive model setup isolated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np

from .common import ROOT, read_jsonl, write_json
from .cpa import (
    BUDGET,
    _base_pairs,
    bootstrap_ci,
    calibrate_margin,
    cpa_at_margin,
    evaluate_method,
    family_pairs,
    group_stats,
    order_statistics,
    paired_bootstrap_ci,
    rank_gap_groups,
)

# Levels that are controls rather than rungs of the ordered ladder. They are
# reported by their own predicate and must not enter the rank ordering: the
# plan's conjunction control is a never-co-present clip, not a severity step,
# so counting it as `occlusion_100`'s equal rewards a comparison that is not a
# severity ordering at all (plan 11.4).
CONTROL_LEVELS = ("conjunction_control",)

# GPU 6 is visible to nvidia-smi but not CUDA-available in this container.
DEFAULT_GPUS = "1,2,3,4,5"
BACKENDS = ("official", "repair")
FAMILY_DESCRIPTION = {
    "fps_resampling": (
        "Resample one trajectory to 8/6/4/2 fps with the duration held fixed.",
        "Invariance: every rung should score the same.",
    ),
    "temporal_relocation": (
        "One fixed subject-region corruption placed at the start, middle and end.",
        "clean > corrupted, and the three positions should tie.",
    ),
    "filename_invariance": (
        "Byte-identical copies named with a correct, wrong and neutral action.",
        "Invariance: only the filename changes.",
    ),
    "directional_flip": (
        "Mirror the axis named by the ordered relation, prompt unchanged.",
        "original > flip.",
    ),
    "environment_coverage": (
        "2x2 grid mixing a wrong-scene donor with target-scene quadrants at 0-100%.",
        "Monotone increasing in target coverage.",
    ),
    "weakest_object_visibility": (
        "Alpha-blend the tracked weaker target towards its local mean at 0-100%.",
        "Monotone decreasing as the weak target disappears.",
    ),
    "temporal_jerk": (
        "Duplicate, skip and locally reverse short segments; frame count and rate fixed.",
        "Monotone decreasing in perturbation severity.",
    ),
}


def expected_shard_counts(rows: list[dict[str, Any]], shards: int) -> list[int]:
    return [len(rows[index::shards]) for index in range(shards)]


def shard_complete(path: Path, expected: int) -> int:
    if not path.is_file():
        return 0
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["derived_id"])
    return len(ids)


def run_shard(
    python: str,
    repo: Path,
    dimension: str,
    backend: str,
    gpu: str,
    index: int,
    shards: int,
    manifest: Path,
    dataset_root: Path,
    annotations_root: Path,
    upstream: Path,
    out: Path,
    log: Path,
    env: dict[str, str],
) -> subprocess.Popen:
    command = [
        python, "-u", "-B", "-m", "scripts.counterfactual.score",
        "--dimension", dimension, "--backend", backend,
        "--manifest", str(manifest), "--dataset-root", str(dataset_root),
        "--annotations-root", str(annotations_root),
        "--output", str(out), "--upstream", str(upstream),
        "--shard-index", str(index), "--num-shards", str(shards),
    ]
    child_env = dict(env)
    child_env["CUDA_VISIBLE_DEVICES"] = gpu
    handle = log.open("w")
    return subprocess.Popen(command, cwd=repo, env=child_env, stdout=handle, stderr=subprocess.STDOUT)


def score_backend(
    dimension: str,
    backend: str,
    gpus: list[str],
    manifest: Path,
    dataset_root: Path,
    annotations_root: Path,
    upstream: Path,
    scores: Path,
    python: str,
    repo: Path,
    env: dict[str, str],
    retries: int = 2,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest) if row["dimension"] == dimension]
    expected = expected_shard_counts(rows, len(gpus))
    pending = list(range(len(gpus)))
    attempt = 0
    while pending and attempt <= retries:
        attempt += 1
        running = {}
        for index in pending:
            out = scores / f"{dimension}__{backend}__shard{index}.jsonl"
            log = scores / f"{dimension}__{backend}__shard{index}.log"
            running[index] = run_shard(
                python, repo, dimension, backend, gpus[index], index, len(gpus),
                manifest, dataset_root, annotations_root, upstream, out, log, env,
            )
        for index, process in running.items():
            process.wait()
        pending = [
            index for index in range(len(gpus))
            if shard_complete(scores / f"{dimension}__{backend}__shard{index}.jsonl", expected[index]) < expected[index]
        ]
        if pending:
            print(f"  [{backend}] retry {attempt}: incomplete shards {pending}", flush=True)
            time.sleep(3)

    merged = scores / f"{dimension}__{backend}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for index in range(len(gpus)):
        for row in read_jsonl(scores / f"{dimension}__{backend}__shard{index}.jsonl"):
            latest[row["derived_id"]] = row
    with merged.open("w", encoding="utf-8") as handle:
        for key in sorted(latest):
            handle.write(json.dumps(latest[key], ensure_ascii=False, sort_keys=True) + "\n")

    succeeded = sum(1 for row in latest.values() if row.get("status") == "succeeded" and row.get("score") is not None)
    return {
        "backend": backend,
        "expected_clips": len(rows),
        "scored_clips": succeeded,
        "merged_rows": len(latest),
        "merged": str(merged),
        "incomplete_shards": pending,
    }


def score_profile(rows: list[dict[str, Any]], backend_scores: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    """Per-level score distribution, to separate invariance from insensitivity.

    A metric that emits one constant value is trivially "invariant" (plan
    section 7.4 explicitly refuses to call that a success), so the report has to
    show whether the metric responds to the transformation at all.
    """
    by_level: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = backend_scores.get(row["derived_id"])
        if value is not None:
            by_level[row["level"]].append(float(value))
    profile: dict[str, dict[str, Any]] = {}
    for level, values in sorted(by_level.items()):
        array = np.asarray(values, dtype=float)
        profile[level] = {
            "n": len(values),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "distinct": int(len(set(values))),
        }
    return profile


def discontinuity_profile(
    rows: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]] | None,
    backend: str,
) -> dict[str, dict[str, Any]]:
    """Per-level mean and tail discontinuity from a backend's own diagnostics.

    Plan 13.3 requires mean discontinuity and tail discontinuity next to the
    score profile.  They exist only when the backend records them, so a backend
    whose scorer does not persist them produces an empty profile instead of a
    fabricated zero.
    """
    by_level: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        entry = (evidence or {}).get(row["derived_id"], {}).get(backend)
        if not entry:
            continue
        mean = entry.get("mean_discontinuity")
        tail = entry.get("tail_discontinuity")
        if mean is None or tail is None:
            continue
        by_level[row["level"]].append((float(mean), float(tail)))
    profile: dict[str, dict[str, Any]] = {}
    for level, values in sorted(by_level.items()):
        means = np.asarray([value[0] for value in values], dtype=float)
        tails = np.asarray([value[1] for value in values], dtype=float)
        profile[level] = {
            "n": len(values),
            "mean_discontinuity": float(means.mean()),
            "tail_discontinuity": float(tails.mean()),
            "std_mean": float(means.std()),
        }
    return profile


def invariance_stats(
    rows: list[dict[str, Any]],
    backend_scores: dict[str, float | None],
    levels: set[str] | None = None,
) -> dict[str, Any]:
    """Within-base coefficient of variation and relative range.

    For an invariance family (every level sharing one expected rank) a
    tie-margin CPA is degenerate: the margin can always be widened until every
    difference counts as a tie, which scores 1.0 while hiding the very
    instability the family exists to detect. Plan sections 6.4 and 8.3 ask for
    the dispersion statistics instead, so they are reported alongside.

    `levels` restricts the dispersion to a declared-equal subset of a
    mixed-rank family (for `temporal_relocation`, the three corrupted
    positions): within that subset the contract is still "tie", so the same
    degeneracy applies and the same statistic is the right one.
    """
    by_base: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if levels is not None and row["level"] not in levels:
            continue
        value = backend_scores.get(row["derived_id"])
        if value is not None:
            by_base[row["base_id"]].append(float(value))
    cvs: list[float] = []
    ranges: list[float] = []
    for values in by_base.values():
        array = np.asarray(values, dtype=float)
        if len(array) < 2:
            continue
        scale = float(np.abs(array).mean())
        if scale <= 0:
            continue
        cvs.append(float(array.std() / scale))
        ranges.append(float((array.max() - array.min()) / scale))
    return {
        "n_bases": len(cvs),
        "mean_cv": float(np.mean(cvs)) if cvs else None,
        "mean_relative_range": float(np.mean(ranges)) if ranges else None,
    }


def tied_level_groups(rows: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Levels grouped by `expected_rank`, largest group last, for mixed-rank families."""
    by_rank: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        by_rank[row["expected_rank"]].add(row["level"])
    return {rank: sorted(levels) for rank, levels in sorted(by_rank.items())}



def contract_split_cpa(
    rows: list[dict[str, Any]],
    backend_scores: dict[str, float | None],
    margin: float,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """CPA split by contract half, with a cluster bootstrap CI for each half.

    A family may mix two contracts: an inequality half (the counterfactual must
    move the score) and an invariance half (relocated variants must tie). A
    single pooled CPA is dominated by whichever half is easier, so each is
    scored on its own pairs *and* carries its own interval: the composite's
    interval says nothing about either half.  `expected` comes from
    `family_pairs`.
    """
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_base[row["base_id"]][row["level"]] = {
            "score": backend_scores.get(row["derived_id"]),
            "expected_rank": row["expected_rank"],
        }
    halves: dict[str, dict[str, list[tuple[int, int, float]]]] = {"sensitivity": {}, "invariance": {}}
    for base_id, levels in by_base.items():
        for expected, rank_gap, delta in family_pairs(levels):
            halves["invariance" if expected == 0 else "sensitivity"].setdefault(base_id, []).append(
                (expected, rank_gap, delta)
            )
    out: dict[str, Any] = {}
    for name, base_pairs in halves.items():
        pairs = [pair for group in base_pairs.values() for pair in group]
        if not pairs:
            out[name] = {"n_pairs": 0, "cpa": None}
            continue
        low, high = bootstrap_ci(base_pairs, margin, iterations, seed)
        out[name] = {
            "n_pairs": len(pairs),
            "n_bases": len(base_pairs),
            "cpa": round(cpa_at_margin(pairs, margin), 4),
            "cpa_zero_margin": round(cpa_at_margin(pairs, 0.0), 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
        }
    return out


def contract_predicate_stats(
    rows: list[dict[str, Any]],
    backend_scores: dict[str, float | None],
    control_level: str,
    reference_level: str,
    *,
    incomplete_threshold: float | None = None,
) -> dict[str, Any]:
    """Plan section 11.4 statistic: is the conjunction control scored incomplete?

    The plan does not ask for the control to *tie* a full-occlusion clip — it
    asks that a clip in which no frame holds both targets is scored as
    incomplete, and section 11.4 states the reason explicitly: it guards against
    a repair that turns same-frame conjunction into temporal union.  That is a
    level predicate plus an ordering against the ladder's worst rung, not an
    exact-tie CPA, so it is measured directly here.
    """
    by_base: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = backend_scores.get(row["derived_id"])
        if value is not None:
            by_base[row["base_id"]][row["level"]] = float(value)
    control: list[float] = []
    reference: list[float] = []
    not_higher = 0
    below_threshold = 0
    for levels in by_base.values():
        if control_level not in levels or reference_level not in levels:
            continue
        control.append(levels[control_level])
        reference.append(levels[reference_level])
        not_higher += int(levels[control_level] <= levels[reference_level])
        if incomplete_threshold is not None:
            below_threshold += int(levels[control_level] <= incomplete_threshold)
    if not control:
        return {"n_bases": 0}
    return {
        "n_bases": len(control),
        "control_mean": round(float(np.mean(control)), 4),
        "reference_mean": round(float(np.mean(reference)), 4),
        "control_not_higher_rate": round(not_higher / len(control), 4),
        "control_below_threshold_rate": (
            round(below_threshold / len(control), 4) if incomplete_threshold is not None else None
        ),
        "incomplete_threshold": incomplete_threshold,
        "control_level": control_level,
        "reference_level": reference_level,
    }


def lag_exponent(rows: list[dict[str, Any]], backend_scores: dict[str, float | None]) -> dict[str, Any]:
    """Signed log-log slope of score against inter-frame interval.

    For an FPS-invariance family the contract is `target exponent = 0`: the score
    must not depend on the sampling interval. Unlike an unsigned dispersion
    measure, the sign distinguishes a metric that inflates at low frame rates
    from one that shrinks, so this is the diagnostic the family exists to supply.
    """
    import re as _re

    by_level: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = backend_scores.get(row["derived_id"])
        if value is not None and value > 0:
            by_level[row["level"]].append(float(value))
    points: list[tuple[float, float]] = []
    level_means: dict[str, float] = {}
    for level, values in by_level.items():
        match = _re.fullmatch(r"fps(\d+)", level)
        if not match or not values:
            continue
        fps = float(match.group(1))
        dt = 1.0 / fps
        mean = float(np.mean(values))
        level_means[level] = mean
        if mean > 0:
            points.append((float(np.log(dt)), float(np.log(mean))))
    if len(points) < 2:
        return {"p": None, "levels": level_means}
    xs = np.array([pt[0] for pt in points])
    ys = np.array([pt[1] for pt in points])
    slope = float(np.polyfit(xs, ys, 1)[0])
    # Per-clip slope: how the effect looks within a single base.
    by_base: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        value = backend_scores.get(row["derived_id"])
        match = _re.fullmatch(r"fps(\d+)", row["level"])
        if value is None or value <= 0 or not match:
            continue
        by_base[row["base_id"]].append((float(np.log(1.0 / float(match.group(1)))), float(np.log(value))))
    per_clip = []
    for pts in by_base.values():
        if len(pts) >= 2:
            bx = np.array([pt[0] for pt in pts])
            by = np.array([pt[1] for pt in pts])
            per_clip.append(float(np.polyfit(bx, by, 1)[0]))
    return {
        "p": round(slope, 4),
        "levels": {k: round(v, 4) for k, v in sorted(level_means.items())},
        "per_clip_p_mean": round(float(np.mean(per_clip)), 4) if per_clip else None,
        "per_clip_p_std": round(float(np.std(per_clip)), 4) if per_clip else None,
        "n_clips_fitted": len(per_clip),
    }



def weak_target_area_profile(rows: list[dict[str, Any]], split: str = "test") -> dict[str, Any]:
    """Per-base area of the suppressed target, as a fraction of the frame.

    The family's premise is that target B becomes progressively invisible, so the
    report has to say how much of the frame B actually occupies: an occlusion of a
    ~2%-of-frame box is a very different intervention from one covering a third of
    it, and a base whose box is nearly the whole frame makes the ladder trivial.
    """
    seen: set[str] = set()
    ratios: list[float] = []
    for row in rows:
        # The conjunction control carries boxes_a/boxes_b, not the suppressed
        # target's `boxes`, and it is not a rung of the occlusion ladder.
        if row["level"] in CONTROL_LEVELS:
            continue
        if row["split"] != split or row["base_id"] in seen:
            continue
        seen.add(row["base_id"])
        parameters = row.get("transformation_parameters") or {}
        boxes = parameters.get("boxes") or []
        width, height = row.get("width"), row.get("height")
        if not boxes or not width or not height:
            continue
        areas = [max(0, int(b[2]) - int(b[0])) * max(0, int(b[3]) - int(b[1])) for b in boxes]
        ratios.append(float(np.median(areas)) / float(width * height))
    if not ratios:
        return {}
    array = np.asarray(ratios, dtype=float)
    return {
        "n_bases": len(ratios),
        "median": round(float(np.median(array)), 5),
        "q25": round(float(np.percentile(array, 25)), 5),
        "q75": round(float(np.percentile(array, 75)), 5),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
    }


def render_report(
    dimension: str,
    family: str,
    rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    cpa: dict[str, Any],
    code_sha: str,
    scores: dict[tuple[str, str], float | None] | None = None,
    repair_config: dict[str, Any] | None = None,
    evidence: dict[str, dict[str, Any]] | None = None,
) -> str:
    transformation, expectation = FAMILY_DESCRIPTION.get(family, ("", ""))
    levels = sorted({row["level"] for row in rows}, key=lambda name: next(r["expected_rank"] for r in rows if r["level"] == name))
    splits = {
        split: sum(1 for row in rows if row["split"] == split) for split in ("dev", "test")
    }
    lines = [
        f"# Counterfactual Pair Accuracy: {dimension}",
        "",
        f"- family: `{family}`",
        f"- transformation: {transformation}",
        f"- expected relation: {expectation}",
        f"- bases: {len({row['base_id'] for row in rows})} "
        f"(dev {len({r['base_id'] for r in rows if r['split'] == 'dev'})}, "
        f"test {len({r['base_id'] for r in rows if r['split'] == 'test'})})",
        f"- derived clips: {len(rows)} (dev {splits['dev']}, test {splits['test']})",
        f"- levels: {', '.join(f'`{level}`' for level in levels)}",
        *(
            [f"- control levels (excluded from the ordered ladder): "
             + ", ".join(f'`{l}`' for l in sorted({r["level"] for r in rows} & set(CONTROL_LEVELS)))]
            if {r["level"] for r in rows} & set(CONTROL_LEVELS) else []
        ),
        f"- code SHA: `{code_sha}`",
        *(
            [
                f"- repair variant: `{repair_config['repair_mode']}`"
                f" (detection-conditioned: {str(bool(repair_config['detection_conditioned'])).lower()})",
            ]
            if repair_config
            else []
        ),
        "",
        "## Score coverage",
        "",
        "| backend | scored clips | expected | incomplete shards |",
        "|---|---:|---:|---|",
    ]
    for entry in coverage:
        lines.append(
            f"| {entry['backend']} | {entry['scored_clips']} | {entry['expected_clips']} | "
            f"{entry['incomplete_shards'] or 'none'} |"
        )
    for entry in cpa.get("coverage", []):
        lines.append(
            f"| {entry['backend']} ({entry['split']}) | {entry['scored']} | {entry['total']} | — |"
        )
    ranks = {row["expected_rank"] for row in rows}
    if "fps" in "".join(levels):
        lines += [
            "",
            "## Sampling-interval response (primary)",
            "",
            "The contract for this family is `score must not depend on the sampling",
            "interval`, i.e. a signed log-log slope of `p = 0`. This is the primary",
            "diagnostic: unlike the unsigned dispersion below it can tell a score that",
            "inflates at low frame rates from one that shrinks.",
            "",
            "| backend | fitted p (target 0) | mean per-clip p | sd | levels (score vs rung) |",
            "|---|---:|---:|---:|---|",
        ]
        for backend in ("official", "repair"):
            info = cpa.get("lag_exponent", {}).get(backend, {})
            if info.get("p") is None:
                continue
            detail = ", ".join(f"{k}={v:.4f}" for k, v in info["levels"].items())
            lines.append(
                f"| {backend} | {info['p']:+.4f} | {info.get('per_clip_p_mean')} | "
                f"{info.get('per_clip_p_std')} | {detail} |"
            )

    # A family is a *contract mixture* only when it declares more than one rank
    # *and* at least one rank is shared by several levels (a tie contract).  A
    # pure invariance family has one rank, and a purely ordered family has one
    # level per rank; neither is a mixture, so the mixed-contract template does
    # not describe them.
    rank_groups = tied_level_groups(rows)
    dispersion_rows = [(rank, levels) for rank, levels in rank_groups.items() if len(levels) > 1]
    mixed_family = len(ranks) > 1 and bool(dispersion_rows)
    if mixed_family:
        lines += [
            "",
            "## CPA by contract half",
            "",
            "This family declares at least one group of levels that must tie, so its",
            "CPA mixes an inequality half (the counterfactual must move the score)",
            "with an invariance half (the declared-equal levels must tie). A pooled",
            "CPA is dominated by whichever half is easier, so each is scored",
            "separately, and each half carries its own cluster-bootstrap interval:",
            "the composite's interval says nothing about either half.",
            "",
            "| backend | split | half | pairs | CPA (dev margin) | CPA (zero margin) | 95% CI |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for backend in ("official", "repair"):
            for split in ("dev", "test"):
                for half, stats in ((cpa.get("contract_split", {}).get(backend, {}) or {}).get(split, {}) or {}).items():
                    if stats.get("cpa") is None:
                        continue
                    interval = (
                        f"[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
                        if stats.get("ci_low") is not None
                        else "—"
                    )
                    lines.append(
                        f"| {backend} | {split} | {half} | {stats['n_pairs']} | {stats['cpa']:.4f} | "
                        f"{stats['cpa_zero_margin']:.4f} | {interval} |"
                    )
        paired_halves = (cpa.get("paired_halves", {}) or {})
        if any(entry.get("delta") is not None for entry in paired_halves.values()):
            lines += [
                "",
                "Per-half paired difference (the same `base_id` clusters resampled once",
                "and both backends re-scored on each resample, plan 5.4). This is the",
                "interval that decides a half; the two marginal intervals above overlap.",
                "",
                "| half | repair − official | paired 95% CI | bases |",
                "|---|---:|---|---:|",
            ]
            for half in ("sensitivity", "invariance"):
                entry = paired_halves.get(half) or {}
                if entry.get("delta") is None:
                    continue
                lines.append(
                    f"| {half} | {entry['delta']:+.4f} | "
                    f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] | {entry['n_bases']} |"
                )
        predicate = {
            backend: (cpa.get("conjunction_control", {}) or {}).get(backend, {})
            for backend in ("official", "repair")
        }
        if any(entry.get("n_bases") for entry in predicate.values()):
            reference = next(
                (entry["reference_level"] for entry in predicate.values() if entry.get("n_bases")),
                "occlusion_100",
            )
            lines += [
                "",
                "### Conjunction control — the plan's actual predicate",
                "",
                "Plan section 11.4 does not ask the temporal-conjunction control to",
                "*tie* the fully occluded clip. It asks that a clip in which no frame",
                "holds both targets is scored as incomplete, because that is what",
                "catches a repair which silently converts same-frame conjunction into",
                "temporal union. Equality between two different corruption geometries",
                "on a continuous score is not achievable by any continuous estimator —",
                "and a hard 0/1 metric reaches it only by saturating, which is what the",
                "Official column below does. The predicate is therefore reported",
                "directly, next to the tie-based CPA that misrepresents it.",
                "",
                f"| backend | bases | control mean | `{reference}` mean | control ≤ `{reference}` |",
                "|---|---:|---:|---:|---:|",
            ]
            for backend in ("official", "repair"):
                entry = predicate[backend]
                if not entry.get("n_bases"):
                    continue
                lines.append(
                    f"| {backend} | {entry['n_bases']} | {entry['control_mean']:.4f} | "
                    f"{entry['reference_mean']:.4f} | {entry['control_not_higher_rate']:.4f} |"
                )
            lines += [
                "",
                "Read the tie-based invariance CPA as *not applicable* whenever the two",
                "levels in the declared-equal group are different corruption geometries:",
                "an exact tie requires the metric to be blind to the difference between",
                "them, which is a property of saturation rather than of contract",
                "fidelity.",
            ]

    area = cpa.get("weak_target_area")
    if area:
        lines += [
            "",
            "## Weak-target construction check",
            "",
            "Median area of the suppressed target B as a fraction of the frame, over",
            "the test bases. This is the premise check for the occlusion ladder: the",
            "severity levels only mean something if B occupies enough of the frame",
            "that suppressing it is visible at all.",
            "",
            f"- bases: {area['n_bases']}",
            f"- median area ratio: **{area['median']:.4f}** (IQR {area['q25']:.4f}–{area['q75']:.4f})",
            f"- range: {area['min']:.4f} – {area['max']:.4f}",
            "",
        ]

    lines += [
        "",
        "## CPA",
        "",
        "`zero-margin` predicts the sign directly; `tie-aware` uses the dev-calibrated",
        "margin. Intervals are 95% cluster-bootstrap CIs over `base_id`.",
        "",
        "| backend | split | margin | pairs | CPA | 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for backend in ("official", "repair"):
        payload = cpa.get(backend)
        if not payload:
            continue
        for split, key in (("dev", "dev_zero_margin"), ("test", "test_zero_margin"), ("test", "test_tie_aware")):
            for family_name, stats in sorted(payload.get(key, {}).items()):
                if stats.get("cpa") is None:
                    continue
                margin = payload.get("dev_margin") if key == "test_tie_aware" else 0.0
                label = "tie-aware" if key == "test_tie_aware" else "zero-margin"
                lines.append(
                    f"| {backend} | {split} ({label}) | {margin:.4g} | {stats['n_pairs']} | "
                    f"{stats['cpa']:.4f} | [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}] |"
                )
    ranks = {row["expected_rank"] for row in rows}
    if len(ranks) > 1:
        lines += [
            "",
            "## Sequence-level order statistics",
            "",
            "Per-base Spearman correlation between the declared rank and the score,",
            "and the fraction of bases whose levels come out in the declared strict",
            "order (plan 5.2 and 13.3). Unlike CPA this does not weight small rank",
            "gaps more heavily; levels the family declares equal are not required",
            "to be strictly ordered, and a base with a missing score is excluded",
            "rather than counted as a violation.",
            "",
            "| backend | split | bases | mean Spearman | median Spearman | strict order |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for backend in ("official", "repair"):
            for split in ("dev", "test"):
                stats = (cpa.get("order", {}).get(backend, {}) or {}).get(split)
                if not stats or not stats.get("n_bases"):
                    continue
                mean = "—" if stats["mean_spearman"] is None else f"{stats['mean_spearman']:.4f}"
                median = "—" if stats["median_spearman"] is None else f"{stats['median_spearman']:.4f}"
                lines.append(
                    f"| {backend} | {split} | {stats['n_bases']} | {mean} | {median} | "
                    f"{stats['strict_order_bases']}/{stats['n_bases']} "
                    f"({stats['strict_order_rate']:.4f}) |"
                )
    if len(ranks) == 1:
        lines += [
            "",
            "## Invariance statistics",
            "",
            "Every level of this family carries the same expected rank, so the",
            "tie-aware CPA above is degenerate — widening the dev margin until all",
            "differences count as ties yields CPA 1.0 regardless of how unstable the",
            "metric is. The dispersion below is the meaningful invariance measure",
            "(plan sections 6.4 and 8.3).",
            "",
            "| backend | split | bases | mean within-base CV | mean relative range |",
            "|---|---|---:|---:|---:|",
        ]
        for backend in ("official", "repair"):
            for split, split_rows in (("dev", [r for r in rows if r["split"] == "dev"]),
                                      ("test", [r for r in rows if r["split"] == "test"])):
                # `scores` is the nested {derived_id: {backend: score}} shape here.
                backend_scores = {
                    r["derived_id"]: (scores or {}).get(r["derived_id"], {}).get(backend)
                    for r in split_rows
                }
                stats = invariance_stats(split_rows, backend_scores)
                if stats["mean_cv"] is None:
                    continue
                lines.append(
                    f"| {backend} | {split} | {stats['n_bases']} | {stats['mean_cv']:.4f} | "
                    f"{stats['mean_relative_range']:.4f} |"
                )

    if mixed_family:
        lines += [
            "",
            "## Contract decomposition",
            "",
            "This family declares levels that must tie, so its CPA is a mixture of",
            "two contracts and is dominated by whichever is easier. Rank",
            "gap > 0 pairs test sensitivity; rank gap 0 pairs test the invariance of",
            "the levels declared equal, and there the only correct prediction is a",
            "tie, so a widening dev margin raises this half without measuring",
            "anything. Read the two halves separately, never the composite alone.",
            "",
            "| backend | split | rank gap | pairs | match rate | tie rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for backend in ("official", "repair"):
            for split, key in (("test (zero-margin)", "test_zero_margin"),
                               ("test (tie-aware)", "test_tie_aware")):
                contract = cpa.get("contracts", {}).get(backend, {}).get(key, {})
                for rank_gap, stats in sorted(contract.items()):
                    lines.append(
                        f"| {backend} | {split} | {rank_gap} | {stats['n_pairs']} | "
                        f"{stats['match_rate']:.4f} | {stats['tie_rate']:.4f} |"
                    )
        lines += [
            "",
            "Rank-gap-0 pairs are the declared-equal contract. Splitting them out",
            "shows whether a Repair gain in the composite comes from sensitivity",
            "(which both backends usually already have) or from the invariant half —",
            "but an exact tie is only a contract test when the levels in the group",
            "really are interchangeable. When they differ in more than severity (for",
            "example a same-frame occlusion against a clip in which no frame holds",
            "both targets), the tie rate measures the metric's saturation, not its",
            "fidelity, and the predicate section below is the test that applies.",
        ]
        if dispersion_rows:
            lines += [
                "",
                "### Declared-equal subgroups — dispersion, not CPA",
                "",
                "Same degeneracy as a same-rank family, applied to each declared-equal",
                "group: the tie-margin CPA of these pairs can be pushed to 1.0 by",
                "widening the margin, so the within-base CV is the meaningful number.",
                "",
                "| backend | split | levels | bases | mean within-base CV | mean relative range |",
                "|---|---|---|---:|---:|---:|",
            ]
            for backend in ("official", "repair"):
                for split, split_rows in (
                    ("dev", [r for r in rows if r["split"] == "dev"]),
                    ("test", [r for r in rows if r["split"] == "test"]),
                ):
                    backend_scores = {
                        r["derived_id"]: (scores or {}).get(r["derived_id"], {}).get(backend)
                        for r in split_rows
                    }
                    for rank, levels in dispersion_rows:
                        stats = invariance_stats(split_rows, backend_scores, set(levels))
                        if stats["mean_cv"] is None:
                            continue
                        lines.append(
                            f"| {backend} | {split} | rank {rank}: "
                            f"{', '.join(f'`{level}`' for level in levels)} | "
                            f"{stats['n_bases']} | {stats['mean_cv']:.4f} | "
                            f"{stats['mean_relative_range']:.4f} |"
                        )

    sensitivity_note = (
        [
            "Per-level score distribution. A metric with a single distinct value is",
            "insensitive rather than invariant: it cannot detect the transformation at",
            "all, so its CPA on an invariance family is vacuous (plan section 7.4).",
        ]
        if len(ranks) == 1
        else [
            "Per-level score distribution. A metric with a single distinct value at",
            "every level is insensitive rather than ordered: it cannot detect the",
            "transformation at all, so no level pair can match.",
        ]
    )
    lines += [
        "",
        "## Score sensitivity",
        "",
        *sensitivity_note,
        "",
        "| backend | split | level | n | mean | std | min | max | distinct |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in ("official", "repair"):
        for split in ("dev", "test"):
            for level, stats in (cpa.get("profiles", {}).get(backend, {}) or {}).get(split, {}).items():
                lines.append(
                    f"| {backend} | {split} | `{level}` | {stats['n']} | {stats['mean']:.4f} | "
                    f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['distinct']} |"
                )
    for backend in ("official", "repair"):
        for split in ("dev", "test"):
            profile = (cpa.get("profiles", {}).get(backend, {}) or {}).get(split, {})
            present = {stats["distinct"] for stats in profile.values()}
            if present == {1} and profile:
                lines.append(
                    f"\n**Warning:** the `{backend}` backend returned a single constant score at every "
                    f"level on the {split} split, so its CPA there measures insensitivity, not invariance."
                )

    continuity = cpa.get("discontinuity", {}).get("repair", {})
    if continuity:
        lines += [
            "",
            "## Repair continuity components (dev + test)",
            "",
            "The repair's own mean and upper-tail discontinuity per level (plan 13.3).",
            "`D_video = (1 - tail_weight) * D_mean + tail_weight * D_tail`, so a level",
            "whose mean is flat while its tail rises is a localised failure the score",
            "alone would hide.",
            "",
            "| level | n | mean D_mean | mean D_tail | sd(D_mean) |",
            "|---|---:|---:|---:|---:|",
        ]
        for level, stats in continuity.items():
            lines.append(
                f"| `{level}` | {stats['n']} | {stats['mean_discontinuity']:.4f} | "
                f"{stats['tail_discontinuity']:.4f} | {stats['std_mean']:.4f} |"
            )

    paired_zero = cpa.get("paired", {}).get("zero_margin", {})
    paired_tie = cpa.get("paired", {}).get("tie_aware", {})
    ci_low = paired_zero.get("ci_low")
    ci_high = paired_zero.get("ci_high")
    paired_ci = "—" if ci_low is None else f"[{ci_low:+.4f}, {ci_high:+.4f}]"
    lines += [
        "",
        "## Official vs Repair (test, tie-aware)",
        "",
        "`repair - official` is the paired difference over the same `base_id`",
        "clusters; its 95% CI resamples those clusters once and re-scores both",
        "backends on each resample (plan 5.4). The interval is what decides whether",
        "the delta is distinguishable from zero — the two marginal intervals in the",
        "`CPA` table do not.",
        "",
        "| metric | official | repair | repair - official | paired 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for family_name in sorted({row["family"] for row in rows}):
        off = cpa.get("official", {}).get("test_tie_aware", {}).get(family_name, {})
        rep = cpa.get("repair", {}).get("test_tie_aware", {}).get(family_name, {})
        if off.get("cpa") is None and rep.get("cpa") is None:
            continue
        delta = (
            f"{rep['cpa'] - off['cpa']:+.4f}"
            if off.get("cpa") is not None and rep.get("cpa") is not None
            else "n/a"
        )
        lines.append(
            f"| {family_name} | {off.get('cpa', float('nan')):.4f} | "
            f"{rep.get('cpa', float('nan')):.4f} | {delta} | {paired_ci} |"
        )
    if paired_zero.get("n_bases"):
        lines += [""]
        if paired_tie.get("delta") is not None:
            lines.append(
                f"Paired zero-margin delta: `{paired_zero['delta']:+.4f}` over "
                f"{paired_zero['n_bases']} test bases; paired tie-aware delta: "
                f"`{paired_tie['delta']:+.4f}`."
            )
        else:
            lines.append(
                f"Paired zero-margin delta: `{paired_zero['delta']:+.4f}` over "
                f"{paired_zero['n_bases']} test bases."
            )
    lines += render_evidence_section(rows, evidence, dimension)

    lines += [
        "",
        "## Status and limitations",
        "",
        "- Weights are local; nothing was downloaded and no upstream checkout was modified.",
        "- Model/CUDA parity against the frozen E0 numbers has **not** been verified: these",
        "  scores come from the current metric packages, so treat absolute values as",
        "  first-run measurements rather than reproduced official baselines.",
        "- A clip whose backend raised is recorded as `failed` and stays out of the CPA",
        "  denominator rather than being scored as zero.",
        "",
    ]
    return "\n".join(lines)


def evidence_summary(
    rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]] | None, backend: str
) -> dict[str, Any]:
    """Aggregate the per-clip frame evidence recorded by the scoring driver.

    A repair that floors at zero and a repair whose direction is inverted both
    report a low mean; only the frame reasons tell them apart, so the report has
    to carry them next to the CPA.
    """

    reasons: dict[str, int] = {}
    clips_with_evidence = frames = 0
    zero_fractions: list[float] = []
    joint_rates: list[float] = []
    mean_scores: list[float] = []
    mean_weakest: list[float] = []
    for row in rows:
        entry = (evidence or {}).get(row["derived_id"], {}).get(backend)
        if not entry:
            continue
        clips_with_evidence += 1
        for reason, count in (entry.get("frame_reason_counts") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
        frames += int(entry.get("frame_count") or sum((entry.get("frame_reason_counts") or {}).values()))
        if entry.get("zero_frame_fraction") is not None:
            zero_fractions.append(float(entry["zero_frame_fraction"]))
        if entry.get("joint_detection_rate") is not None:
            joint_rates.append(float(entry["joint_detection_rate"]))
        if entry.get("mean_frame_score") is not None:
            mean_scores.append(float(entry["mean_frame_score"]))
        if entry.get("mean_weakest_confidence") is not None:
            mean_weakest.append(float(entry["mean_weakest_confidence"]))
    satisfied = sum(count for reason, count in reasons.items() if reason.startswith("relation_satisfied"))
    return {
        "backend": backend,
        "clips": clips_with_evidence,
        "frames": frames,
        "missing_subject": reasons.get("missing_subject", 0),
        "missing_object": reasons.get("missing_object", 0),
        "direction_mismatch": reasons.get("direction_mismatch", 0),
        "axis_mismatch": reasons.get("axis_mismatch", 0),
        "satisfied": satisfied,
        "mean_joint_detection_rate": float(np.mean(joint_rates)) if joint_rates else None,
        "mean_frame_score": float(np.mean(mean_scores)) if mean_scores else None,
        "mean_weakest_confidence": float(np.mean(mean_weakest)) if mean_weakest else None,
        "other": sum(reasons.values()) - sum(
            reasons.get(reason, 0)
            for reason in ("missing_subject", "missing_object", "direction_mismatch", "axis_mismatch")
        ) - satisfied,
        "mean_zero_frame_fraction": (
            sum(zero_fractions) / len(zero_fractions) if zero_fractions else None
        ),
    }


def render_evidence_section(
    rows: list[dict[str, Any]], evidence: dict[str, dict[str, Any]] | None, dimension: str | None = None
) -> list[str]:
    # The frame-reason table is a detector diagnostic; a continuity dimension
    # (Motion Smoothness) records mean/tail discontinuity instead and reports it
    # in its own section, so the detector wording would be misleading here.
    if dimension in {"motion_smoothness"}:
        return []
    test_rows = [row for row in rows if row["split"] == "test"]
    summaries = {
        backend: evidence_summary(test_rows, evidence, backend) for backend in ("official", "repair")
    }
    if not any(summary["clips"] for summary in summaries.values()):
        return [
            "",
            "## Frame evidence",
            "",
            "No per-clip evidence was recorded for this run, so a low repair score cannot be",
            "attributed to detector drop-outs rather than to a wrong direction. Re-score with the",
            "current `score.py` to populate it.",
        ]
    if dimension == "multiplt_object":
        # The generic frame-reason table is empty for this backend: its evidence
        # is per-target confidence and a co-presence decision, which is exactly
        # what shows whether a suppressed target is *absent* or merely
        # low-confidence.
        lines = [
            "",
            "## Frame evidence (test split)",
            "",
            "Per-clip detection evidence behind the scores. `joint co-presence` is the",
            "fraction of sampled frames in which both targets pass the official 0.5",
            "gate, read from the separate official-threshold pass for the Repair;",
            "`mean frame score` is the repair's SoftMin value averaged over frames",
            "(the official score is that same co-presence rate, so it has no separate",
            "SoftMin column), and `weakest confidence` is the mean per-frame confidence",
            "of the weaker target. A repair whose `weakest confidence` stays well above",
            "zero at full suppression is reading context, not the object.",
            "",
            "| backend | clips | frames | mean joint co-presence | mean frame score | mean weakest confidence | zero-frame rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for backend, summary in summaries.items():
            joint = (
                "—" if summary.get("mean_joint_detection_rate") is None
                else f"{summary['mean_joint_detection_rate']:.4f}"
            )
            frame_score = (
                "—" if summary.get("mean_frame_score") is None
                else f"{summary['mean_frame_score']:.4f}"
            )
            weakest = (
                "—" if summary.get("mean_weakest_confidence") is None
                else f"{summary['mean_weakest_confidence']:.4f}"
            )
            zero = (
                "—" if summary["mean_zero_frame_fraction"] is None
                else f"{summary['mean_zero_frame_fraction']:.4f}"
            )
            lines.append(
                f"| {backend} | {summary['clips']} | {summary['frames']} | {joint} | "
                f"{frame_score} | {weakest} | {zero} |"
            )
        return lines
    lines = [
        "",
        "## Frame evidence (test split)",
        "",
        "Per-frame reasons behind each backend's scores. `missing_*` frames are detector",
        "drop-outs, `direction_mismatch` is a resolved arrangement with the wrong sign, and",
        "`axis_mismatch` is a resolved arrangement on the wrong axis. The official backend",
        "reports raw frame scores instead of reasons, so only its zero-frame rate is shown.",
        "",
        "| backend | clips | frames | missing_subject | missing_object | direction_mismatch | axis_mismatch | other | satisfied | mean zero-frame rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, summary in summaries.items():
        zero = (
            "—" if summary["mean_zero_frame_fraction"] is None
            else f"{summary['mean_zero_frame_fraction']:.4f}"
        )
        lines.append(
            f"| {backend} | {summary['clips']} | {summary['frames']} | {summary['missing_subject']} | "
            f"{summary['missing_object']} | {summary['direction_mismatch']} | "
            f"{summary['axis_mismatch']} | {summary['other']} | {summary['satisfied']} | {zero} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="derived counterfactual dataset holding the clips")
    parser.add_argument("--annotations-root", type=Path, default=None,
                        help="source VBench dataset holding annotations/")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the report from cached scores without scoring")
    parser.add_argument(
        "--repair-mode",
        default=os.environ.get("VBENCH_AUDIT_SPATIAL_MODE", "ordered_role_identity_assignment"),
        help="Spatial Relationship repair variant forwarded to the scoring workers",
    )
    parser.add_argument(
        "--repair-detection-conditioned",
        action="store_true",
        default=os.environ.get("VBENCH_AUDIT_SPATIAL_DETECTION_CONDITIONED", "").lower() in {"1", "true", "yes"},
        help="score the Spatial Relationship repair only over frames where both roles were detected",
    )
    args = parser.parse_args()

    if args.annotations_root is None:
        args.annotations_root = args.dataset_root
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    args.scores.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_jsonl(args.manifest) if row["dimension"] == args.dimension]
    if not rows:
        raise SystemExit(f"no manifest rows for {args.dimension}")
    family = rows[0]["family"]

    env = dict(os.environ)
    env["VBENCH_AUDIT_SPATIAL_MODE"] = args.repair_mode
    env["VBENCH_AUDIT_SPATIAL_DETECTION_CONDITIONED"] = "1" if args.repair_detection_conditioned else "0"
    repair_config = {
        "repair_mode": args.repair_mode,
        "detection_conditioned": bool(args.repair_detection_conditioned),
    }
    coverage = []
    for backend in ([] if args.report_only else BACKENDS):
        print(f"[{args.dimension}] scoring {backend} on GPUs {gpus}", flush=True)
        coverage.append(
            score_backend(
                args.dimension, backend, gpus, args.manifest, args.dataset_root,
                args.annotations_root, args.upstream, args.scores, args.python, ROOT, env,
            )
        )

    hollow = [entry for entry in coverage if entry["scored_clips"] == 0]
    if hollow:
        raise SystemExit(
            "refusing to report: no clip scored for "
            + ", ".join(entry["backend"] for entry in hollow)
            + " (check --annotations-root and the shard logs)"
        )

    if args.report_only:
        coverage = [
            {"backend": backend,
             "expected_clips": len(rows),
             "scored_clips": sum(1 for r in read_jsonl(args.scores / f"{args.dimension}__{backend}.jsonl")
                                  if r.get("status") == "succeeded" and r.get("score") is not None),
             "merged_rows": 0, "merged": str(args.scores / f"{args.dimension}__{backend}.jsonl"),
             "incomplete_shards": []}
            for backend in BACKENDS
        ]

    scores: dict[tuple[str, str], float | None] = {}
    nested_scores: dict[str, dict[str, float]] = {}
    nested_evidence: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        for row in read_jsonl(args.scores / f"{args.dimension}__{backend}.jsonl"):
            if row.get("status") == "succeeded" and row.get("score") is not None:
                scores[(row["derived_id"], backend)] = float(row["score"])
                nested_scores.setdefault(row["derived_id"], {})[backend] = float(row["score"])
            if row.get("evidence"):
                nested_evidence.setdefault(row["derived_id"], {})[backend] = row["evidence"]

    cpa: dict[str, Any] = {"coverage": [], "profiles": {}, "contracts": {}, "order": {}, "paired": {}}
    test_pairs_by_backend: dict[str, dict[str, list[tuple[int, int, float]]]] = {}
    margins: dict[str, float] = {}
    for backend in BACKENDS:
        backend_scores = {row["derived_id"]: scores.get((row["derived_id"], backend)) for row in rows}
        # Rank-based statistics see the ladder only; the control level is handled
        # by its own predicate below.
        ladder = [row for row in rows if row["level"] not in CONTROL_LEVELS]
        dev_rows = [row for row in ladder if row["split"] == "dev"]
        test_rows = [row for row in ladder if row["split"] == "test"]
        margin = calibrate_margin(
            [pair for group in _base_pairs(dev_rows, backend_scores).values() for pair in group]
        )
        margins[backend] = margin
        cpa[backend] = {
            "dev_margin": margin,
            "dev_zero_margin": evaluate_method(dev_rows, backend_scores, 0.0, args.iterations, args.seed),
            "test_zero_margin": evaluate_method(test_rows, backend_scores, 0.0, args.iterations, args.seed),
            "test_tie_aware": evaluate_method(test_rows, backend_scores, margin, args.iterations, args.seed),
        }
        # Rank-gap split: a mixed-rank family's composite CPA is a mixture of a
        # sensitivity contract and a tie contract, and only the latter is the
        # invariance target, so it must be readable on its own.
        test_base_pairs = _base_pairs(test_rows, backend_scores)
        test_pairs_by_backend[backend] = test_base_pairs
        cpa["contracts"][backend] = {
            "test_zero_margin": group_stats(rank_gap_groups(test_base_pairs), 0.0),
            "test_tie_aware": group_stats(rank_gap_groups(test_base_pairs), margin),
        }
        cpa["profiles"][backend] = {
            "dev": score_profile(dev_rows, backend_scores),
            "test": score_profile(test_rows, backend_scores),
        }
        cpa.setdefault("discontinuity", {})[backend] = discontinuity_profile(
            rows, nested_evidence, backend
        )
        cpa.setdefault("contract_split", {})[backend] = {
            split: contract_split_cpa(
                split_rows, backend_scores, margin, args.iterations, args.seed
            )
            for split, split_rows in (("dev", dev_rows), ("test", test_rows))
        }
        cpa.setdefault("conjunction_control", {})[backend] = contract_predicate_stats(
            rows, backend_scores, "conjunction_control", "occlusion_100"
        )
        cpa.setdefault("lag_exponent", {})[backend] = lag_exponent(rows, backend_scores)
        if not cpa.get("weak_target_area"):
            cpa["weak_target_area"] = weak_target_area_profile(rows)
        # Sequence-level order statistics answer the same question as CPA
        # without the pair-count weighting (plan 5.2 and 13.3).
        cpa["order"][backend] = {
            "dev": order_statistics(dev_rows, backend_scores),
            "test": order_statistics(test_rows, backend_scores),
        }
        for split, split_rows in (("dev", [r for r in rows if r["split"] == "dev"]),
                                  ("test", [r for r in rows if r["split"] == "test"])):
            scored = sum(1 for row in split_rows if backend_scores.get(row["derived_id"]) is not None)
            cpa["coverage"].append({"backend": backend, "split": split, "scored": scored, "total": len(split_rows)})

    # Paired difference over the *same* bases: the two marginal intervals alone
    # cannot say whether the delta is distinguishable from zero (plan 5.4).
    cpa["paired"] = {
        "zero_margin": paired_bootstrap_ci(
            test_pairs_by_backend["official"], test_pairs_by_backend["repair"],
            0.0, 0.0, args.iterations, args.seed,
        ),
        "tie_aware": paired_bootstrap_ci(
            test_pairs_by_backend["official"], test_pairs_by_backend["repair"],
            margins["official"], margins["repair"], args.iterations, args.seed,
        ),
    }
    # The composite interval hides which contract moved, so repeat the paired
    # bootstrap for each half over **all** bases the halves are pooled on: the
    # halves table pools dev+test, and a test-only interval would not describe it.
    all_rows = rows
    all_scores = {row["derived_id"]: scores.get((row["derived_id"], "official")) for row in all_rows}
    repair_scores = {row["derived_id"]: scores.get((row["derived_id"], "repair")) for row in all_rows}
    official_all = _base_pairs(all_rows, all_scores)
    repair_all = _base_pairs(all_rows, repair_scores)
    cpa["paired_halves"] = {}
    for half in ("sensitivity", "invariance"):
        expected = 0 if half == "invariance" else 1
        official_half = {
            base_id: [pair for pair in pairs if (pair[0] == 0) == (expected == 0)]
            for base_id, pairs in official_all.items()
        }
        repair_half = {
            base_id: [pair for pair in pairs if (pair[0] == 0) == (expected == 0)]
            for base_id, pairs in repair_all.items()
        }
        official_half = {key: value for key, value in official_half.items() if value}
        repair_half = {key: value for key, value in repair_half.items() if value}
        cpa["paired_halves"][half] = paired_bootstrap_ci(
            official_half, repair_half,
            margins["official"], margins["repair"], args.iterations, args.seed,
        )

    from .build import git_sha
    code_sha = git_sha()  # scoring revision, not the dataset-build revision in the manifest
    args.scores.mkdir(parents=True, exist_ok=True)
    write_json(args.scores / f"{args.dimension}__cpa.json", cpa)
    report = render_report(
        args.dimension, family, rows, coverage, cpa, code_sha, nested_scores, repair_config, nested_evidence
    )
    (args.reports / f"{args.dimension}.md").write_text(report, encoding="utf-8")
    print(json.dumps({"dimension": args.dimension, "coverage": coverage, "report": str(args.reports / f"{args.dimension}.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
