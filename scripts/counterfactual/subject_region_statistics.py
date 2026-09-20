"""Preregistered paired source-prompt bootstrap for region discrimination.

Sensitivity gates are analysis exclusions only. They never feed construction
or base selection. Negative background drops are kept, never clipped to zero.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from .region_discrimination import LEVELS, POSITIONS


BACKENDS = ("official", "aggregation", "masked_zero", "masked_exclude")


def paired_bootstrap(values: dict[str, dict[str, float]], groups: dict[str, str], *,
                     resamples: int = 10000, seed: int = 20260920, statistic_name: str = "R") -> dict:
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not values:
        return {}
    backends = list(values)
    ids = sorted(values[backends[0]])
    if any(set(column) != set(ids) for column in values.values()):
        raise ValueError("paired bootstrap columns must use the same bases")
    clusters = defaultdict(list)
    for index, base in enumerate(ids):
        clusters[groups[base]].append(index)
    cluster_ids = sorted(clusters)
    matrix = np.asarray([[values[name][base] for base in ids] for name in backends], dtype=float)
    if matrix.size and not np.isfinite(matrix).all():
        raise ValueError("ratios must be finite")
    if len(cluster_ids) < 2:
        return {name: {statistic_name: float(np.median(matrix[i])) if ids else None, "ci95": None,
                       "n_bases": len(ids), "n_prompt_clusters": len(cluster_ids), "status": "insufficient_clusters"}
                for i, name in enumerate(backends)}
    rng = np.random.default_rng(seed)
    samples = np.empty((len(backends), resamples), dtype=float)
    # One set of sampled cluster indices for every backend; preserve all base
    # members of a repeated prompt cluster, not independent frame resampling.
    for iteration in range(resamples):
        selected = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        indices = [i for cluster in selected for i in clusters[cluster_ids[int(cluster)]]]
        samples[:, iteration] = np.median(matrix[:, indices], axis=1)
    return {name: {statistic_name: float(np.median(matrix[i])), "ci95": np.quantile(samples[i], [.025, .975]).tolist(),
                   "n_bases": len(ids), "n_prompt_clusters": len(cluster_ids), "status": "measured"}
            for i, name in enumerate(backends)}


def analyze(index: list[dict], scores: list[dict], *, resamples: int = 10000, seed: int = 20260920) -> dict:
    ids = {row["base_id"] for row in index}
    if len(ids) != len(index):
        raise ValueError("duplicate dataset base IDs")
    expected = {(base, pos, level, backend) for base in ids for pos in POSITIONS for level in LEVELS for backend in BACKENDS}
    records = {}
    for row in scores:
        key = (row["base_id"], row["position"], row["level"], row["backend"])
        if key in records or key not in expected:
            raise ValueError(f"duplicate or unexpected score: {key}")
        if row.get("status") == "succeeded":
            score = row.get("score")
            if score is None or not math.isfinite(score) or not -.000001 <= score <= 1.000001:
                raise ValueError(f"invalid succeeded score: {key}")
        records[key] = row
    if set(records) != expected:
        raise ValueError(f"incomplete score grid: {len(expected - set(records))} missing rows")
    groups = {row["base_id"]: row["source_prompt_id"] for row in index}
    report = {"total_preregistered_bases": len(ids), "construction_rejection_counts": dict(Counter(
        reason for row in index for reason in row["rejection_reasons"])),
        "positions": {}, "minimum_subject_drop": .05, "bootstrap_unit": "source_prompt_id",
        "bootstrap_seed": seed, "bootstrap_resamples": resamples,
        "natural_preference_measurement": "NOT RUN", "real_model_parity": "NOT RUN"}
    for position in POSITIONS:
        drops, failures, aggregates = {}, {}, {}
        for backend in BACKENDS:
            drops[backend], failures[backend] = {}, {}
            aggregates[backend] = {}
            for level in LEVELS:
                rows = [records[(base, position, level, backend)] for base in sorted(ids)]
                good = [row for row in rows if row["status"] == "succeeded"]
                coverage = [row["diagnostics"] for row in rows if row.get("diagnostics") and "num_present_frames" in row["diagnostics"]]
                total_frames = sum(d["num_frames"] for d in coverage)
                aggregates[backend][level] = {
                    "fixed_denominator_mean": (sum(row["score"] for row in good) / len(rows)
                                               if rows and not any(row["status"] == "not_run" for row in rows) else None),
                    "fixed_denominator": len(rows),
                    "conditional_mean": float(np.mean([row["score"] for row in good])) if good else None,
                    "conditional_denominator": len(good), "status_counts": dict(Counter(row["status"] for row in rows)),
                    "mask_present_frame_fraction": sum(d["num_present_frames"] for d in coverage) / total_frames if total_frames else None,
                    "mean_mask_area_fraction": float(np.mean([v for d in coverage for v in d["coverage"]])) if coverage else None,
                }
            for base in sorted(ids):
                rows = {level: records[(base, position, level, backend)] for level in LEVELS}
                if any(row["status"] != "succeeded" for row in rows.values()):
                    failures[backend][base] = "unsupported_or_failed"
                    continue
                subject = rows["clean"]["score"] - rows["subject_corrupt"]["score"]
                background = rows["clean"]["score"] - rows["background_corrupt"]["score"]
                if subject <= 0:
                    failures[backend][base] = "sensitivity_failed"
                elif subject < .05:
                    failures[backend][base] = "subject_drop_below_0.05"
                else:
                    drops[backend][base] = {"subject_drop": subject, "background_drop": background,
                                            "ratio": background / subject}
        comparisons, changes = {}, {}
        for policy in ("zero", "exclude"):
            columns = ("official", "aggregation", f"masked_{policy}")
            common = set.intersection(*(set(drops[name]) for name in columns))
            ratios = {name: {base: drops[name][base]["ratio"] for base in sorted(common)} for name in columns}
            bootstrap = paired_bootstrap(ratios, groups, resamples=resamples, seed=seed)
            comparisons[policy] = {"common_base_ids": sorted(common), "excluded_base_ids": sorted(ids - common),
                                   "statistics": bootstrap,
                                   "equal_area_criterion": "NOT APPLICABLE: whole background and subject have unequal areas"}
            observed = sorted(base for base in ids if all(
                records[(base, position, level, name)]["status"] == "succeeded"
                for name in columns for level in ("clean", "background_corrupt")))
            signed = {name: {base: records[(base, position, "clean", name)]["score"] -
                                  records[(base, position, "background_corrupt", name)]["score"]
                             for base in observed} for name in columns}
            absolute = {name: {base: abs(value) for base, value in column.items()} for name, column in signed.items()}
            absolute["official_minus_masked"] = {base: absolute["official"][base] - absolute[f"masked_{policy}"][base]
                                                  for base in observed}
            absolute_stats = paired_bootstrap(absolute, groups, resamples=resamples, seed=seed, statistic_name="median")
            signed_stats = paired_bootstrap(signed, groups, resamples=resamples, seed=seed, statistic_name="median")
            masked_ci = absolute_stats[f"masked_{policy}"]["ci95"]
            reduction_ci = absolute_stats["official_minus_masked"]["ci95"]
            criterion = (masked_ci[1] < .02 and reduction_ci[0] > 0) if masked_ci and reduction_ci else None
            changes[policy] = {"common_base_ids": observed, "n_bases": len(observed),
                              "per_base_signed_drop": signed, "absolute_change": absolute_stats,
                              "signed_drop": signed_stats, "stability_tolerance": .02,
                              "background_stability_criterion_met": criterion,
                              "sensitivity_warning": "Small background change is insufficient without subject-corruption sensitivity."}
        report["positions"][position] = {"aggregates": aggregates, "drops": drops, "exclusions": failures,
            "exclusion_counts": {backend: dict(Counter(values.values())) for backend, values in failures.items()},
            "comparisons": comparisons, "background_change": changes,
            "per_column_ratio_diagnostics": {name: paired_bootstrap(
                {name: {base: data["ratio"] for base, data in column.items()}}, groups,
                resamples=resamples, seed=seed)[name] for name, column in drops.items()}}
    report["primary"] = report["positions"]["full"]["background_change"]["zero"]
    return report
