"""Re-verification of the `human_action` conclusions, from the frozen score tree.

CPU only, no GPU and no model: every number here is either a statistic over the
frozen scores or a replay of the *code* that produced them.

  * the split-aware per-level score profile (n / mean / std / min / max /
    distinct) that the report's `Score sensitivity` table prints;
  * the within-base dispersion block, including the bases `invariance_stats`
    silently drops because every level scored zero (`scale <= 0`);
  * the six CPA rows and the dev-calibrated tie margins;
  * the paired `base_id`-cluster bootstrap interval of
    `CPA_repair - CPA_official` at both margins, using the repository's own
    `cpa.paired_bootstrap_ci`;
  * the byte-identity check the `filename_invariance` construction promises:
    every derived file must hash to its source (manifest + files on disk);
  * a replay of the Official target parser over the 75 published filenames, with
    membership of each parsed target in the locked Kinetics-400 list;
  * the published-base census against `configs/counterfactual/bases_published.jsonl`.

Nothing is written into the frozen trees; the only output is stdout plus an
optional JSON report.

Run on the scoring host:
  /root/wenbiao_zhao/venvs/vbench/bin/python -m scripts.counterfactual.verify_human_action_review
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .cpa import (
    _base_pairs,
    calibrate_margin,
    cpa_at_margin,
    evaluate_method,
    family_pairs,
    paired_bootstrap_ci,
)

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("official", "repair")
LEVELS = ("filename_correct", "filename_neutral", "filename_wrong")

# What `docs/counterfactual-reports/human_action.md` prints at code SHA
# `5c4a1309`.  Listed here so a reproduction mismatch is loud rather than silent.
REPORT_PROFILE = {
    ("official", "dev", "filename_correct"): (5, 0.8000, 2),
    ("official", "dev", "filename_neutral"): (5, 0.0000, 1),
    ("official", "dev", "filename_wrong"): (5, 0.0000, 1),
    ("official", "test", "filename_correct"): (20, 0.7500, 2),
    ("official", "test", "filename_neutral"): (20, 0.0000, 1),
    ("official", "test", "filename_wrong"): (20, 0.0000, 1),
    ("repair", "dev", "filename_correct"): (5, 0.9629, 5),
    ("repair", "dev", "filename_neutral"): (5, 0.9629, 5),
    ("repair", "dev", "filename_wrong"): (5, 0.9629, 5),
    ("repair", "test", "filename_correct"): (20, 0.9206, 20),
    ("repair", "test", "filename_neutral"): (20, 0.9206, 20),
    ("repair", "test", "filename_wrong"): (20, 0.9206, 20),
}
REPORT_DISPERSION = {
    ("official", "dev"): (4, 1.4142, 3.0000),
    ("official", "test"): (15, 1.4142, 3.0000),
    ("repair", "dev"): (5, 0.0000, 0.0000),
    ("repair", "test"): (20, 0.0000, 0.0000),
}
REPORT_CPA = {
    ("official", "dev_zero_margin"): (15, 0.4667, 0.3333, 0.7333),
    ("official", "test_zero_margin"): (60, 0.5000, 0.4000, 0.6333),
    ("official", "test_tie_aware"): (60, 1.0000, 1.0000, 1.0000),
    ("repair", "dev_zero_margin"): (15, 1.0000, 1.0000, 1.0000),
    ("repair", "test_zero_margin"): (60, 1.0000, 1.0000, 1.0000),
    ("repair", "test_tie_aware"): (60, 1.0000, 1.0000, 1.0000),
}
REPORT_MARGIN = {"official": 1.0, "repair": 0.0}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(dataset_root: Path, dimension: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["dimension"] == dimension]


def load_scores(scores_root: Path, dimension: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for backend in BACKENDS:
        path = scores_root / f"{dimension}__{backend}.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            if row.get("status") == "succeeded" and row.get("score") is not None:
                out[(row["derived_id"], backend)] = float(row["score"])
    return out


def kinectics400(upstream: Path) -> tuple[str, ...]:
    """The locked category list, in index order, without importing the model."""
    path = upstream / "vbench/third_party/umt/kinetics_400_categories.txt"
    pairs = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(action.lower() for action, _index in sorted(pairs, key=lambda pair: int(pair[1])))


def official_target_from_filename(name: str) -> str:
    """The Official parser, copied verbatim from the locked adapter."""
    return name.split("/")[-1].lower().split("-")[0].split("person is ")[-1].split("_")[0]


def score_map(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float], backend: str) -> dict[str, float | None]:
    return {row["derived_id"]: scores.get((row["derived_id"], backend)) for row in rows}


def profile(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for backend in BACKENDS:
        for split in ("dev", "test"):
            for level in LEVELS:
                values = [
                    scores[(row["derived_id"], backend)]
                    for row in rows
                    if row["split"] == split and row["level"] == level and (row["derived_id"], backend) in scores
                ]
                if not values:
                    continue
                array = np.asarray(values, dtype=float)
                out[(backend, split, level)] = {
                    "n": len(values),
                    "mean": float(array.mean()),
                    "std": float(array.std()),
                    "min": float(array.min()),
                    "max": float(array.max()),
                    "distinct": len(set(values)),
                }
    return out


def dispersion(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float]) -> dict[tuple[str, str], dict[str, Any]]:
    """`invariance_stats`, plus the bases it drops because `scale <= 0`."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for backend in BACKENDS:
        for split in ("dev", "test"):
            by_base: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row["split"] != split:
                    continue
                if (row["derived_id"], backend) in scores:
                    by_base[row["base_id"]].append(scores[(row["derived_id"], backend)])
            cvs, ranges, dropped = [], [], []
            for base, values in sorted(by_base.items()):
                array = np.asarray(values, dtype=float)
                if len(array) < 2:
                    continue
                scale = float(np.abs(array).mean())
                if scale <= 0:
                    dropped.append(base)
                    continue
                cvs.append(float(array.std() / scale))
                ranges.append(float((array.max() - array.min()) / scale))
            out[(backend, split)] = {
                "n_bases": len(cvs),
                "bases_in_manifest": len(by_base),
                "dropped_zero_scale": dropped,
                "mean_cv": float(np.mean(cvs)) if cvs else None,
                "mean_relative_range": float(np.mean(ranges)) if ranges else None,
                "per_base_vectors": {
                    base: sorted(values)
                    for base, values in sorted(by_base.items())
                    if len(values) >= 2
                },
            }
    return out


def cpa_table(
    rows: list[dict[str, Any]], scores: dict[tuple[str, str], float], iterations: int, seed: int
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    margins: dict[str, float] = {}
    for backend in BACKENDS:
        dev_scores = score_map(dev, scores, backend)
        flat = [pair for group in _base_pairs(dev, dev_scores).values() for pair in group]
        margin = calibrate_margin(flat)
        margins[backend] = margin
        for label, subset, used in (
            ("dev_zero_margin", dev, 0.0),
            ("test_zero_margin", test, 0.0),
            ("test_tie_aware", test, margin),
        ):
            stats = evaluate_method(subset, score_map(subset, scores, backend), used, iterations, seed)
            for family, payload in stats.items():
                out[(backend, label)] = {"family": family, **payload}
    return out, margins


def paired_table(
    rows: list[dict[str, Any]], scores: dict[tuple[str, str], float], margins: dict[str, float], iterations: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Paired delta at both margins, `base_id` clusters resampled once for both backends."""
    test = [row for row in rows if row["split"] == "test"]
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in test:
        for backend in BACKENDS:
            by_base[row["base_id"]][backend][row["level"]] = {
                "score": scores.get((row["derived_id"], backend)),
                "expected_rank": row["expected_rank"],
            }
    official_pairs = {base: family_pairs(levels["official"]) for base, levels in by_base.items()}
    repair_pairs = {base: family_pairs(levels["repair"]) for base, levels in by_base.items()}
    out: dict[str, dict[str, Any]] = {}
    for label, official_margin, repair_margin in (
        ("zero_margin", 0.0, 0.0),
        ("tie_aware", margins["official"], margins["repair"]),
    ):
        result = paired_bootstrap_ci(
            official_pairs, repair_pairs, official_margin, repair_margin, iterations, seed
        )
        out[label] = {
            "official_margin": official_margin,
            "repair_margin": repair_margin,
            **result,
        }
    return out


def construction_checks(rows: list[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    """The promises `filename_invariance` makes about its own artefacts."""
    same_output_input = sum(1 for row in rows if row["input_sha256"] == row["output_sha256"])
    same_input_source = sum(1 for row in rows if row["input_sha256"] == row["source_video_sha256"])
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)
    identical = 0
    for group in by_base.values():
        digests = {
            hashlib.sha256((dataset_root / row["output_path"]).read_bytes()).hexdigest() for row in group
        }
        if len(digests) == 1:
            identical += 1
    return {
        "clips": len(rows),
        "output_sha256_equals_input": same_output_input,
        "input_sha256_equals_source": same_input_source,
        "bases_with_byte_identical_files": identical,
        "bases": len(by_base),
        "extensions": dict(Counter(Path(row["output_path"]).suffix for row in rows)),
    }


def parser_census(rows: list[dict[str, Any]], categories: tuple[str, ...]) -> dict[str, Any]:
    """What the Official backend reads off each published filename."""
    out: dict[str, Any] = {}
    for level in LEVELS:
        targets = [official_target_from_filename(row["output_path"]) for row in rows if row["level"] == level]
        counter = Counter(targets)
        out[level] = {
            "n": len(targets),
            "distinct_targets": len(counter),
            "targets_in_kinetics400": sum(count for target, count in counter.items() if target in categories),
            "most_common": counter.most_common(3),
        }
    correct = {
        row["base_id"]: (official_target_from_filename(row["output_path"]), row["transformation_parameters"]["filename_action"])
        for row in rows
        if row["level"] == "filename_correct"
    }
    out["parser_recovers_intended_correct_action"] = sum(
        1 for parsed, intended in correct.values() if parsed == intended
    )
    out["bases"] = len(correct)
    return out


def published_bases(dimension: str) -> set[str]:
    path = ROOT / "configs/counterfactual/bases_published.jsonl"
    return {row["base_id"] for row in read_jsonl(path) if row["dimension"] == dimension}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench/scores"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench"),
    )
    parser.add_argument("--upstream", type=Path, default=Path("/root/wenbiao_zhao/VBench"))
    parser.add_argument("--dimension", default="human_action")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()

    rows = load_manifest(args.dataset_root, args.dimension)
    if not rows:
        raise SystemExit(f"no manifest rows for {args.dimension} under {args.dataset_root}")
    scores = load_scores(args.scores_root, args.dimension)

    report: dict[str, Any] = {"dimension": args.dimension, "clips": len(rows)}

    print("== coverage ==")
    for backend in BACKENDS:
        for split in ("dev", "test"):
            total = sum(1 for row in rows if row["split"] == split)
            scored = sum(
                1 for row in rows if row["split"] == split and (row["derived_id"], backend) in scores
            )
            print(f"  {backend:8s} {split:4s} {scored}/{total}")
    report["coverage"] = {
        f"{backend}/{split}": (
            sum(1 for row in rows if row["split"] == split and (row["derived_id"], backend) in scores),
            sum(1 for row in rows if row["split"] == split),
        )
        for backend in BACKENDS
        for split in ("dev", "test")
    }

    print("\n== construction: is the transformation only a rename? ==")
    construction = construction_checks(rows, args.dataset_root)
    report["construction"] = construction
    for key, value in construction.items():
        print(f"  {key}: {value}")

    print("\n== what the Official parser reads off the published filenames ==")
    census = parser_census(rows, kinectics400(args.upstream))
    report["parser_census"] = census
    for level in LEVELS:
        entry = census[level]
        print(
            f"  {level:18s} n={entry['n']} distinct={entry['distinct_targets']} "
            f"in_K400={entry['targets_in_kinetics400']} top={entry['most_common']}"
        )
    print(
        f"  parser recovers the intended correct action on "
        f"{census['parser_recovers_intended_correct_action']}/{census['bases']} bases"
    )

    print("\n== published-base census ==")
    published = published_bases(args.dimension)
    manifest_bases = {row["base_id"] for row in rows}
    report["published_bases"] = {
        "manifest": len(manifest_bases),
        "published_config": len(published),
        "identical": manifest_bases == published,
    }
    print(
        f"  manifest {len(manifest_bases)} vs bases_published.jsonl {len(published)}: "
        f"{'identical' if manifest_bases == published else 'MISMATCH'}"
    )

    print("\n== per-level profile (frozen tree) ==")
    profiles = profile(rows, scores)
    mismatches: list[str] = []
    for key in sorted(profiles):
        entry = profiles[key]
        print(
            f"  {key[0]:8s} {key[1]:4s} {key[2]:18s} n={entry['n']:2d} mean={entry['mean']:.4f} "
            f"std={entry['std']:.4f} min={entry['min']:.4f} max={entry['max']:.4f} "
            f"distinct={entry['distinct']}"
        )
        expected = REPORT_PROFILE.get(key)
        if expected is None:
            continue
        if (entry["n"], round(entry["mean"], 4), entry["distinct"]) != expected:
            mismatches.append(f"profile {key}: {entry} != report {expected}")
    report["profiles"] = {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in profiles.items()}

    print("\n== within-base dispersion (invariance_stats + dropped bases) ==")
    spread = dispersion(rows, scores)
    for key in sorted(spread):
        entry = spread[key]
        print(
            f"  {key[0]:8s} {key[1]:4s} bases={entry['n_bases']}/{entry['bases_in_manifest']} "
            f"mean_cv={entry['mean_cv']:.4f} mean_rel_range={entry['mean_relative_range']:.4f} "
            f"dropped={entry['dropped_zero_scale']}"
        )
        expected = REPORT_DISPERSION.get(key)
        if expected is not None and (
            entry["n_bases"],
            round(entry["mean_cv"], 4),
            round(entry["mean_relative_range"], 4),
        ) != expected:
            mismatches.append(f"dispersion {key}: {entry} != report {expected}")
    report["dispersion"] = {
        f"{k[0]}/{k[1]}": {kk: vv for kk, vv in v.items() if kk != "per_base_vectors"}
        for k, v in spread.items()
    }

    print("\n== CPA and dev margins ==")
    cpa, margins = cpa_table(rows, scores, args.iterations, args.seed)
    for backend in BACKENDS:
        print(f"  {backend}: dev margin = {margins[backend]:.4g}")
    for key in sorted(cpa):
        entry = cpa[key]
        print(
            f"  {key[0]:8s} {key[1]:18s} pairs={entry['n_pairs']:3d} cpa={entry['cpa']:.4f} "
            f"ci=[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
        )
        expected = REPORT_CPA.get(key)
        if expected is not None and (
            entry["n_pairs"],
            round(entry["cpa"], 4),
            round(entry["ci_low"], 4),
            round(entry["ci_high"], 4),
        ) != expected:
            mismatches.append(f"cpa {key}: {entry} != report {expected}")
    for backend, expected_margin in REPORT_MARGIN.items():
        if round(margins[backend], 6) != expected_margin:
            mismatches.append(f"margin {backend}: {margins[backend]} != report {expected_margin}")
    report["cpa"] = {f"{k[0]}/{k[1]}": v for k, v in cpa.items()}
    report["margins"] = margins

    print("\n== paired delta (both margins) ==")
    paired = paired_table(rows, scores, margins, args.iterations, args.seed)
    for label, entry in paired.items():
        print(
            f"  {label:11s} delta={entry['delta']:+.4f} ci=[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] "
            f"over {entry['n_bases']} test bases"
        )
    report["paired"] = paired

    print("\n== reproduction against the report ==")
    if mismatches:
        print("FAIL:")
        for line in mismatches:
            print(f"  - {line}")
    else:
        print("PASS: every quoted number reproduces")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
