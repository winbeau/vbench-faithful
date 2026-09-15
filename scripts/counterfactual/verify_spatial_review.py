"""Adversarial re-verification of the `spatial_relationship` conclusions.

Recomputes, on CPU only and from the frozen score tree, every number the review
quotes that does not need the detector:

  * the split-aware per-level score profile (n / mean / std / min / max /
    distinct / exact zeros) that the report's `Score sensitivity` table prints;
  * the rank-gap-1 pair census: correct / tied / inverted, match rate and tie
    rate, against the report's `Contract decomposition` table;
  * the per-base order statistics (mean and median Spearman, strict-order rate);
  * the paired `base_id`-cluster bootstrap interval of `CPA_repair - CPA_official`
    at the dev-calibrated margins, using the repository's own
    `cpa.paired_bootstrap_ci`;
  * an exact two-sided sign test on the pairs whose two backends disagree, to
    ask whether Official's non-tied differences carry any directional signal at
    all (they cannot, if `get_position_score` never reads the sign);
  * how many bases have *any* non-zero repair score, i.e. how much of the family
    is measurable at all.

Nothing is written into the frozen trees; the only output is a JSON report plus
the summary on stdout.

Run on the scoring host:
  /root/wenbiao_zhao/venvs/vbench/bin/python -m scripts.counterfactual.verify_spatial_review
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .cpa import (
    _base_pairs,
    calibrate_margin,
    group_stats,
    order_statistics,
    paired_bootstrap_ci,
    rank_gap_groups,
)

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("official", "repair")

# What `docs/counterfactual-reports/spatial_relationship.md` prints at code SHA
# 5c4a1309.  Listed so a reproduction mismatch is loud rather than silent.
REPORT_PROFILE = {
    ("official", "dev", "horizontal_flip"): (7, 0.5080, 7),
    ("official", "dev", "original"): (10, 0.3908, 7),
    ("official", "dev", "vertical_flip"): (3, 0.2488, 3),
    ("official", "test", "horizontal_flip"): (21, 0.2451, 12),
    ("official", "test", "original"): (30, 0.2836, 17),
    ("official", "test", "vertical_flip"): (9, 0.3597, 9),
    ("repair", "dev", "horizontal_flip"): (7, 0.0446, 3),
    ("repair", "dev", "original"): (10, 0.1428, 3),
    ("repair", "dev", "vertical_flip"): (3, 0.0000, 1),
    ("repair", "test", "horizontal_flip"): (21, 0.0268, 2),
    ("repair", "test", "original"): (30, 0.0122, 4),
    ("repair", "test", "vertical_flip"): (9, 0.0694, 3),
}
REPORT_TEST_CPA = {"official": 0.3667, "repair": 0.0667}
REPORT_PAIRED = {"delta": -0.3000, "ci_low": -0.4667, "ci_high": -0.1333}


def read_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_map(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {row["derived_id"]: row.get("score") for row in rows}


def profile(rows: list[dict[str, Any]], scores: dict[str, float | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "test"):
        for level in sorted({row["level"] for row in rows}):
            values = [
                float(value) for row in rows
                if row["split"] == split and row["level"] == level
                for value in [scores.get(row["derived_id"])]
                if value is not None
            ]
            if not values:
                continue
            array = np.array(values, dtype=np.float64)
            out[f"{split}/{level}"] = {
                "split": split,
                "level": level,
                "n": len(values),
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
                "distinct": int(len(set(values))),
                "exact_zero": int(sum(1 for value in values if value == 0.0)),
            }
    return out


def pair_census(
    rows: list[dict[str, Any]], scores: dict[str, float | None], margin: float
) -> dict[str, Any]:
    base_pairs = _base_pairs([row for row in rows if row["split"] == "test"], scores)
    stats = group_stats(rank_gap_groups(base_pairs), margin)
    gap1 = stats.get(1, {})
    ordered = [pair for pairs in base_pairs.values() for pair in pairs if pair[1] > 0]
    correct = sum(1 for expected, _gap, delta in ordered if delta > margin)
    inverted = sum(1 for expected, _gap, delta in ordered if delta < -margin)
    tied = len(ordered) - correct - inverted
    return {
        "n_pairs": len(ordered),
        "correct": correct,
        "tied": tied,
        "inverted": inverted,
        "match_rate": gap1.get("match_rate"),
        "tie_rate": gap1.get("tie_rate"),
        "per_base": {
            base: [round(delta, 6) for _e, _g, delta in pairs if _g > 0]
            for base, pairs in base_pairs.items()
        },
    }


def sign_test(census: dict[str, Any]) -> dict[str, Any]:
    """Two-sided exact test that the non-tied pairs split 50/50.

    Official cannot read the sign of the geometry, so its non-tied differences
    are re-encoding and detector noise.  Under that null the discordant pairs are
    a fair coin; a p-value near 1 is the *expected* outcome, and a small one
    would mean some other systematic effect is present.
    """

    from scipy.stats import binomtest

    discordant = census["correct"] + census["inverted"]
    if discordant == 0:
        return {"discordant": 0, "p_value": None}
    return {
        "discordant": discordant,
        "prefer_original": census["correct"],
        "prefer_flip": census["inverted"],
        "p_value": float(binomtest(census["correct"], discordant, 0.5).pvalue),
    }


def evidence_report(paths: list[Path], frozen: Path | None) -> int:
    """Summarise the frame evidence recorded by a re-scored repair run.

    `score.py` now keeps a per-clip `frame_reason` histogram, so the detector
    floor the review talks about can be recomputed from score rows instead of a
    bespoke probe.  With `--frozen` the same run is also checked against the
    frozen tree, which is the reproducibility control for the re-score itself.
    """

    for path in paths:
        rows = read_scores(path)
        print(f"\n=== {path} ({len(rows)} rows) ===")
        if frozen is not None:
            reference = {
                row["derived_id"]: row.get("score")
                for row in read_scores(frozen / "spatial_relationship__repair.jsonl")
            }
            identical = sum(
                1 for row in rows if reference.get(row["derived_id"], "missing") == row.get("score")
            )
            print(f"score reproduction against the frozen tree: {identical}/{len(rows)} identical")
        for label, selector in (
            ("original level only", lambda row: row["level"] == "original"),
            ("all levels", lambda row: True),
        ):
            selected = [row for row in rows if selector(row)]
            reasons: Counter[str] = Counter()
            frames = 0
            for row in selected:
                counts = (row.get("evidence") or {}).get("frame_reason_counts") or {}
                reasons.update(counts)
                frames += sum(counts.values())
            if not frames:
                print(f"\n-- {label}: no frame evidence recorded")
                continue
            drop = reasons["missing_subject"] + reasons["missing_object"]
            resolved = reasons["direction_mismatch"] + reasons["axis_mismatch"]
            satisfied = (
                reasons["relation_satisfied"] + reasons["relation_satisfied_with_iou_penalty"]
            )
            print(f"\n-- {label}: {len(selected)} clips, {frames} frames")
            for reason, count in reasons.most_common():
                print(f"     {reason:36s} {count:5d}  {count / frames:6.1%}")
            print(
                f"     no-evidence (missing_subject|missing_object) {drop}/{frames} = {drop / frames:.1%}"
                f"; resolved {resolved}; satisfied {satisfied}"
                f" ({satisfied / (resolved + satisfied):.1%} of resolved)"
                if resolved + satisfied
                else f"     no-evidence {drop}/{frames} = {drop / frames:.1%}"
            )
        originals = [row for row in rows if row["level"] == "original"]
        if originals:
            buckets = {0: [], "1-15": [], 16: []}
            for row in originals:
                detected = (row.get("evidence") or {}).get("detected_frame_count") or 0
                buckets[0 if detected == 0 else (16 if detected >= 16 else "1-15")].append(row["base_id"])
            print(
                f"\n-- originals by frames with both roles detected: "
                f"0 -> {len(buckets[0])}, 1-15 -> {len(buckets['1-15'])}, 16 -> {len(buckets[16])}"
                f" (of {len(originals)} bases)"
            )
            for base in buckets[16]:
                print(f"     fully detected: {base}")
    return 0


def probe_published_bases(args: argparse.Namespace) -> int:
    """Ask whether the *published* spatial bases survive the new eligibility rules.

    `pick_detectable.py` now scans Spatial Relationship, so the documented
    two-step reproduction of the published 205-base selection only still holds if
    all 40 published spatial bases pass.  This walks exactly those 40 source
    clips through the gate's own `detector_verdict`, in both its box-presence-only
    and full directional modes, and prints the components behind each verdict.

    Each source clip is decoded and detected exactly once.  The two
    `detector_verdict` calls are then served from that single forward pass by an
    index-ordered detector, because iterating a decoded clip yields a *fresh*
    numpy view per frame and an `id()`-keyed cache would collide across clips.
    """

    import sys

    from . import pick_detectable
    from .grit import GritDetector, coverage, track_target

    sys.path.insert(0, str(ROOT / "metrics" / "spatial-relationship" / "src"))
    import torch  # noqa: PLC0415, F401

    bases = [
        row for row in read_scores(args.probe_bases)
        if row["dimension"] == "spatial_relationship"
    ]
    real_decode = pick_detectable.decode_video
    grit = GritDetector(args.probe_weight, device=f"cuda:{args.probe_gpu}")

    class IndexedDetector:
        """Replays one clip's detections in frame order for a single pass."""

        def __init__(self, detections: list[Any]) -> None:
            self.detections = detections
            self.index = 0

        def detect(self, frame: Any) -> Any:
            result = self.detections[self.index]
            self.index += 1
            return result

    records = []
    presence_pass = 0
    gate_pass = 0
    for base in bases:
        targets = pick_detectable.detectable_targets("spatial_relationship", base, args.probe_dataset_root)
        query = pick_detectable.spatial_query(base, args.probe_dataset_root)
        video = args.probe_dataset_root / "videos" / base["relative_video_path"]
        frames, meta = real_decode(video)
        detections = [grit.detect(frame) for frame in frames]
        # Serve both verdict calls from this one forward pass.
        pick_detectable.decode_video = lambda _path, _f=frames, _m=meta: (_f, _m)
        coverage_by_target = {}
        for target in targets:
            boxes = track_target(detections, target, pick_detectable.DETECTION_MIN_SCORE)
            coverage_by_target[target] = coverage(boxes)
        rate, scored = pick_detectable.relation_frame_rate(
            detections, query[0], query[1], query[2], pick_detectable.DETECTION_MIN_SCORE
        )
        presence_ok, presence_why, _ = pick_detectable.detector_verdict(
            IndexedDetector(detections), base, args.probe_dataset_root, targets, None,
            args.relation_min_frames, args.min_co_detected_frames,
        )
        gate_ok, gate_why, _ = pick_detectable.detector_verdict(
            IndexedDetector(detections), base, args.probe_dataset_root, targets, query,
            args.relation_min_frames, args.min_co_detected_frames,
        )
        presence_pass += int(presence_ok)
        gate_pass += int(gate_ok)
        records.append({
            "base_id": base["base_id"],
            "split": base["split"],
            "query": f"{query[0]} {query[2]} {query[1]}",
            "frames": len(frames),
            "co_detected_frames": scored,
            "co_detected_fraction": round(scored / len(frames), 4) if len(frames) else 0.0,
            "relation_frame_rate": round(rate, 4),
            "coverage": {name: round(value, 4) for name, value in coverage_by_target.items()},
            "box_presence_pass": bool(presence_ok),
            "box_presence_reason": presence_why,
            "gate_pass": bool(gate_ok),
            "gate_reason": gate_why,
        })
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    payload = {
        "bases": len(records),
        "box_presence_pass": presence_pass,
        "gate_pass": gate_pass,
        "relation_min_frames": args.relation_min_frames,
        "min_co_detected_frames": args.min_co_detected_frames,
        "per_base": records,
    }
    print(
        f"\npublished spatial bases: {len(records)}; "
        f"box-presence-only pass {presence_pass}/{len(records)}; "
        f"full directional gate pass {gate_pass}/{len(records)}"
    )
    if args.probe_output:
        args.probe_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.probe_output}")
    return 0




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench/scores"),
        help="frozen score tree (read-only)",
    )
    parser.add_argument("--dimension", default="spatial_relationship")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    parser.add_argument(
        "--probe-bases", type=Path, default=None,
        help="published bases jsonl; enables the GPU probe instead of the CPU checks",
    )
    parser.add_argument("--probe-dataset-root", type=Path,
                        default=Path("/root/wenbiao_zhao/datasets/vbench-1.0-human-preference"))
    parser.add_argument("--probe-weight", type=Path,
                        default=Path("/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth"))
    parser.add_argument("--probe-gpu", type=int, default=1)
    parser.add_argument("--probe-output", type=Path, default=None)
    parser.add_argument(
        "--evidence-jsonl", type=Path, action="append", default=None,
        help="re-scored repair rows with frame evidence; summarise instead of the CPU checks",
    )
    parser.add_argument(
        "--frozen", type=Path,
        default=Path("/root/wenbiao_zhao/datasets/counterfactual-vbench/scores"),
        help="frozen score tree used as the reproduction control for --evidence-jsonl",
    )
    parser.add_argument("--relation-min-frames", type=float, default=0.75)
    parser.add_argument("--min-co-detected-frames", type=float, default=0.25)
    args = parser.parse_args()

    if args.evidence_jsonl:
        return evidence_report(args.evidence_jsonl, args.frozen)
    if args.probe_bases is not None:
        return probe_published_bases(args)

    rows_by_backend = {
        backend: read_scores(args.scores / f"{args.dimension}__{backend}.jsonl")
        for backend in BACKENDS
    }
    scores_by_backend = {backend: score_map(rows) for backend, rows in rows_by_backend.items()}

    results: dict[str, Any] = {"scores_dir": str(args.scores), "dimension": args.dimension}
    mismatches: list[str] = []

    # 1. per-level profile
    profiles = {backend: profile(rows_by_backend[backend], scores_by_backend[backend]) for backend in BACKENDS}
    results["profile"] = profiles
    print("== per-level profile (frozen tree) ==")
    for key, entry in sorted(profiles["repair"].items()):
        print(
            f"repair  {key:24s} n={entry['n']:3d} mean={entry['mean']:.4f} "
            f"distinct={entry['distinct']:3d} zeros={entry['exact_zero']:3d}"
        )
    for key, entry in sorted(profiles["official"].items()):
        print(
            f"official{key:24s} n={entry['n']:3d} mean={entry['mean']:.4f} "
            f"distinct={entry['distinct']:3d} zeros={entry['exact_zero']:3d}"
        )
    for (backend, split, level), (n, mean, distinct) in REPORT_PROFILE.items():
        entry = profiles[backend].get(f"{split}/{level}")
        if entry is None:
            mismatches.append(f"missing profile row {backend}/{split}/{level}")
            continue
        if entry["n"] != n or abs(entry["mean"] - mean) > 5e-5 or entry["distinct"] != distinct:
            mismatches.append(
                f"profile {backend}/{split}/{level}: report ({n}, {mean}, {distinct}) "
                f"!= recomputed ({entry['n']}, {round(entry['mean'], 4)}, {entry['distinct']})"
            )

    # 2. how much of the family is measurable at all
    all_clips = {backend: sum(1 for row in rows_by_backend[backend]) for backend in BACKENDS}
    zero_clips = {
        backend: sum(1 for row in rows_by_backend[backend] if row.get("score") == 0.0)
        for backend in BACKENDS
    }
    bases_with_signal = {
        backend: len({
            row["base_id"] for row in rows_by_backend[backend]
            if isinstance(row.get("score"), (int, float)) and row["score"] > 0
        })
        for backend in BACKENDS
    }
    original_signal = {
        backend: len({
            row["base_id"] for row in rows_by_backend[backend]
            if row["level"] == "original" and isinstance(row.get("score"), (int, float)) and row["score"] > 0
        })
        for backend in BACKENDS
    }
    results["coverage_of_signal"] = {
        "clips": all_clips,
        "exact_zero_clips": zero_clips,
        "bases_with_any_nonzero_level": bases_with_signal,
        "bases_with_nonzero_original": original_signal,
    }
    print("\n== measurable content ==")
    for backend in BACKENDS:
        print(
            f"{backend:9s} clips={all_clips[backend]:3d} exact-zero={zero_clips[backend]:3d} "
            f"bases with any non-zero level={bases_with_signal[backend]:3d} "
            f"bases with non-zero original={original_signal[backend]:3d}"
        )

    # 3. pair census, order statistics, margins, paired CI
    censuses: dict[str, Any] = {}
    for backend in BACKENDS:
        dev_rows = [row for row in rows_by_backend[backend] if row["split"] == "dev"]
        margin = calibrate_margin(
            [pair for group in _base_pairs(dev_rows, scores_by_backend[backend]).values() for pair in group]
        )
        censuses[backend] = pair_census(rows_by_backend[backend], scores_by_backend[backend], margin)
        censuses[backend]["dev_margin"] = margin
    results["pair_census"] = censuses
    print("\n== rank-gap-1 pair census (test, zero margin) ==")
    for backend in BACKENDS:
        entry = censuses[backend]
        print(
            f"{backend:9s} pairs={entry['n_pairs']:3d} correct={entry['correct']:3d} "
            f"tied={entry['tied']:3d} inverted={entry['inverted']:3d} "
            f"match={entry['match_rate']:.4f} tie={entry['tie_rate']:.4f} "
            f"dev_margin={entry['dev_margin']:.5f}"
        )
        if abs((entry["match_rate"] or 0) - REPORT_TEST_CPA[backend]) > 5e-5:
            mismatches.append(
                f"test CPA {backend}: report {REPORT_TEST_CPA[backend]} != recomputed {entry['match_rate']}"
            )
    print("\n== order statistics (test) ==")
    for backend in BACKENDS:
        test_rows = [row for row in rows_by_backend[backend] if row["split"] == "test"]
        stats = order_statistics(test_rows, scores_by_backend[backend])
        results.setdefault("order_statistics", {})[backend] = stats
        print(
            f"{backend:9s} bases={stats['n_bases']:3d} mean_rho={stats['mean_spearman']:+.4f} "
            f"median_rho={stats['median_spearman']:+.4f} strict={stats['strict_order_bases']}/"
            f"{stats['n_bases']} ({stats['strict_order_rate']:.4f})"
        )
    print("\n== discordance sign test (test, rank gap 1) ==")
    for backend in BACKENDS:
        results.setdefault("sign_test", {})[backend] = sign_test(censuses[backend])
        entry = results["sign_test"][backend]
        print(
            f"{backend:9s} discordant={entry['discordant']:3d} "
            f"prefer_original={entry.get('prefer_original')} prefer_flip={entry.get('prefer_flip')} "
            f"p={entry['p_value'] if entry['p_value'] is None else round(entry['p_value'], 4)}"
        )

    official_pairs = _base_pairs(
        [row for row in rows_by_backend["official"] if row["split"] == "test"],
        scores_by_backend["official"],
    )
    repair_pairs = _base_pairs(
        [row for row in rows_by_backend["repair"] if row["split"] == "test"],
        scores_by_backend["repair"],
    )
    paired = paired_bootstrap_ci(
        official_pairs, repair_pairs, censuses["official"]["dev_margin"],
        censuses["repair"]["dev_margin"], args.iterations, args.seed,
    )
    results["paired_delta"] = paired
    print("\n== paired delta ==")
    print(
        f"delta={paired['delta']:+.4f} 95% CI [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] "
        f"over {paired['n_bases']} bases"
    )
    for key, expected in REPORT_PAIRED.items():
        if paired.get(key) is None or abs(paired[key] - expected) > 5e-4:
            mismatches.append(f"paired {key}: report {expected} != recomputed {paired.get(key)}")

    results["reproduction_mismatches"] = mismatches
    print("\n== reproduction against the report ==")
    print("PASS: every quoted number reproduces" if not mismatches else "FAIL:")
    for line in mismatches:
        print(f"  - {line}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
