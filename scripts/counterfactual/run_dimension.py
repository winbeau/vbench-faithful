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
from typing import Any

from .common import ROOT, read_jsonl, write_json
from .cpa import BUDGET, _base_pairs, calibrate_margin, evaluate_method

DEFAULT_GPUS = "1,2,3,4,5,6"
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


def render_report(
    dimension: str,
    family: str,
    rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    cpa: dict[str, Any],
    code_sha: str,
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
        f"- code SHA: `{code_sha}`",
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
    lines += ["", "## Official vs Repair (test, tie-aware)", "", "| metric | official | repair | repair - official |", "|---|---:|---:|---:|"]
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
            f"{rep.get('cpa', float('nan')):.4f} | {delta} |"
        )
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
    coverage = []
    for backend in BACKENDS:
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

    scores: dict[tuple[str, str], float | None] = {}
    for backend in BACKENDS:
        for row in read_jsonl(args.scores / f"{args.dimension}__{backend}.jsonl"):
            if row.get("status") == "succeeded" and row.get("score") is not None:
                scores[(row["derived_id"], backend)] = float(row["score"])

    cpa: dict[str, Any] = {"coverage": []}
    for backend in BACKENDS:
        backend_scores = {row["derived_id"]: scores.get((row["derived_id"], backend)) for row in rows}
        dev_rows = [row for row in rows if row["split"] == "dev"]
        test_rows = [row for row in rows if row["split"] == "test"]
        margin = calibrate_margin(
            [pair for group in _base_pairs(dev_rows, backend_scores).values() for pair in group]
        )
        cpa[backend] = {
            "dev_margin": margin,
            "dev_zero_margin": evaluate_method(dev_rows, backend_scores, 0.0, args.iterations, args.seed),
            "test_zero_margin": evaluate_method(test_rows, backend_scores, 0.0, args.iterations, args.seed),
            "test_tie_aware": evaluate_method(test_rows, backend_scores, margin, args.iterations, args.seed),
        }
        for split, split_rows in (("dev", dev_rows), ("test", test_rows)):
            scored = sum(1 for row in split_rows if backend_scores.get(row["derived_id"]) is not None)
            cpa["coverage"].append({"backend": backend, "split": split, "scored": scored, "total": len(split_rows)})

    code_sha = next((row.get("code_sha") for row in rows if row.get("code_sha")), "unknown")
    args.scores.mkdir(parents=True, exist_ok=True)
    write_json(args.scores / f"{args.dimension}__cpa.json", cpa)
    report = render_report(args.dimension, family, rows, coverage, cpa, code_sha)
    (args.reports / f"{args.dimension}.md").write_text(report, encoding="utf-8")
    print(json.dumps({"dimension": args.dimension, "coverage": coverage, "report": str(args.reports / f"{args.dimension}.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
