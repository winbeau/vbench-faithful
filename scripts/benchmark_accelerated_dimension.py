#!/usr/bin/env python3
"""Validate one accelerator against retained, complete official video scores.

Runs fresh inference in the dimension's isolated environment. Worker and full
wall times are reported separately; neither replaces a final all-16 comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from eval import Workers, gpu_tokens, probe_gpu, source_identity, worker_env, writable_path
from paper_common import ROOT, digest, load_assets, load_inputs, verify_assets, write_json
from vbench_audit_core.environments import prepare_environment
from vbench_audit_core.eval_config import load_config
from vbench_audit_core.score_agreement import compare_scores


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    assets = load_assets(config.assets)
    rows = [r for r in load_inputs(config.input, config.video_root) if args.dimension in r["dimensions"]]
    if not rows:
        raise ValueError("No input for selected dimension")
    if len(config.gpus) != 1:
        raise ValueError("Single-dimension acceptance requires exactly one selected GPU")
    import os
    gpu = gpu_tokens(config.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))[0]
    output = writable_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checked = verify_assets(assets, [args.dimension], "origin", workers=4)
    from vbench_audit_core.upstream import verify_upstream
    state = verify_upstream(assets["vbench"])
    source = source_identity(assets)
    runtime = probe_gpu(assets["visual_python"], gpu)
    python = prepare_environment(args.dimension, "visual", Path(assets["visual_python"]), config.env_dir)
    write_json(output / "inputs.json", rows)
    write_json(output / "assets.json", assets)
    env = worker_env(args.dimension, gpu, assets)
    env["VBENCH_EVAL_RUN_INFERENCE_CACHE"] = str(output / "model-inference")
    env["VBENCH_EVAL_RUN_INFERENCE_CONTEXT"] = json.dumps({"source": source, "runtime": runtime,
        "gpu": gpu, "python": assets["visual_python"], "assets": checked,
        "precision": "source-bound-model-dtypes-tf32-disabled", "cpu_threads": 3})
    command = [python, ROOT / "scripts/paper_visual.py", "--input", output / "inputs.json",
               "--assets", output / "assets.json", "--output", output / "result.json",
               "--dimension", args.dimension, "--mode", "accelerated"]
    before_worker = time.monotonic()
    Workers().run(command, env, output / "worker.log")
    worker_seconds = time.monotonic() - before_worker
    candidate = json.loads((output / "result.json").read_text())
    reference = json.loads(args.reference.read_text())
    comparison = compare_scores(reference["rows"], candidate["rows"])
    aggregate_error = abs(candidate["aggregate"] - reference["official_aggregate"])
    aggregate_tolerance = max(1e-6, .01 * abs(reference["official_aggregate"]))
    reference_stages = reference.get("execution", {}).get("cache", [])
    report = {"dimension": args.dimension, "code_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "source_sha256": source, "upstream_sha": state.sha, "gpu": gpu, "gpu_info": runtime,
        "input_sha256": digest(config.input), "selected_inputs_sha256": digest(output / "inputs.json"),
        "reference": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "output": str(output), "worker_seconds": worker_seconds,
        "wall_seconds": time.monotonic() - started, "previous_run_cache_hits": 0,
        "reference_worker_seconds": sum(r["worker_seconds"] for r in reference_stages),
        "comparison": comparison, "aggregate_error": aggregate_error,
        "aggregate_tolerance": aggregate_tolerance,
        "passed": comparison["passed"] and aggregate_error <= aggregate_tolerance}
    write_json(output / "acceptance.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "comparison"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
