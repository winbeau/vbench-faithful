#!/usr/bin/env python3
"""Plan or run supplementary workers with strict dimension serialization.

One dimension is selected at a time. Its four shards run concurrently on the
four requested physical GPUs, each isolated with a one-device
``CUDA_VISIBLE_DEVICES`` mask. The next dimension is not started until all
four shard commands finish. The default is a dry run.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMS = ("dynamic_degree", "subject_consistency", "human_action", "spatial_relationship", "scene", "multiple_objects", "overall_consistency", "motion_smoothness")


def command(template: str, dimension: str, shard: int, physical_gpu: int, output: Path) -> list[str]:
    rendered = template.format(dimension=dimension, shard=shard, gpu=0, physical_gpu=physical_gpu, output=str(output))
    return shlex.split(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=DIMS, action="append", help="Repeatable; omitted means all dimensions in fixed order")
    parser.add_argument("--gpus", default="4,5,6,7", help="Exactly four physical GPU IDs; each worker sees its card as logical cuda:0")
    parser.add_argument("--worker-command", default="echo worker dimension={dimension} shard={shard} gpu={gpu} physical_gpu={physical_gpu} output={output}", help="Command template; placeholders: {dimension}, {shard}, {gpu}, {physical_gpu}, {output}")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/supplementary_20260914")
    parser.add_argument("--execute", action="store_true", help="Run workers; without this flag only emit the schedule")
    args = parser.parse_args()
    try:
        gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"--gpus must be comma-separated integers: {exc}")
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(gpu < 0 for gpu in gpus):
        parser.error("--gpus must contain four distinct non-negative physical GPU IDs")
    dimensions = args.dimension or list(DIMS)
    args.output_root.mkdir(parents=True, exist_ok=True)
    schedule = []
    for dimension in dimensions:
        shard_commands = []
        for shard, physical_gpu in enumerate(gpus):
            output = args.output_root / dimension / f"shard_{shard}.jsonl"
            env = {"CUDA_VISIBLE_DEVICES": str(physical_gpu)}
            shard_commands.append({"dimension": dimension, "shard": shard, "gpu": 0, "physical_gpu": physical_gpu, "output": str(output), "command": command(args.worker_command, dimension, shard, physical_gpu, output), "env": env})
        schedule.append({"dimension": dimension, "shards": shard_commands})
    schedule_path = args.output_root / "orchestrator_schedule.json"
    schedule_path.write_text(json.dumps({"policy": "dimension_serial_shard_parallel", "physical_gpus": gpus, "schedule": schedule}, indent=2) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN", "dimensions": dimensions, "schedule": str(schedule_path), "policy": f"one dimension at a time; four shards in parallel on physical GPUs {gpus}"}, indent=2))
        return 0
    for entry in schedule:
        processes = []
        for item in entry["shards"]:
            env = os.environ.copy()
            env.update(item["env"])
            processes.append((item, subprocess.Popen(item["command"], cwd=ROOT, env=env)))
        statuses = []
        for item, process in processes:
            statuses.append({"shard": item["shard"], "returncode": process.wait()})
        if any(item["returncode"] != 0 for item in statuses):
            print(json.dumps({"status": "BLOCKED", "dimension": entry["dimension"], "shards": statuses}))
            return 2
        print(json.dumps({"status": "DIMENSION_COMPLETE", "dimension": entry["dimension"], "shards": statuses}), flush=True)
    print(json.dumps({"status": "COMPLETE", "dimensions": dimensions, "schedule": str(schedule_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
