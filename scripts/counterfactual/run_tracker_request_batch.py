"""Execute every bound query phase once, with isolated logs and no retry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .static_jitter import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("requests", "tracker-root", "tracker-weight", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=4096)
    args = p.parse_args()
    if not 0 <= args.shard < args.shards:
        p.error("valid disjoint phase shard required")
    requests = sorted(Path(args.requests).glob("frame-*.json"))
    if [f.name for f in requests] != [f"frame-{i:02}.json" for i in range(16)]:
        raise ValueError("exactly all 16 native query phases required")
    selected = requests[args.shard::args.shards]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "score": None, "pid": os.getpid(),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "shard": args.shard, "shards": args.shards, "phases": [f.stem for f in selected],
              "script_sha256": digest(Path(__file__)), "jobs": []}
    def save():
        (output / "batch.json").write_text(json.dumps(record, indent=2))
    start = time.monotonic()
    save()
    for path in selected:
        command = [sys.executable, "-m", "scripts.counterfactual.probe_feature_tracks", "infer",
                   "--request", str(path), "--tracker-root", args.tracker_root,
                   "--tracker-weight", args.tracker_weight, "--output", str(output / path.stem),
                   "--max-queries", str(args.max_queries)]
        for root in args.video_root:
            command += ["--video-root", root]
        job = {"phase": path.stem, "request_sha256": digest(path), "command": command,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        record["jobs"].append(job)
        save()
        now = time.monotonic()
        with (output / f"{path.stem}.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        job.update(returncode=process.returncode, elapsed_seconds=time.monotonic()-now,
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save()
        print(json.dumps({"phase": path.stem, "returncode": process.returncode, "elapsed_seconds": job["elapsed_seconds"]}), flush=True)
    record.update(status="finished" if all(j["returncode"] == 0 for j in record["jobs"]) else "failed",
                  elapsed_seconds=time.monotonic()-start,
                  finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save()
    return int(record["status"] != "finished")


if __name__ == "__main__":
    raise SystemExit(main())
