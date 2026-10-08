#!/usr/bin/env python3
"""Verify and summarize the same-32 H100 experiment from its retained artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from paper_common import ROOT, OFFICIAL, PAPER, TASKS, digest, write_json


def read(path):
    return json.loads(Path(path).read_text())


def normalized_rows(run, dimension, backend):
    payload = read(run / dimension / (backend + ".json"))
    replacements = [(r["artifacts"], "<STAGE:" + dimension + "/" + r["stage"] + ">")
                    for r in payload["execution"]["cache"]]
    replacements.append((str(run), "<OUTPUT>"))
    replacements.sort(key=lambda pair: -len(pair[0]))

    def normalize(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"} and Path(value["path"]).suffix == ".json":
                assert digest(value["path"]) == value["sha256"], value["path"]
                return {"verified_content": normalize(read(value["path"]))}
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
        return value

    return normalize(payload["rows"])


def summarize(root):
    names = ["origin-baseline-v2", "repair-baseline-v2", "repair-optimized"]
    inputs = read(root / "inputs-v2.json")
    assert len(inputs) == 512 and len({r["video_sha256"] for r in inputs}) == 32
    for dimension in OFFICIAL:
        rows = [r for r in inputs if r["dimensions"] == [dimension]]
        assert len(rows) == 32 and len({r["video_sha256"] for r in rows}) == 32
    report = {"schema": "same32-performance/1", "input_sha256": digest(root / "inputs-v2.json"),
              "input_records": 512, "unique_media": 32, "media_seconds": 64,
              "host": "h100-server", "runs": {}, "repair_parity": [], "fallback_parity": [],
              "native_parity": [], "shared_inference": {},
              "scope": "agent-annotated timing queries, including negative probes; not official preference evaluation"}
    for name in names:
        run = root / name
        plan, summary, clock = read(run / "plan.json"), read(run / "summary.json"), read(root / (name + ".timing.json"))
        assert plan["input_sha256"] == report["input_sha256"] and not plan["git_dirty"]
        assert len(summary["dimensions"]) == 16
        assert all(s["input_count"] == 32 and not s["cached"] for s in summary["dimensions"].values())
        report["runs"][name] = {"output": str(run), "wall_seconds": clock["wall_seconds"],
            "returncode": clock["returncode"], "command": clock["command"],
            "code_sha": plan["code_sha"], "source_sha256": plan["source_sha256"],
            "upstream_sha": plan["upstream_sha"], "gpu_info": plan["gpu_info"],
            "lock_sha256": plan["lock_sha256"], "input_sha256": plan["input_sha256"],
            "complete": summary["complete"], "dimensions": summary["dimensions"],
            "preflight_seconds": summary["timing_seconds"], "previous_run_cache_hits": 0}
    old, new = root / names[1], root / names[2]
    for dimension in OFFICIAL:
        backend = "repair" if dimension in PAPER else "origin"
        a, b = [read(p / dimension / (backend + ".json")) for p in (old, new)]
        assert [(r["id"], r["video_sha256"], r["status"]) for r in a["rows"]] == [
            (r["id"], r["video_sha256"], r["status"]) for r in b["rows"]], dimension
        errors = [abs(x["score"] - y["score"]) for x, y in zip(a["rows"], b["rows"])
                  if x["score"] is not None]
        entry = {"dimension": dimension, "records": len(a["rows"]),
                 "succeeded": sum(r["status"] == "succeeded" for r in a["rows"]),
                 "max_score_error": max(errors, default=0), "statuses_equal": True}
        if backend == "repair":
            assert all(x["score"] == y["score"] for x, y in zip(a["rows"], b["rows"])), dimension
            entry["diagnostics_equal"] = normalized_rows(old, dimension, backend) == normalized_rows(new, dimension, backend)
            assert entry["diagnostics_equal"], (dimension, "diagnostics")
            if dimension in TASKS:
                assert a["predictions"] == b["predictions"], (dimension, "raw predictions")
                assert read(old / dimension / "evidence/result.json")["rows"] == read(new / dimension / "evidence/result.json")["rows"], (dimension, "visual evidence")
                entry.update(predictions_equal=True, visual_evidence_equal=True)
            elif dimension in {"object_class", "color"}:
                assert read(old / dimension / "compile/result.json") == read(new / dimension / "compile/result.json"), (dimension, "compilation")
                entry["compilation_equal"] = True
            report["repair_parity"].append(entry)
        else:
            entry["aggregate_error"] = abs(a["official_aggregate"] - b["official_aggregate"])
            assert max(entry["max_score_error"], entry["aggregate_error"]) <= 1e-6, dimension
            entry["tolerance"] = 1e-6
            report["fallback_parity"].append(entry)
    for dimension in ("spatial_relationship", "multiple_objects"):
        report["shared_inference"][dimension] = read(new / dimension / "evidence/result.json")["run_shared_inference"]
    for dimension in ("object_class", "color"):
        log = (new / dimension / "repair/worker.log").read_text()
        records = [json.loads(line.split("GRiT run inference ", 1)[1]) for line in log.splitlines()
                   if line.startswith("GRiT run inference ")]
        assert records, dimension
        report["shared_inference"][dimension] = records[-1]
    native = read(root / "origin-native/timing.json")
    clock = read(root / "origin-native.timing.json")
    assert clock["returncode"] == 0 and len(native["dimensions"]) == 16
    assert native["input_sha256"] == report["input_sha256"]
    report["runs"]["origin-native"] = {**native, "wall_seconds": clock["wall_seconds"],
        "output": str(root / "origin-native"), "command": clock["command"],
        "driver_sha256": digest(root / "benchmark_vbench_native.py")}
    for dimension in OFFICIAL:
        official = read(root / names[0] / dimension / "origin.json")
        native_rows = read(root / "origin-native" / (dimension + ".json"))
        returned = {r["video_path"]: r for r in native_rows}
        valid = [r for r in official["rows"] if r["status"] == "succeeded"]
        assert returned.keys() == {r["video"] for r in valid}, dimension
        error = max(abs(returned[r["video"]]["video_results"] - r["score"]) for r in valid)
        aggregate_error = abs(native["dimensions"][dimension]["official_aggregate"] - official["official_aggregate"])
        assert max(error, aggregate_error) <= 1e-6, (dimension, error, aggregate_error)
        report["native_parity"].append({"dimension": dimension, "returned": len(returned),
            "max_score_error": error, "aggregate_error": aggregate_error, "tolerance": 1e-6})
    times = {name: run["wall_seconds"] for name, run in report["runs"].items()}
    report["comparisons"] = {name: {"reference_seconds": times[name], "optimized_seconds": times[names[2]],
        "speedup": times[name] / times[names[2]],
        "reduction_percent": 100 * (1 - times[names[2]] / times[name])}
        for name in (names[0], names[1], "origin-native")}
    report["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader"], text=True).splitlines()
    report["media"] = read(ROOT / "configs/benchmarks/same32-20261008.json")["videos"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("Use a new report outside frozen research directories")
    report = summarize(args.experiment.resolve())
    write_json(output, report)
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
