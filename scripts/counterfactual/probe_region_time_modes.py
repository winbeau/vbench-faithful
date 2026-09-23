"""Crossed region/time prediction on complete image-plane diagnostic caches.

Rank zero is the constant-velocity baseline estimated WITHOUT the tested time.
Temporal bases use other regions; all three time folds are evaluated and kept.
This validates predictability of model flow, not physical nuisance causation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.image_plane_modes import leave_region_time_prediction
from .score_static_jitter import source_hashes
from .static_jitter import digest


def compare_predictions(values, reliable, predictions, times):
    dt = np.diff(times)
    common = np.asarray(reliable, bool).copy()
    for field in predictions.values():
        common &= np.isfinite(field).all(axis=-1)
    errors = {rank: float(np.sum(np.where(common, np.sum((np.nan_to_num(field) - values) ** 2, axis=-1), 0)
                                       * dt[:, None])) if common.any() else None for rank, field in predictions.items()}
    result = []
    for rank, field in predictions.items():
        tested = reliable & np.isfinite(field).all(axis=-1)
        result.append({"rank": rank, "testable_point_pairs": int(tested.sum()), "total_point_pairs": int(common.size),
                       "common_point_pairs": int(common.sum()), "squared_error_on_common": errors[rank],
                       "improvement_over_heldout_constant": (1 - errors[rank] / errors[0]) if errors[0] is not None and errors[0] > 1e-20 else None})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    source, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise FileExistsError("fresh crossed-prediction run required")
    parent = json.loads((source / "provenance.json").read_text())
    state = json.loads((source / "runtime.json").read_text())
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    if (state["status"] != "finished" or state["completed"] != state["expected"] or state["failed"]
            or digest(source / "diagnostics.jsonl") != parent["diagnostics_sha256"]
            or len(rows) != len(parent["input_sha256"]) or {r["candidate_id"] for r in rows} != set(parent["input_sha256"])):
        raise ValueError("complete same-video model cache required")
    for row in rows:
        key = row["candidate_id"]
        if row["score"] is not None or digest(source / "evidence" / f"{key}.npz") != parent["evidence_sha256"][key]:
            raise ValueError("changed evidence/null-score contract")
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    identity = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                "source_provenance_sha256": digest(source / "provenance.json"), "source_diagnostics_sha256": parent["diagnostics_sha256"],
                "source_evidence_sha256": parent["evidence_sha256"], "input_sha256": parent["input_sha256"],
                "ranks": [0, 1, 2, 3], "time_folds": 3, "fold_assignment": "transition_index modulo 3, each transition held out once",
                "point_valid_fraction": .6, "basis_source": "other reference-frame SAM ownership regions only",
                "scope": "model-flow prediction; not a physical-motion or nuisance classifier; missing stays missing",
                "script_sha256": digest(Path(__file__)), "code_files": source_hashes(Path(__file__).resolve().parents[2]),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu"}, "evidence_sha256": {}}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0, "failed": 0,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()

    def update(**fields):
        runtime.update(fields, elapsed_seconds=time.monotonic() - started)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))

    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    update()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key = row["candidate_id"]
            record = {k: row[k] for k in ("candidate_id", "base_id", "family", "seed", "original_reliability_status")}
            record.update(status="diagnostic_only", score=None)
            try:
                with np.load(source / "evidence" / f"{key}.npz", allow_pickle=False) as z:
                    values, times, owners, reliable = z["velocity"], z["timestamps"], z["owners"], z["reliable_pair"]
                    reference_valid = z["point_valid"]
                if not np.array_equal(np.average(reliable, axis=0, weights=np.diff(times)) >= .6, reference_valid):
                    raise ValueError("point evidence definition changed")
                predictions, details = {}, {}
                for rank in identity["ranks"]:
                    predictions[rank], details[str(rank)] = leave_region_time_prediction(values, times, owners, reliable, rank=rank)
                record.update(comparisons=compare_predictions(values, reliable, predictions, times), regions=details)
                path = output / "evidence" / f"{key}.npz"
                np.savez_compressed(path, **{f"prediction_rank_{rank}": field for rank, field in predictions.items()})
                identity["evidence_sha256"][key] = digest(path)
                print(json.dumps({"candidate_id": key, "comparisons": record["comparisons"]}), flush=True)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            update(completed=runtime["completed"] + 1, candidate_id=key)
    update(status="finished", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    identity["diagnostics_sha256"] = digest(output / "diagnostics.jsonl")
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    return int(runtime["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
