"""Video-only SAM boundary checks and region-protected reversal ablation.

Reuses complete, hash-bound image-plane evidence. The boundary variant deletes
no motion and inspects modes 0/1/2; the regional-reversal variant tests constrained
subtraction of the dominant mode. Neither is a physical-cause classifier or a
complete score. Both retain all reviewed videos and all native time phases.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.image_plane_modes import (
    affine_field_split, boundary_motion, fit_temporal_field, temporal_modes, regional_reversal_ablation,
)
from .score_static_jitter import source_hashes
from .static_jitter import digest


def inspect_boundaries(values, times, xy, owners, reliable, shape, *, grid_shape=(32, 32)):
    values, times, xy = np.asarray(values), np.asarray(times), np.asarray(xy)
    owners, reliable = np.asarray(owners), np.asarray(reliable, bool)
    if np.prod(grid_shape) != len(xy):
        raise ValueError("boundaries require the original complete rectangular grid")
    locations = xy.reshape(*grid_shape, 2)
    if (not np.allclose(np.diff(locations[..., 0], axis=0), 0)
            or not np.allclose(np.diff(locations[..., 1], axis=1), 0)
            or not np.allclose(np.diff(locations[..., 0], axis=1), np.diff(locations[..., 0], axis=1)[0, 0])
            or not np.allclose(np.diff(locations[..., 1], axis=0), np.diff(locations[..., 1], axis=0)[0, 0])):
        raise ValueError("equally spaced x-fastest grid required")
    dt = np.diff(times)
    valid = np.average(reliable, axis=0, weights=dt) >= .6
    modes = temporal_modes(values[:, valid], times, xy[valid], shape, owners[valid])
    arrays = {"xy": xy, "owners": owners, "reliable_pair": reliable, "timestamps": times}
    results = []
    for index, mode in enumerate(modes):
        wave = np.array(mode["temporal_velocity_coefficients"])
        wave /= np.sqrt(np.average(wave ** 2, weights=dt))
        field, constant, observed = fit_temporal_field(values, times, wave, reliable)
        affine, residual = affine_field_split(field, xy, shape, observed)
        arrays.update({f"mode_{index}_{name}": value for name, value in (
            ("wave", wave), ("field", field), ("constant", constant), ("observed", observed),
            ("affine", affine), ("nonaffine", residual))})
        energy = float(np.sum(field[observed] ** 2)) if observed.any() else None
        rms = float(np.sqrt(energy / observed.sum())) if observed.any() else None
        boundaries = []
        for stride in (1, 2, 4):
            statistics, _ = boundary_motion(field.reshape(*grid_shape, 2), owners.reshape(grid_shape),
                                           observed.reshape(grid_shape), stride=stride)
            for which in ("cross_region", "same_region"):
                measured = statistics[which]["jump_rms"]
                statistics[which]["jump_over_field_rms"] = measured / rms if measured is not None and rms > 1e-20 else None
            boundaries.append(statistics)
        results.append({"mode": index, "energy_fraction": mode["energy_fraction"],
                        "reversal_fraction": mode["reversal_fraction"], "participation_fraction": mode["participation_fraction"],
                        "region_energy": mode["region_energy"], "observed_points": int(observed.sum()),
                        "field_rms_velocity": rms,
                        "affine_explained_fraction": float(1 - np.sum(residual[observed] ** 2) / energy) if energy and np.isfinite(residual[observed]).all() else None,
                        "boundaries": boundaries})
    return {"status": "diagnostic_only", "score": None, "total_points": len(xy), "modes": results,
            "warning": "reference-frame SAM boundaries and model velocity only; smoothness is not a nuisance label"}, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["boundaries", "regional-reversal"], default="boundaries")
    args = parser.parse_args(argv)
    source, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise FileExistsError("fresh boundary run required")
    parent = json.loads((source / "provenance.json").read_text())
    state = json.loads((source / "runtime.json").read_text())
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    if (state["status"] != "finished" or state["completed"] != state["expected"] or state["failed"]
            or digest(source / "diagnostics.jsonl") != parent["diagnostics_sha256"]
            or len(rows) != len(parent["input_sha256"]) or {r["candidate_id"] for r in rows} != set(parent["input_sha256"])):
        raise ValueError("complete same-video image-plane cache required")
    for row in rows:
        if row["status"] != "diagnostic_only" or row["score"] is not None or digest(source / "evidence" / f'{row["candidate_id"]}.npz') != parent["evidence_sha256"][row["candidate_id"]]:
            raise ValueError("source evidence identity or diagnostic contract changed")
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    identity = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED", "variant": args.variant,
                "source_provenance_sha256": digest(source / "provenance.json"), "source_diagnostics_sha256": parent["diagnostics_sha256"],
                "source_evidence_sha256": parent["evidence_sha256"], "input_sha256": parent["input_sha256"],
                "config": {"grid_shape": [32, 32], "strides": [1, 2, 4], "modes": 3, "point_valid_fraction": .6,
                           "fit": "observed pairs only; intercept plus unit-weighted-RMS temporal mode",
                           "boundary": "reference middle-frame smallest-containing SAM region; includes uncovered -1"},
                "script_sha256": digest(Path(__file__)), "code_files": source_hashes(Path(__file__).resolve().parents[2]),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu"},
                "evidence_sha256": {}}
    if args.variant == "regional-reversal":
        identity["config"] = {"posthoc_development_ablation_not_frozen_protocol": True,
                              "grid_shape": [32, 32], "leading_mode_energy_min": .5, "reversal_fraction_gt": .5,
                              "participation_min": .1, "pair_evidence_min": .6, "point_evidence_min": .6,
                              "crossed_rank1_prediction_gain_gt": .5, "singular_gap_min": .05,
                              "projection_max_iterations": 400, "projection_relative_tolerance": 1e-9,
                              "constraints": "preserve per-time region-affine projections and per-point dt-weighted mean velocity",
                              "missing": "no removal on unknown pairs; conditional speed is not a full score"}
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
                    inspect = inspect_boundaries if args.variant == "boundaries" else regional_reversal_ablation
                    result, arrays = inspect(z["velocity"], z["timestamps"], z["xy"], z["owners"],
                                             z["reliable_pair"], row["sampling"]["sampled_shape"][:2])
                record.update(result)
                path = output / "evidence" / f"{key}.npz"
                np.savez_compressed(path, **arrays)
                identity["evidence_sha256"][key] = digest(path)
                if args.variant == "boundaries":
                    printed = {"modes": [{"mode": m["mode"], "energy": m["energy_fraction"],
                               "cross": m["boundaries"][0]["cross_region"], "inside": m["boundaries"][0]["same_region"]} for m in result["modes"]]}
                else:
                    printed = {k: result[k] for k in ("subtracted", "retention_reasons", "constraints", "conditional_raw_speed", "conditional_guarded_speed")}
                print(json.dumps({"candidate_id": key, **printed}), flush=True)
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
