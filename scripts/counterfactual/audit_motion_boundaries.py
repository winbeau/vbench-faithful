"""Verify boundary diagnostics and actual constrained-removal arrays, not scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamic_degree.image_plane_modes import boundary_motion, spatial_design
from .static_jitter import digest


def verify_removal(arrays, shape, record):
    raw, corrected, removal = (arrays[k] for k in ("raw_velocity", "corrected_velocity", "removal"))
    reliable, times, owners, xy = (arrays[k] for k in ("reliable_pair", "timestamps", "owners", "xy"))
    dt = np.diff(times)
    if (not np.array_equal(raw - removal, corrected) or not np.isfinite(corrected).all()
            or np.any(removal[~reliable]) or record["score"] is not None
            or record["subtracted"] != bool(np.any(removal))):
        raise ValueError("invalid deletion, missing-to-zero or score promotion")
    denom = float(np.sum(reliable * dt[:, None]) * min(shape))
    for key, values in (("conditional_raw_speed", raw), ("conditional_guarded_speed", corrected)):
        measured = float(np.sum(np.linalg.norm(values, axis=-1) * reliable * dt[:, None]) / denom) if denom else None
        if measured != record[key]:
            raise ValueError("reported conditional speed differs from arrays")
    if not record["subtracted"]:
        return {"subtracted": False, "constraints_checked": False}
    constraints = record["constraints"]
    if not constraints["converged"] or record["retention_reasons"]:
        raise ValueError("unconverged or vetoed component was deleted")
    point_valid = np.average(reliable, weights=dt, axis=0) >= .6
    observed = arrays.get("projection_observed", reliable & point_valid[None])
    if np.any(observed & ~(reliable & point_valid[None])) or constraints["observed_pairs"] != int(observed.sum()):
        raise ValueError("projection evidence mask changed or exceeds observed support")
    design = spatial_design(xy, shape, 0)
    # Independent least-squares check, not the implementation's QR projector.
    affine = np.zeros_like(removal)
    for t in range(len(dt)):
        for region in np.unique(owners):
            points = observed[t] & (owners == region)
            if not points.any():
                continue
            if points.sum() <= 3 or np.linalg.matrix_rank(design[points]) < 3:
                if np.any(removal[t, points]):
                    raise ValueError("small/degenerate region protection violated")
            else:
                affine[t, points] = design[points] @ np.linalg.lstsq(design[points], removal[t, points], rcond=None)[0]
    weights = observed * dt[:, None]
    totals = weights.sum(axis=0)
    mean = np.divide((weights[..., None] * removal).sum(axis=0), totals[:, None],
                     out=np.zeros_like(removal[0]), where=totals[:, None] > 0)
    affine_error = float(np.sqrt(np.sum(affine ** 2 * dt[:, None, None])))
    mean_error = float(np.sqrt(np.sum((mean[None] * observed[..., None]) ** 2 * dt[:, None, None])))
    if max(affine_error, mean_error) > constraints["constraint_limit"] * (1 + 1e-6):
        raise ValueError("affine or mean-velocity safeguard violated in actual arrays")
    return {"subtracted": True, "constraints_checked": True, "affine_error": affine_error,
            "mean_velocity_error": mean_error, "limit": constraints["constraint_limit"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "parent-run", "execution-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    source, parent, execution, output = map(Path, (args.source_run, args.parent_run, args.execution_root, args.output))
    if output.exists():
        raise FileExistsError("fresh audit required")
    identity = json.loads((source / "provenance.json").read_text())
    runtime = json.loads((source / "runtime.json").read_text())
    parent_id = json.loads((parent / "provenance.json").read_text())
    parent_rows = {r["candidate_id"]: r for r in map(json.loads, (parent / "diagnostics.jsonl").read_text().splitlines())}
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    if (runtime["status"] != "finished" or runtime["failed"] or runtime["completed"] != runtime["expected"]
            or len(rows) != runtime["expected"] or {r["candidate_id"] for r in rows} != set(identity["input_sha256"])
            or identity["input_sha256"] != parent_id["input_sha256"]
            or digest(parent / "provenance.json") != identity["source_provenance_sha256"]
            or digest(parent / "diagnostics.jsonl") != identity["source_diagnostics_sha256"]
            or digest(source / "diagnostics.jsonl") != identity["diagnostics_sha256"]
            or {p.stem for p in (source / "evidence").glob("*.npz")} != set(identity["input_sha256"])):
        raise ValueError("incomplete or changed source cohort")
    for name, sha in {**identity["code_files"], "scripts/counterfactual/probe_motion_boundaries.py": identity["script_sha256"]}.items():
        if digest(execution / name) != sha:
            raise ValueError("execution snapshot changed")
    originals, controls, videos = {}, [], []
    for row in rows:
        key = row["candidate_id"]
        path = source / "evidence" / f"{key}.npz"
        parent_path = parent / "evidence" / f"{key}.npz"
        if (row["status"] != "diagnostic_only" or row["score"] is not None
                or digest(path) != identity["evidence_sha256"][key]
                or digest(parent_path) != identity["source_evidence_sha256"][key]):
            raise ValueError("altered arrays or diagnostic contract")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(parent_path, allow_pickle=False) as z:
            for shared in ("xy", "owners", "reliable_pair", "timestamps"):
                if not np.array_equal(arrays[shared], z[shared]):
                    raise ValueError("input geometry, phase or evidence changed")
            if identity.get("variant") == "regional-reversal" and not np.array_equal(arrays["raw_velocity"], z["velocity"]):
                raise ValueError("original model velocity changed")
        if identity.get("variant") == "regional-reversal":
            check = verify_removal(arrays, parent_rows[key]["sampling"]["sampled_shape"][:2], row)
            videos.append({k: row[k] for k in ("candidate_id", "base_id", "family", "seed", "original_reliability_status",
                                              "conditional_raw_speed", "conditional_guarded_speed", "retention_reasons")} | check)
        else:
            for mode in row["modes"]:
                index = mode["mode"]
                field = arrays[f"mode_{index}_field"]
                valid = arrays[f"mode_{index}_observed"]
                rms = float(np.sqrt(np.sum(field[valid] ** 2) / valid.sum())) if valid.any() else None
                for reported in mode["boundaries"]:
                    actual, _ = boundary_motion(field.reshape(32, 32, 2), arrays["owners"].reshape(32, 32),
                                                valid.reshape(32, 32), stride=reported["stride"])
                    for which in ("cross_region", "same_region"):
                        measured = actual[which]["jump_rms"]
                        actual[which]["jump_over_field_rms"] = measured / rms if measured is not None and rms > 1e-20 else None
                    if actual != reported:
                        raise ValueError("reported boundary statistic differs from arrays")
            videos.append({"candidate_id": key, "mode_count": len(row["modes"]), "checked": True})
        scientific = {k: v for k, v in row.items() if k not in ("candidate_id", "base_id", "family", "seed")}
        if row["family"] == "original":
            originals[row["base_id"]] = (scientific, arrays)
        elif row["family"] == "encoding_control":
            original, original_arrays = originals[row["base_id"]]
            check = {"candidate_id": key, "fields_exact": scientific == original,
                     "arrays_exact": set(arrays) == set(original_arrays) and all(np.array_equal(v, original_arrays[k], equal_nan=True) for k, v in arrays.items())}
            if not check["fields_exact"] or not check["arrays_exact"]:
                raise ValueError("encoding control parity failed")
            controls.append(check)
    statuses = {r["base_id"]: r["original_reliability_status"] for r in rows if r["family"] == "original"}
    valid_pairs = sum(r["family"] == "local_texture_alternating" and r["original_reliability_status"] == statuses[r["base_id"]] == "succeeded" for r in rows)
    report = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
              "runtime": runtime, "source_provenance_sha256": digest(source / "provenance.json"),
              "source_diagnostics_sha256": identity["diagnostics_sha256"], "script_sha256": digest(Path(__file__)),
              "metric_helper_sha256": digest(Path(__file__).resolve().parents[2] / "metrics/dynamic-degree/src/dynamic_degree/image_plane_modes.py"),
              "videos": videos, "encoding_controls": controls, "existing_valid_CF_pairs_not_new_scores": valid_pairs,
              "warning": "conditional model-flow reduction and algebraic constraints are not completed Repair validation"}
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"videos": len(videos), "controls": controls, "existing_valid_CF_pairs": valid_pairs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
