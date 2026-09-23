"""Verify complete motion-explanation caches, controls and reported errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe_region_time_modes import compare_predictions
from .static_jitter import digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "execution-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--parent-run")
    args = p.parse_args(argv)
    source, execution, output = map(Path, (args.source_run, args.execution_root, args.output))
    if output.exists():
        raise FileExistsError("fresh audit required")
    identity = json.loads((source / "provenance.json").read_text())
    runtime = json.loads((source / "runtime.json").read_text())
    rows = list(map(json.loads, (source / "diagnostics.jsonl").read_text().splitlines()))
    crossed = "ranks" in identity
    script = "probe_region_time_modes.py" if crossed else "probe_image_plane_modes.py"
    if (runtime["status"] != "finished" or runtime["completed"] != runtime["expected"] or runtime["failed"]
            or len(rows) != len(identity["input_sha256"]) or {r["candidate_id"] for r in rows} != set(identity["input_sha256"])
            or digest(source / "diagnostics.jsonl") != identity["diagnostics_sha256"]
            or {p.stem for p in (source / "evidence").glob("*.npz")} != set(identity["input_sha256"])):
        raise ValueError("partial/changed video cohort or cache")
    for name, sha in {**identity["code_files"], **identity.get("helper_sha256", {}),
                      f"scripts/counterfactual/{script}": identity["script_sha256"]}.items():
        if digest(execution / name) != sha:
            raise ValueError("execution snapshot differs")
    parent = Path(args.parent_run) if args.parent_run else None
    if crossed and (parent is None or digest(parent / "provenance.json") != identity["source_provenance_sha256"]
                    or digest(parent / "diagnostics.jsonl") != identity["source_diagnostics_sha256"]):
        raise ValueError("crossed prediction needs its exact parent cache")
    originals, controls, stats = {}, [], []
    for row in rows:
        key = row["candidate_id"]
        cache = source / "evidence" / f"{key}.npz"
        if digest(cache) != identity["evidence_sha256"][key] or row["score"] is not None or row["status"] != "diagnostic_only":
            raise ValueError("changed output or non-diagnostic result")
        with np.load(cache, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if any(np.isinf(value).any() for value in arrays.values()):
            raise ValueError("nonfinite overflow is not missing evidence")
        if crossed:
            path = parent / "evidence" / f"{key}.npz"
            if digest(path) != identity["source_evidence_sha256"][key]:
                raise ValueError("parent cache changed")
            with np.load(path, allow_pickle=False) as z:
                predicted = {rank: arrays[f"prediction_rank_{rank}"] for rank in identity["ranks"]}
                calculated = compare_predictions(z["velocity"], z["reliable_pair"], predicted, z["timestamps"])
            if calculated != row["comparisons"]:
                raise ValueError("reported crossed-prediction errors differ from actual arrays")
            scientific = {k: row[k] for k in ("comparisons", "regions", "original_reliability_status")}
            stats.append({"candidate_id": key, "comparisons": calculated})
        else:
            velocity, dt = arrays["velocity"], np.diff(arrays["timestamps"])
            if velocity.shape != (len(dt), row["points"], 2) or len(dt) + 1 != len(row["sampling"]["timestamps"]):
                raise ValueError("native phase coverage differs")
            if arrays["point_valid"].sum() != row["point_valid"]:
                raise ValueError("evidence counts differ")
            common = arrays["point_valid"].copy()
            for grid in identity["config"]["spatial_grids"]:
                common &= np.isfinite(arrays[f"prediction_grid_{grid}"]).all(axis=(0, 2))
            truth = arrays["centered_velocity"][:, common]
            energy = float(np.sum(truth ** 2 * dt[:, None, None])) if common.any() else None
            for comparison in row["leave_region_out"]:
                field = arrays[f'prediction_grid_{comparison["grid"]}']
                error = float(np.sum((field[:, common] - truth) ** 2 * dt[:, None, None])) if common.any() else None
                if (comparison["common_comparison_points"] != int(common.sum())
                        or comparison["observed_energy_on_common"] != energy or comparison["squared_error_on_common"] != error):
                    raise ValueError("reported leave-region error differs from arrays")
            scientific = {k: row[k] for k in ("point_valid", "pair_reliable_fraction", "region_ownership", "modes_all_points",
                                             "modes_reliable_points", "leave_region_out", "original_reliability_status")}
            stats.append({"candidate_id": key, "point_valid": row["point_valid"], "common_points": int(common.sum()),
                          "leading_mode": row["modes_reliable_points"][0] if row["modes_reliable_points"] else None})
        if row["family"] == "original":
            originals[row["base_id"]] = (scientific, arrays)
        elif row["family"] == "encoding_control":
            before, saved = originals[row["base_id"]]
            controls.append({"candidate_id": key, "all_scientific_fields_exact": before == scientific,
                             "all_arrays_exact": set(saved) == set(arrays) and all(np.array_equal(saved[k], value, equal_nan=True) for k, value in arrays.items())})
    valid_sources = {r["base_id"]: r["original_reliability_status"] for r in rows if r["family"] == "original"}
    valid_pairs = sum(r["original_reliability_status"] == valid_sources[r["base_id"]] == "succeeded"
                      for r in rows if r["family"] == "local_texture_alternating")
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
              "source_provenance_sha256": digest(source / "provenance.json"), "source_diagnostics_sha256": identity["diagnostics_sha256"],
              "script_sha256": digest(Path(__file__)), "comparison_helper_sha256": digest(Path(__file__).with_name("probe_region_time_modes.py")),
              "runtime": runtime, "videos": stats, "encoding_controls": controls,
              "existing_source_valid_CF_pairs_not_new_scores": valid_pairs,
              "warning": "error reduction predicts model flow; it does not prove physical artifact removal or valid Repair"}
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"videos": len(rows), "controls": controls, "existing_source_valid_CF_pairs": valid_pairs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
