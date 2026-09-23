"""Audit the complete DEV tracker comparison; predicted motion is NOT a score.

All 16 query starts, all 20 videos and all query locations are retained.
Only fixed-grid fields enter the existing guarded decomposition. Learned
visibility is explicitly a proxy, not independent physical correspondence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamic_degree.image_plane_modes import region_partition, regional_reversal_ablation, spatial_design
from vbench_audit_models.sam_regions import unpack_masks
from .static_jitter import digest


def pair_values(arrays, source, target):
    tracks = np.asarray(arrays["tracks"], float)
    shape = arrays["shape"]
    delta = tracks[target] - tracks[source]
    in_frame = np.all((tracks >= 0) & (tracks <= [shape[2]-1, shape[1]-1]), axis=-1)
    observed = arrays["visible"][source] & arrays["visible"][target] & in_frame[source] & in_frame[target]
    return delta, observed


def motion_summary(phases):
    """Equal time-start weights; adaptive SIFT spatial density remains a caveat."""
    times = phases[0]["timestamps"]
    short_side = min(phases[0]["shape"][1:3])
    lags = []
    for lag in (1, 2, 3, 4):
        all_start, conditional_start, coverage = [], [], []
        for t in range(len(times)-lag):
            delta, observed = pair_values(phases[t], t, t+lag)
            speed = np.linalg.norm(delta, axis=-1) / (times[t+lag]-times[t]) / short_side
            all_start.append(float(speed.mean()))
            conditional_start.append(float(speed[observed].mean()) if observed.any() else None)
            coverage.append(float(observed.mean()))
        lags.append({"lag": lag, "native_start_count": len(all_start),
                     "unfiltered_model_speed": float(np.mean(all_start)),
                     "visible_inframe_conditional_speed": float(np.mean(conditional_start)) if all(v is not None for v in conditional_start) else None,
                     "model_visible_inframe_fraction": float(np.mean(coverage)),
                     "per_start_unfiltered_model_speed": all_start,
                     "per_start_conditional_speed": conditional_start,
                     "per_start_proxy_coverage": coverage})
    return {"units": "short_side_lengths_per_second", "lags": lags, "score": None,
            "warning": "model predictions only, unfiltered includes invisible coordinates; conditional speed is not a complete score"}


def verify_arrays(arrays, row):
    queries = np.asarray(row["queries_txy"], np.float32)
    t, n = row["input_shape"][0], len(queries)
    if (set(arrays) != {"queries_txy", "timestamps", "shape", "tracks", "visible"}
            or not np.array_equal(arrays["queries_txy"], queries)
            or not np.array_equal(arrays["timestamps"], row["timestamps"])
            or not np.array_equal(arrays["shape"], row["input_shape"])
            or arrays["tracks"].shape != (t, n, 2) or not np.isfinite(arrays["tracks"]).all()
            or arrays["visible"].shape != (t, n) or arrays["visible"].dtype != np.bool_):
        raise ValueError("native geometry/query/cache contract differs")
    return float(np.max(np.abs(arrays["tracks"][queries[:, 0].astype(int), np.arange(n)]-queries[:, 1:])))


def audit_arm(run, requests, execution):
    files = sorted(run.glob("shard-*/frame-*/provenance.json"))
    if len(files) != 16:
        raise ValueError("all 16 query phases must have final provenance")
    all_phase, metadata, provenance = {}, {}, []
    controls = []
    expected_candidates = {f"video_{i:06}" for i in range(20)}
    for path in files:
        identity = json.loads(path.read_text())
        request = identity["request"]
        phase = request["query_frame"]
        if phase in all_phase or not 0 <= phase < 16:
            raise ValueError("duplicate/unexpected phase")
        bound_request = requests / f"frame-{phase:02}.json"
        if (digest(bound_request) != identity["request_sha256"]
                or json.loads(bound_request.read_text()) != request
                or request["tracker_kind"] != "cotracker3-offline"):
            raise ValueError("prepared request differs")
        for name, sha in {**identity["code_files"], **identity["helper_sha256"],
                          "scripts/counterfactual/probe_feature_tracks.py": identity["script_sha256"]}.items():
            if digest(execution / name) != sha:
                raise ValueError(f"execution snapshot differs: {name}")
        assets = identity["verified_tracker_assets"]
        if (assets["source_files"] != request["tracker_code"]
                or assets["checkpoint_sha256"] != request["tracker_weight_sha256"]
                or assets["checkpoint_size_bytes"] != request["tracker_weight_size_bytes"]):
            raise ValueError("verified model identity differs")
        runtime = json.loads((path.parent / "runtime.json").read_text())
        rows = [json.loads(s) for s in (path.parent / "diagnostics.jsonl").read_text().splitlines()]
        if (runtime["status"] != "finished" or runtime["expected"] != 20 or runtime["completed"] != 20 or runtime["failed"]
                or len(rows) != 20 or {r["candidate_id"] for r in rows} != expected_candidates
                or any(r["status"] != "diagnostic_only" or r["score"] is not None for r in rows)
                or {r["candidate_id"] for r in request["videos"]} != expected_candidates
                or {p.stem for p in (path.parent / "evidence").glob("*.npz")} != expected_candidates):
            raise ValueError("incomplete/failed comparison retained; cannot issue complete audit")
        current = {}
        forced_max = 0.
        for row in request["videos"]:
            key = row["candidate_id"]
            cache = path.parent / "evidence" / f"{key}.npz"
            if digest(cache) != identity["cache_sha256"][key]:
                raise ValueError("changed prediction cache")
            with np.load(cache, allow_pickle=False) as z:
                arrays = {name: z[name] for name in z.files}
            forced_max = max(forced_max, verify_arrays(arrays, row))
            fixed = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed", "sha256", "input_shape", "timestamps", "frames_sha256")}
            if key in metadata and fixed != metadata[key]:
                raise ValueError("video identity differs across query phases")
            metadata[key] = fixed
            current[key] = arrays
        for row in request["videos"]:
            if row["family"] != "encoding_control":
                continue
            original = next(r["candidate_id"] for r in request["videos"] if r["base_id"] == row["base_id"] and r["family"] == "original")
            a, b = current[original], current[row["candidate_id"]]
            controls.append({"phase": phase, "candidate_id": row["candidate_id"],
                             "all_arrays_exact": all(np.array_equal(a[k], b[k]) for k in a),
                             "max_coordinate_difference_pixels": float(np.max(np.abs(a["tracks"]-b["tracks"]))),
                             "visibility_equal": np.array_equal(a["visible"], b["visible"])})
        all_phase[phase] = current
        provenance.append({"phase": phase, "provenance_sha256": digest(path),
                           "diagnostics_sha256": digest(path.parent / "diagnostics.jsonl"),
                           "runtime": runtime, "environment": identity["environment"],
                           "forced_query_coordinate_max_error_pixels": forced_max})
    videos = {key: [all_phase[t][key] for t in range(16)] for key in sorted(expected_candidates)}
    return videos, metadata, {"source_runs": sorted(provenance, key=lambda r:r["phase"]),
                              "encoding_controls": controls, "complete_calls": 320,
                              "query_observations": sum(len(z["queries_txy"]) for phases in videos.values() for z in phases)}


def guarded_grid(phases, masks):
    xy = phases[0]["queries_txy"][:, 1:].astype(float)
    times, shape = phases[0]["timestamps"], phases[0]["shape"][1:3]
    if any(not np.array_equal(z["queries_txy"][:, 1:], xy) for z in phases):
        raise ValueError("fixed spatial denominator required")
    deltas, observed = zip(*(pair_values(phases[t], t, t+1) for t in range(15)))
    velocity = np.stack(deltas)/np.diff(times)[:, None, None]
    known = np.stack(observed)
    owners = region_partition(masks, xy)
    result, arrays = regional_reversal_ablation(velocity, times, xy, owners, known, shape)
    # Independently check the defining safeguards, not just the reported residual.
    removal = arrays["removal"]
    affine_error = 0.
    design = spatial_design(xy, shape, 0)
    for t in range(15):
        for region in np.unique(owners):
            selected = known[t] & (owners == region)
            affine_error = max(affine_error, float(np.max(np.abs(design[selected].T @ removal[t, selected]), initial=0.)))
    mean_error = float(np.max(np.abs(np.sum(removal*np.diff(times)[:, None, None], axis=0))))
    if np.any(removal[~known]) or max(affine_error, mean_error) > 1e-5:
        raise ValueError("protected decomposition violated affine/mean/missing constraints")
    result.update(evidence_role="learned visibility plus image bounds ONLY, not independently certified correspondence",
                  independent_affine_constraint_max_error=affine_error,
                  independent_mean_constraint_max_error=mean_error,
                  reference_region_frame=8, grid_size=12)
    return result, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "sift-requests", "grid-requests", "execution-root", "region-run", "output"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args()
    output, run, execution = Path(args.output), Path(args.run_root), Path(args.execution_root)
    if output.exists():
        raise FileExistsError("fresh audit required")
    result = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
              "script_sha256": digest(Path(__file__)), "arms": {}}
    grid = None
    for arm in ("sift", "grid"):
        videos, metadata, audit = audit_arm(run / arm, Path(getattr(args, arm+"_requests")), execution)
        audit["videos"] = [{**metadata[key], "motion": motion_summary(phases)} for key, phases in videos.items()]
        result["arms"][arm] = audit
        if arm == "grid":
            grid = videos
    regions = Path(args.region_run)
    region_id = json.loads((regions/"provenance.json").read_text())
    if region_id["input_sha256"] != {key: row["sha256"] for key, row in metadata.items()}:
        raise ValueError("SAM cohort differs")
    result["region_provenance_sha256"] = digest(regions/"provenance.json")
    output.mkdir(parents=True)
    (output/"guarded_evidence").mkdir()
    result["guarded_grid"] = []
    for key, phases in grid.items():
        region = regions/"evidence"/f"{key}.npz"
        if digest(region) != region_id["cache_sha256"][key]:
            raise ValueError("SAM bytes changed")
        with np.load(region, allow_pickle=False) as z:
            if not np.array_equal(z["timestamps"], phases[0]["timestamps"]) or not np.array_equal(z["input_shape"], phases[0]["shape"]):
                raise ValueError("SAM geometry differs")
            masks = unpack_masks(z["frame_8_masks_packed"], z["frame_8_image_shape"])
        row, arrays = guarded_grid(phases, masks)
        np.savez_compressed(output/"guarded_evidence"/f"{key}.npz", **arrays)
        result["guarded_grid"].append({**metadata[key], **row,
            "evidence_sha256": digest(output/"guarded_evidence"/f"{key}.npz")})
    result["not_verified"] = ["independent physical point identity", "natural-motion calibration/noninferiority", "complete Repair scores", "formal holdout"]
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"arms": {k:{"calls": v["complete_calls"],"queries":v["query_observations"]} for k,v in result["arms"].items()},
                      "guarded": [{"id":r["candidate_id"], "raw":r["conditional_raw_speed"],"guarded":r["conditional_guarded_speed"],"coverage":r["reliable_pair_fraction"]} for r in result["guarded_grid"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
