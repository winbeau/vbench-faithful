"""All-phase, all-reviewed-video image-plane mode/leave-region-out diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.dense_correspondence import DenseCorrespondenceConfig, dense_evidence
from dynamic_degree.image_plane_modes import region_partition, temporal_modes, leave_region_prediction
from dynamic_degree.trajectory import decode_video
from vbench_audit_models.sam_regions import unpack_masks
from .probe_native_region_motion import resolve_media
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def inspect_fields(frames, times, forward, backward, masks, config):
    height, width = frames.shape[1:3]
    margin = config.patch_radius + 1
    xx, yy = np.meshgrid(np.linspace(margin, width - 1 - margin, 32), np.linspace(margin, height - 1 - margin, 32))
    xy = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    evidence = dense_evidence(frames, forward, backward, config, query_positions=xy)
    reliable = (evidence["inside_pair"] & evidence["visible_pair"]
                & (evidence["cycle_error"] <= config.cycle_error_max)
                & (evidence["moving_error"] <= config.match_error_max))
    dt = np.diff(times)
    fraction = np.average(reliable, axis=0, weights=dt)
    valid = fraction >= .6
    owners = region_partition(masks, xy)
    velocity = evidence["flow_vectors"] / dt[:, None, None]
    centered = velocity - np.average(velocity, axis=0, weights=dt)
    arrays = {"xy": xy, "owners": owners, "reliable_pair": reliable, "point_valid": valid,
              "velocity": velocity, "centered_velocity": centered, "timestamps": times}
    predictions, folds = {}, {}
    for grid in (0, 3, 5, 9):
        predicted, records = leave_region_prediction(centered, xy, owners, valid, (height, width), grid=grid)
        predictions[grid] = predicted; folds[grid] = records
        arrays[f"prediction_grid_{grid}"] = predicted
    common = valid.copy()
    for prediction in predictions.values():
        common &= np.isfinite(prediction).all(axis=(0, 2))
    observed_energy = float(np.sum(centered[:, common] ** 2 * dt[:, None, None])) if common.any() else None
    comparisons = []
    for grid, predicted in predictions.items():
        tested = np.isfinite(predicted).all(axis=(0, 2)) & valid
        mse = float(np.sum((predicted[:, common] - centered[:, common]) ** 2 * dt[:, None, None])) if common.any() else None
        comparisons.append({"grid": grid, "model": "affine" if grid == 0 else "affine_plus_gaussian_basis",
                            "testable_points": int(tested.sum()), "common_comparison_points": int(common.sum()),
                            "squared_error_on_common": mse, "observed_energy_on_common": observed_energy,
                            "explained_on_common": 1 - mse / observed_energy if observed_energy is not None and observed_energy > 1e-20 else None,
                            "folds": folds[grid]})
    result = {"status": "diagnostic_only", "score": None, "points": len(xy), "point_valid": int(valid.sum()),
              "pair_reliable_fraction": float(np.average(reliable.mean(axis=1), weights=dt)),
              "region_ownership": [{"region": int(r), "points": int(np.sum(owners == r)),
                                    "valid_points": int(np.sum((owners == r) & valid))} for r in np.unique(owners)],
              "modes_all_points": temporal_modes(velocity, times, xy, (height, width), owners),
              "modes_reliable_points": temporal_modes(velocity[:, valid], times, xy[valid], (height, width), owners[valid]),
              "leave_region_out": comparisons,
              "warning": "SAM reference-frame partition is not persistent object identity; no motion is removed or scored"}
    return result, arrays


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "source-scores", "flow-provenance", "flow-cache", "region-run", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-root", action="append", default=[])
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest, review, scores, flow_identity, cache, regions, output = map(Path, (
        args.manifest, args.review, args.source_scores, args.flow_provenance, args.flow_cache, args.region_run, args.output))
    if output.exists():
        raise FileExistsError("fresh diagnostic output required")
    rows = select_reviewed_candidates(list(map(json.loads, manifest.read_text().splitlines())), review, manifest, root)
    previous_rows = list(map(json.loads, scores.read_text().splitlines()))
    prior = {r["candidate_id"]: r for r in previous_rows}
    flow_id = json.loads(flow_identity.read_text())
    region_id = json.loads((regions / "provenance.json").read_text())
    region_time = json.loads((regions / "runtime.json").read_text())
    expected_inputs = {r["candidate_id"]: r["sha256"] for r in rows}
    if (len(previous_rows) != len(prior) or set(prior) != set(expected_inputs)
            or flow_id["input_sha256"] != expected_inputs or region_id["input_sha256"] != expected_inputs
            or flow_id["source_scores_sha256"] != digest(scores)
            or any(i["manifest_sha256"] != digest(manifest) or i["review_sha256"] != digest(review) for i in (flow_id, region_id))
            or region_time["status"] != "finished" or region_time["failed"] or region_time["completed"] != len(rows)):
        raise ValueError("complete same-video model/region/cache provenance required")
    paths = {r["candidate_id"]: resolve_media(r, args.video_root) for r in rows}
    # Verify every transfer before creating output; never run on a partial copy.
    for row in rows:
        key = row["candidate_id"]
        if (digest(cache / f"{key}.npz") != flow_id["cache_sha256"][key]
                or digest(regions / "evidence" / f"{key}.npz") != region_id["cache_sha256"][key]):
            raise ValueError("flow or region cache identity changed/incomplete")
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    identity = {"status": "diagnostic_only", "score": None, "formal_acceptance": "NOT EVALUATED",
                "manifest_sha256": digest(manifest), "review_sha256": digest(review), "input_sha256": expected_inputs,
                "flow_provenance_sha256": digest(flow_identity), "flow_cache_sha256": flow_id["cache_sha256"],
                "region_provenance_sha256": digest(regions / "provenance.json"), "region_cache_sha256": region_id["cache_sha256"],
                "source_scores_sha256": digest(scores), "source_reliability_counts_not_upgraded": True,
                "config": {"query_grid": 32, "spatial_grids": [0, 3, 5, 9], "ridge": .001, "point_valid_fraction": .6,
                           "region_frame": "middle native frame", "modes": 3, "constant_velocity_removed_for_diagnostic": True},
                "code_files": source_hashes(root), "script_sha256": digest(Path(__file__)),
                "helper_sha256": {"scripts/counterfactual/probe_native_region_motion.py": digest(root / "scripts/counterfactual/probe_native_region_motion.py")},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "device": "cpu"},
                "evidence_sha256": {}}
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
            record = {k: row[k] for k in ("candidate_id", "base_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None, original_reliability_status=prior[key]["repair"]["status"])
            try:
                config = DenseCorrespondenceConfig(**prior[key]["repair"]["config"])
                frames, times, sampling = decode_video(paths[key], config)
                with np.load(cache / f"{key}.npz", allow_pickle=False) as z:
                    if not np.array_equal(z["shape"], frames.shape) or not np.array_equal(z["timestamps"], times):
                        raise ValueError("original flow decoding/time parity failed")
                    forward, backward = z["forward"], z["backward"]
                middle = len(frames) // 2
                with np.load(regions / "evidence" / f"{key}.npz", allow_pickle=False) as z:
                    if not np.array_equal(z["timestamps"], times) or list(z["input_shape"]) != row["decoded_shape"]:
                        raise ValueError("original native region geometry/time changed")
                    native = unpack_masks(z[f"frame_{middle}_masks_packed"], z[f"frame_{middle}_image_shape"])
                masks = np.stack([cv2.resize(m.astype(np.uint8), (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_NEAREST).astype(bool) for m in native]) if len(native) else np.zeros((0, *frames.shape[1:3]), bool)
                result, arrays = inspect_fields(frames, times, forward, backward, masks, config)
                record.update(result, sampling=sampling, reference_region_frame=middle, reference_native_mask_count=len(native))
                path = output / "evidence" / f"{key}.npz"
                np.savez_compressed(path, **arrays)
                identity["evidence_sha256"][key] = digest(path)
                print(json.dumps({"candidate_id": key, "valid_points": result["point_valid"],
                                  "cv": [{"grid": r["grid"], "explained": r["explained_on_common"]} for r in result["leave_region_out"]]}), flush=True)
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
