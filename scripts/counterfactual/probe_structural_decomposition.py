"""DEV ONLY: motion-decomposition diagnostics on cached real local tracks.

This is not a final scorer: all-track estimates include uncertain tracks, so no
diagnostic is promoted to a valid Repair score. Source insufficiency is retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from dynamic_degree.dense_correspondence import sample_field
from dynamic_degree.common_mode import CommonModeConfig, decompose_common_modes
from dynamic_degree.dense_paths import local_dense_paths
from dynamic_degree.local_trajectory import LocalTrajectoryConfig
from dynamic_degree.structural_decomposition import DecompositionConfig, decompose_tracks
from dynamic_degree.trajectory import decode_video
from .analyze_static_jitter import stats
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def probe(frames, timestamps, cached, source_config, config, *, variant="local-support"):
    if variant not in ("local-support", "common-mode"):
        raise ValueError("unknown decomposition variant")
    decomposer = decompose_tracks if variant == "local-support" else decompose_common_modes
    windows = cached["windows"]
    owner = cached["transition_owner"]
    names = ("raw", "temporal_only", "structure_only", "full")
    count = len(cached["queries"])
    values = {name: np.empty((len(frames) - 1, count, 2)) for name in names}
    gates = np.ones((len(frames) - 1, count))
    tested = np.zeros(gates.shape, bool)
    correspondence = np.zeros(gates.shape, bool)
    summaries = []
    for index, (start, stop) in enumerate(windows):
        xy = cached[f"window_{index}_tracks"]
        if f"window_{index}_reliable_pair" in cached:
            pair_reliable = cached[f"window_{index}_reliable_pair"]
        else:
            reverse = cached[f"window_{index}_reverse_tracks"]
            reliable = (cached[f"window_{index}_visible"] & cached[f"window_{index}_reverse_visible"]
                        & (np.linalg.norm(xy - reverse, axis=-1) <= source_config.cycle_error_max)
                        & (xy[..., 0] >= 0) & (xy[..., 0] <= frames.shape[2] - 1)
                        & (xy[..., 1] >= 0) & (xy[..., 1] <= frames.shape[1] - 1))
            pair_reliable = reliable[:-1] & reliable[1:]
        # Appearance only; no object names, intervention labels, or pairing.
        mean_color = cv2.GaussianBlur(frames[start].astype(np.float32) / 255, (0, 0), 3.)
        colors = sample_field(mean_color, xy[0])
        result = decomposer(xy, timestamps[start:stop], min(frames.shape[1:3]), config,
                                  colors=np.clip(colors, 0, 1), reliable=pair_reliable)
        pairs = np.flatnonzero(owner == index)
        local = pairs - start
        correspondence[pairs] = pair_reliable[local]
        for name in names:
            values[name][pairs] = result[f"{name}_deltas"][local]
        if variant == "local-support":
            gates[pairs] = result["residual_gate"][local]
            tested[pairs] = result["structural_tested"][local]
        summaries.append({"start": int(start), "stop": int(stop), "owned_pairs": pairs.tolist(),
                          "lag_speeds": result["lag_speeds"],
                          "pair_correspondence_fraction": float(pair_reliable.mean()),
                          "reversing_fraction": float(np.mean(result["reversal_fraction"] > config.minimum_reversal_fraction))})
        if "modes" in result:
            summaries[-1]["modes"] = result["modes"]
    denominator = (timestamps[-1] - timestamps[0]) * count * min(frames.shape[1:3])
    return {"all_track_diagnostics": {name: float(np.linalg.norm(value, axis=-1).sum() / denominator)
                                       for name, value in values.items()},
            "variant": variant,
            "decomposition_pair_evidence_fraction": float(np.average(correspondence.mean(axis=1), weights=np.diff(timestamps))),
            "mean_residual_gate": float(gates.mean()) if variant == "local-support" else None,
            "structurally_tested_fraction": float(tested.mean()) if variant == "local-support" else None,
            "suppressed_point_pair_fraction": float(np.mean(gates < .5)) if variant == "local-support" else None,
            "windows": summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--source-scores", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("local-support", "common-mode"), default="local-support")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    manifest = Path(args.manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows = select_reviewed_candidates(rows, Path(args.review), manifest, root)
    source_path = Path(args.source_scores)
    source_rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    source = {r["candidate_id"]: r for r in source_rows}
    if len(source_rows) != len(source) or set(source) != {r["candidate_id"] for r in rows}:
        raise ValueError("source scoring must uniquely cover the entire reviewed cohort")
    source_provenance = source_path.with_name("provenance.json")
    original_identity = json.loads(source_provenance.read_text())
    source_variant = original_identity.get("repair_variant")
    if original_identity.get("review_sha256") != digest(Path(args.review)) or source_variant not in ("local-trajectory", "dense-correspondence"):
        raise ValueError("expected a bound local-trajectory or dense-correspondence source run")
    config_type = DecompositionConfig if args.variant == "local-support" else CommonModeConfig
    config = config_type(**json.loads(Path(args.config).read_text()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"manifest_sha256": digest(manifest), "review_sha256": digest(Path(args.review)),
                  "source_scores_sha256": digest(source_path), "source_provenance_sha256": digest(source_provenance),
                  "config": asdict(config), "config_sha256": digest(Path(args.config)),
                  "decomposition_variant": args.variant,
                  "code_files": source_hashes(root), "probe_script_sha256": digest(Path(__file__)),
                  "source_model_identity": original_identity,
                  "model_inference_reused": True, "formal_acceptance": "NOT EVALUATED",
                  "path_source": source_variant,
                  "reliability_used": ("tracker_visibility_and_local_cycle; not_all_track_truth" if source_variant == "local-trajectory"
                                       else "advected_dense_paths; pair_local_cycle_and_deformable_patch; geometric_visibility_proxy"),
                  "input_sha256": {}, "cache_sha256": {}}
    runtime = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "expected": len(rows)}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    records = []
    started = time.monotonic()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key = row["candidate_id"]
            prior = source[key]["repair"]
            video = Path(row["video"])
            if digest(video) != row["sha256"] or source[key]["input_sha256"] != row["sha256"]:
                raise ValueError("source video identity changed")
            config_before = LocalTrajectoryConfig(**prior["config"])
            frames, timestamps, _ = decode_video(video, config_before)
            cache_path = source_path.parent / "evidence" / f"{key}.npz"
            with np.load(cache_path, allow_pickle=False) as cache:
                if not np.array_equal(cache["shape"], frames.shape) or not np.array_equal(cache["timestamps"], timestamps):
                    raise ValueError("cached tracks do not match decoding geometry/timing")
                path_cache = cache
                if source_variant == "dense-correspondence":
                    fixed_grid_raw = float(np.linalg.norm(cache['flow_vectors'], axis=-1).sum()
                                           / ((timestamps[-1] - timestamps[0]) * len(cache['queries']) * min(frames.shape[1:3])))
                    if not np.isclose(fixed_grid_raw, prior['ablations']['raw_tracks'], rtol=1e-6, atol=1e-9):
                        raise ValueError("cached Eulerian flow replay parity failed")
                    path_cache = local_dense_paths(frames, timestamps, cache['forward'], cache['backward'], cache['queries'], config_before)
                diagnostics = probe(frames, timestamps, path_cache, config_before, config, variant=args.variant)
            if source_variant == "local-trajectory" and not np.isclose(
                    diagnostics["all_track_diagnostics"]["raw"], prior["ablations"]["raw_tracks"], rtol=1e-6, atol=1e-9):
                raise ValueError("raw cached trajectory replay parity failed")
            record = {"candidate_id": key, "input_sha256": row["sha256"], "base_id": row["base_id"],
                      "prompt_id": row["prompt_id"], "family": row["family"], "seed": row["seed"],
                      "status": "diagnostic_only", "score": None, "original_reliability_status": prior["status"],
                      "path_source": source_variant,
                      "original_coverage": prior["coverage"], **diagnostics}
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            provenance["input_sha256"][key] = row["sha256"]
            provenance["cache_sha256"][key] = digest(cache_path)
            print(json.dumps({"candidate_id": key, "all_track_diagnostics": diagnostics["all_track_diagnostics"]}), flush=True)
    bases = {r["base_id"]: r for r in records if r["family"] == "original"}
    cfs = [r for r in records if r["family"] == "local_texture_alternating"]
    summary = {"status": "all_track_development_diagnostics_not_verified_repair_scores", "count": len(records),
               "variant": args.variant,
               "path_source": source_variant,
               "diagnostic_pair_count": len(cfs), "source_reliability_valid_pair_count": sum(
                   r["original_reliability_status"] == bases[r["base_id"]]["original_reliability_status"] == "succeeded" for r in cfs),
               "methods": {}}
    for method in ("raw", "temporal_only", "structure_only", "full"):
        before = [bases[r["base_id"]]["all_track_diagnostics"][method] for r in cfs]
        after = [r["all_track_diagnostics"][method] for r in cfs]
        groups = [r["prompt_id"] for r in cfs]
        summary["methods"][method] = {"base": stats(before, groups), "counterfactual": stats(after, groups),
                                       "absolute_delta": stats(np.abs(np.asarray(after) - before).tolist(), groups)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    runtime.update(status="finished", completed=len(records), elapsed_seconds=time.monotonic() - started,
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
