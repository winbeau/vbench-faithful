"""DEV-only independent DINO structure evidence on the fixed reviewed cohort.

This does not score, tune, change Origin, or assume descriptor matches are true
motion. All original/control/CF inputs and all lag phases remain visible.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.structure_correspondence import match_similarity, pair_diagnostics, transitivity_diagnostics
from dynamic_degree.trajectory import TrajectoryConfig, decode_video
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes
from .static_jitter import digest


def describe_evidence(cache, timestamps, facets, lags, device):
    import torch
    from torch.nn.functional import normalize

    q, grid = cache["queries"], cache["grid_shape"]
    short_side = min(cache["input_shape"][1:3])
    arrays, summary = {}, {}
    for facet in facets:
        features = normalize(torch.from_numpy(cache[facet].astype(np.float32)).to(device), dim=-1)
        matches, records = {}, []
        for lag in lags:
            for start in range(len(timestamps) - lag):
                with torch.inference_mode():
                    similarity = (features[start] @ features[start + lag].T).cpu().numpy()
                match = match_similarity(similarity, q, grid)
                matches[start, lag] = match
                prefix = f"{facet}_{start}_{lag}_"
                arrays.update({prefix + name: value for name, value in match.items()})
                diagnostic = pair_diagnostics(match, q, cache["attention"][start],
                                              timestamps[start + lag] - timestamps[start], short_side)
                records.append({"start": start, "lag": lag, "seconds": float(timestamps[start + lag] - timestamps[start]),
                                **diagnostic})
        triples = [{"start": i, **transitivity_diagnostics(matches[i, 1], matches[i + 1, 1], matches[i, 2], q)}
                   for i in range(len(timestamps) - 2)] if 1 in lags and 2 in lags else []
        by_lag = {}
        for lag in lags:
            subset = [r for r in records if r["lag"] == lag]
            keys = [k for k in subset[0] if k not in ("start", "lag", "seconds")]
            by_lag[str(lag)] = {"frame_pairs": len(subset), "summary": {
                name: {"mean": float(np.mean([r[name] for r in subset if r[name] is not None]))
                       if any(r[name] is not None for r in subset) else None,
                       "finite_pairs": sum(r[name] is not None for r in subset)} for name in keys}}
        summary[facet] = {"by_lag": by_lag, "pairs": records, "transitivity": triples}
        del features
    return summary, arrays


def native_token_parity(model, frame, cache):
    import torch
    from torch.nn import functional as F

    x = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None].to(model.device, dtype=torch.float32) / 255
    x = F.interpolate(x, tuple(map(int, cache["model_shape"])), mode="bilinear", align_corners=False)
    x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.inference_mode():
        native = model.model.get_intermediate_layers(x, n=1)[0][:, 1:]
        native = F.normalize(native, dim=-1)[0].cpu().numpy().astype(np.float16)
    if not np.array_equal(native, cache["token11"][0]):
        raise ValueError("native final patch-token extraction parity failed")
    return {"frame": 0, "facet": "token11", "cache_float16_max_absolute_difference": 0.0,
            "scope": "first_video_first_frame_same_preprocessing; not motion accuracy"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dino-root")
    parser.add_argument("--dino-weight")
    parser.add_argument("--replay-source", help="completed bound descriptor run; reuse features without model inference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-free-gib", type=float, default=20.)
    args = parser.parse_args(argv)
    if args.replay_source and (args.dino_root or args.dino_weight):
        parser.error("replay uses bound caches, not new model options")
    if not args.replay_source and not (args.dino_root and args.dino_weight):
        parser.error("local DINO root/weight required unless replaying completed caches")
    import torch
    from vbench_audit_models.dino_dense import DinoDenseModel

    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device.startswith("cuda"):
        device_index = torch.device(args.device).index or 0
        free, _ = torch.cuda.mem_get_info(device_index)
        if free < args.minimum_free_gib * 1024 ** 3:
            raise RuntimeError("insufficient free GPU memory; no model/run started")
        torch.cuda.set_per_process_memory_fraction(.10, device_index)
    root = Path(__file__).resolve().parents[2]
    manifest, review = Path(args.manifest), Path(args.review)
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], review, manifest, root)
    config = json.loads(Path(args.config).read_text())
    if config["facets"] != ["key9", "token11"] or config["lags"] != [1, 2, 3, 4] or config["patch_size"] != 16:
        raise ValueError("fixed diagnostic facets/lags/patch size required")
    sampling = TrajectoryConfig(sample_fps=config["sample_fps"], max_side=config["decode_max_side"])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fresh output required; no overwriting or implicit restart")
    replay = Path(args.replay_source) if args.replay_source else None
    original_identity = None
    model = None
    if replay:
        previous_runtime = json.loads((replay / "runtime.json").read_text())
        original_identity = json.loads((replay / "provenance.json").read_text())
        if (previous_runtime["status"] != "finished" or previous_runtime["failed"]
                or set(original_identity["cache_sha256"]) != {r["candidate_id"] for r in rows}
                or original_identity["manifest_sha256"] != digest(manifest)
                or original_identity["review_sha256"] != digest(review)):
            raise ValueError("replay requires a completed, bound, failure-free descriptor run")
        for k in ("sample_fps", "decode_max_side", "model_max_side", "patch_size", "facets", "lags"):
            if config[k] != original_identity["config"][k]:
                raise ValueError(f"replay cannot change descriptor geometry/facets: {k}")
    else:
        model = DinoDenseModel(args.dino_root, args.dino_weight, args.device, max_side=config["model_max_side"])
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    identity = {"manifest_sha256": digest(manifest), "review_sha256": digest(review),
                "config": config, "config_sha256": digest(Path(args.config)),
                "model_identity": original_identity["model_identity"] if replay else model.identity,
                "model_inference_reused": bool(replay), "feature_source": str(replay) if replay else None,
                "source_provenance_sha256": digest(replay / "provenance.json") if replay else None,
                "source_cache_sha256": original_identity["cache_sha256"] if replay else None,
                "code_files": source_hashes(root),
                "script_sha256": digest(Path(__file__)), "input_sha256": {}, "cache_sha256": {},
                "status": "diagnostic_only", "formal_acceptance": "NOT EVALUATED",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    runtime = {"status": "running", "pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "expected": len(rows), "completed": 0, "failed": 0}
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    started = time.monotonic()
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key, video = row["candidate_id"], Path(row["video"])
            record = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None)
            try:
                if digest(video) != row["sha256"]:
                    raise ValueError("immutable media identity changed")
                frames, timestamps, sampled = decode_video(video, sampling)
                if replay:
                    source_cache = replay / "evidence" / f"{key}.npz"
                    if digest(source_cache) != original_identity["cache_sha256"][key]:
                        raise ValueError("source descriptor cache changed")
                    with np.load(source_cache, allow_pickle=False) as z:
                        if not np.array_equal(z["timestamps"], timestamps) or not np.array_equal(z["input_shape"], frames.shape):
                            raise ValueError("descriptor cache timeline/geometry mismatch")
                        cache = {k: z[k] for k in ("key9", "token11", "attention", "queries", "grid_shape", "model_shape", "input_shape")}
                else:
                    cache = model.extract(frames)
                if not replay and runtime["completed"] == 0 and config["first_video_native_forward_parity"]:
                    identity["native_token_parity"] = native_token_parity(model, frames[0], cache)
                description, matched = describe_evidence(cache, timestamps, config["facets"], config["lags"], args.device)
                path = output / "evidence" / f"{key}.npz"
                # Replays bind original feature caches, do not duplicate a GB of descriptors.
                saved = {k: v for k, v in cache.items() if not replay or k not in ("key9", "token11")}
                np.savez_compressed(path, **saved, **matched, timestamps=timestamps)
                record.update(sampling=sampled, model_shape=cache["model_shape"].tolist(),
                              grid_shape=cache["grid_shape"].tolist(), descriptors=description)
                identity["input_sha256"][key] = row["sha256"]
                identity["cache_sha256"][key] = digest(path)
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            runtime["completed"] += 1
            print(json.dumps({"candidate_id": key, "status": record["status"], "error": record.get("error")}), flush=True)
            (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
            (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    runtime.update(status="finished", elapsed_seconds=time.monotonic() - started,
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if args.device.startswith("cuda"):
        runtime["gpu_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        runtime["gpu_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    return 1 if runtime["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
