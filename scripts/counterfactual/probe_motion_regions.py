"""DEV-only automatic regions on all frames of the fixed reviewed cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.trajectory import TrajectoryConfig, decode_video
from .review_selection import select_reviewed_candidates
from .score_static_jitter import source_hashes, command_output
from .static_jitter import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "config", "feature-run", "sam-root", "sam-weight", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-gib", type=float, default=24.)
    args = parser.parse_args(argv)
    import torch
    from vbench_audit_models.sam_regions import SamRegionModel

    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[2]
    manifest, review, feature_root = Path(args.manifest), Path(args.review), Path(args.feature_run)
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()], review, manifest, root)
    config = json.loads(Path(args.config).read_text())
    if config["sam_architecture"] != "vit_h" or config["post_selection"] != "none":
        raise ValueError("this diagnostic requires ViT-H with no extra proposal selection")
    feature_identity = json.loads((feature_root / "provenance.json").read_text())
    feature_runtime = json.loads((feature_root / "runtime.json").read_text())
    if (feature_runtime["status"] != "finished" or feature_runtime["failed"]
            or feature_identity["manifest_sha256"] != digest(manifest)
            or feature_identity["review_sha256"] != digest(review)
            or set(feature_identity["cache_sha256"]) != {r["candidate_id"] for r in rows}
            or feature_identity["config"]["sample_fps"] != config["sample_fps"]
            or feature_identity["config"]["decode_max_side"] != config["decode_max_side"]):
        raise ValueError("completed bound feature run with identical cohort/sampling required")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fresh output required; no overwrite or implicit resume")
    if args.device.startswith("cuda"):
        index = torch.device(args.device).index or 0
        free, _ = torch.cuda.mem_get_info(index)
        if free < args.minimum_free_gib * 1024 ** 3:
            raise RuntimeError("not enough spare GPU memory; no model started")
        torch.cuda.set_per_process_memory_fraction(.15, index)
    model = SamRegionModel(args.sam_root, args.sam_weight, args.device, config["generator"])
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    identity = {"manifest_sha256": digest(manifest), "review_sha256": digest(review),
                "config": config, "config_sha256": digest(Path(args.config)),
                "model_identity": model.identity,
                "sam_revision": command_output(["git", "-C", args.sam_root, "rev-parse", "HEAD"]),
                "sam_dirty": command_output(["git", "-C", args.sam_root, "status", "--porcelain"]),
                "feature_source": str(feature_root),
                "feature_provenance_sha256": digest(feature_root / "provenance.json"),
                "feature_cache_sha256": feature_identity["cache_sha256"],
                "code_files": source_hashes(root), "script_sha256": digest(Path(__file__)),
                "input_sha256": {}, "cache_sha256": {},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "status": "diagnostic_only", "formal_acceptance": "NOT EVALUATED"}
    runtime = {"status": "running", "pid": os.getpid(), "expected": len(rows), "completed": 0,
               "failed": 0, "frames_completed": 0,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()
    (output / "provenance.json").write_text(json.dumps(identity, indent=2))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    with (output / "diagnostics.jsonl").open("x") as handle:
        for row in rows:
            key, video = row["candidate_id"], Path(row["video"])
            record = {k: row[k] for k in ("candidate_id", "base_id", "prompt_id", "family", "seed")}
            record.update(status="diagnostic_only", score=None)
            try:
                if digest(video) != row["sha256"]:
                    raise ValueError("immutable video identity changed")
                frames, timestamps, sampled = decode_video(video, TrajectoryConfig(
                    sample_fps=config["sample_fps"], max_side=config["decode_max_side"]))
                feature_path = feature_root / "evidence" / f"{key}.npz"
                if digest(feature_path) != feature_identity["cache_sha256"][key]:
                    raise ValueError("feature cache identity changed")
                with np.load(feature_path, allow_pickle=False) as f:
                    if not np.array_equal(f["timestamps"], timestamps) or not np.array_equal(f["input_shape"], frames.shape):
                        raise ValueError("RGB/descriptor timeline or geometry mismatch")
                summaries, arrays = [], {"timestamps": timestamps, "input_shape": np.array(frames.shape)}
                for i, frame in enumerate(frames):
                    proposals = model.propose(frame)
                    arrays.update({f"frame_{i}_" + k: v for k, v in proposals.items()})
                    summaries.append({"frame": i, "count": len(proposals["areas"]),
                                      "union_fraction": float(proposals["union_fraction"]),
                                      "areas": proposals["areas"].tolist()})
                    runtime.update(current_candidate=key, current_frame=i,
                                   frames_completed=runtime["frames_completed"] + 1)
                    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
                    print(json.dumps({"candidate_id": key, "frame": i, "regions": len(proposals["areas"])}), flush=True)
                path = output / "evidence" / f"{key}.npz"
                np.savez_compressed(path, **arrays)
                identity["input_sha256"][key] = row["sha256"]
                identity["cache_sha256"][key] = digest(path)
                record.update(sampling=sampled, frames=summaries,
                              mean_union_fraction=float(np.mean([s["union_fraction"] for s in summaries])),
                              empty_frames=sum(s["count"] == 0 for s in summaries))
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                runtime["failed"] += 1
            handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
            runtime["completed"] += 1
            (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
            (output / "provenance.json").write_text(json.dumps(identity, indent=2))
            print(json.dumps({"candidate_id": key, "status": record["status"], "error": record.get("error")}), flush=True)
    runtime.update(status="finished", elapsed_seconds=time.monotonic() - started,
                   finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if args.device.startswith("cuda"):
        runtime.update(gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    return 1 if runtime["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
