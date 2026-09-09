from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from vbench_audit_core.inputs import sha256_file

from .backends.audit import score_frame, to_diagnostic
from .backends.vbench import (
    OfficialVBenchSceneEvaluator,
    extract_scene_label,
    upstream_path,
    verify_upstream,
    video_score,
)
from .diagnostics import diagnostics_payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def scene_label(metadata_item: Mapping[str, Any]) -> str:
    return extract_scene_label(metadata_item)


def decode_video(video: Path, num_frames: int = 16) -> tuple[list[int], list[Any]]:
    """Use the locked VBench Decord path and return source indices plus PIL RGB frames."""
    import importlib

    locked_upstream = upstream_path()
    verify_upstream(locked_upstream)
    resolved = str(locked_upstream.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.utils")
    decord = importlib.import_module("decord")
    decord.bridge.set_bridge("native")
    reader = module.VideoReader(str(video), width=384, height=384, num_threads=1)
    indices = module.get_frame_indices(num_frames, len(reader), sample="middle")
    arrays = reader.get_batch(indices).asnumpy()
    frames = [module.Image.fromarray(array).convert("RGB") for array in arrays]
    return list(indices), frames


def evaluate_repaired_video(
    video: Path,
    metadata_item: Mapping[str, Any],
    scorer: Any,
    mode: str,
    *,
    frames: list[Any] | None = None,
    sampled_frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    label = scene_label(metadata_item)
    if frames is None:
        sampled_frame_indices, frames = decode_video(video)
    indices = sampled_frame_indices or list(range(len(frames)))
    if len(indices) != len(frames):
        raise ValueError("sampled_frame_indices must align with frames")
    evidence = [score_frame(frame, label, scorer, mode=mode, frame_index=index) for index, frame in zip(indices, frames)]
    diagnostics = [to_diagnostic(item, label, mode) for item in evidence]
    return {
        "video_path": str(video),
        "video_results": video_score([item.final_frame_score for item in evidence]),
        "frame_results": [item.final_frame_score for item in evidence],
        "diagnostics": diagnostics_payload(diagnostics, indices),
    }


def evaluate_global_or_environment_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    scorer: Any,
    mode: str,
) -> list[dict[str, Any]]:
    results = []
    for video in videos:
        item = metadata[video.name]
        try:
            raw = evaluate_repaired_video(video, item, scorer, mode)
            results.append(
                {
                    "video": str(video),
                    "prompt": item.get("prompt", ""),
                    "scene": scene_label(item),
                    "backend": mode,
                    "score": raw["video_results"],
                    "status": "succeeded",
                    "failure_reason": None,
                    "error": None,
                    "diagnostics": raw["diagnostics"],
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "video": str(video),
                    "prompt": item.get("prompt", ""),
                    "scene": scene_label(item),
                    "backend": mode,
                    "score": None,
                    "status": "failed",
                    "failure_reason": error,
                    "error": error,
                    "diagnostics": None,
                }
            )
    return results


def evaluate_official_batch(
    videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], device: Any, *, evaluator: Any | None = None
) -> list[dict[str, Any]]:
    evaluator = evaluator or OfficialVBenchSceneEvaluator(device)
    results = []
    for video in videos:
        item = metadata[video.name]
        try:
            raw = evaluator.evaluate_video(video, item)
            results.append(
                {
                    "video": str(video),
                    "prompt": item.get("prompt", ""),
                    "scene": scene_label(item),
                    "backend": "official",
                    "score": float(raw["video_results"]),
                    "status": "succeeded",
                    "failure_reason": None,
                    "error": None,
                    "diagnostics": {
                        "frame_results": raw.get("frame_results"),
                        "success_frame_count": raw.get("success_frame_count"),
                        "frame_count": raw.get("frame_count"),
                    },
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "video": str(video),
                    "prompt": item.get("prompt", ""),
                    "scene": scene_label(item),
                    "backend": "official",
                    "score": None,
                    "status": "failed",
                    "failure_reason": error,
                    "error": error,
                    "diagnostics": None,
                }
            )
    return results


def evaluate_backend_sharded(
    mode: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    gpu_ids: list[int],
    *,
    scorer_factory: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate deterministic round-robin shards and restore input order.

    The official evaluator retains VBench's own distributed gather when run in
    a distributed process. This lightweight shard coordinator provides the
    same global video-result contract for the audit CLI without changing the
    metric or output schemas.
    """
    from vbench_audit_core.devices import round_robin_shards

    shards = round_robin_shards(videos, gpu_ids)
    collected: list[dict[str, Any]] = []
    for gpu_id, shard in shards.items():
        device = f"cuda:{gpu_id}"
        if mode == "official":
            collected.extend(evaluate_official_batch(list(shard), metadata, device))
        else:
            scorer = scorer_factory(device) if scorer_factory else None
            if scorer is None:
                raise ValueError("a scorer_factory is required for repaired modes")
            collected.extend(evaluate_global_or_environment_batch(list(shard), metadata, scorer, mode))
    by_video = {item["video"]: item for item in collected}
    return [by_video[str(video)] for video in videos]


def environment_record(videos: list[Path], metadata_path: Path | None, model_name: str) -> dict[str, Any]:
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_sha = None
    return {
        "input_sha256": {video.name: sha256_file(video) for video in videos if video.is_file()},
        "metadata_sha256": sha256_file(metadata_path) if metadata_path and metadata_path.is_file() else None,
        "model": model_name,
        "upstream_path": str(upstream_path()),
        "code_sha": code_sha,
    }
