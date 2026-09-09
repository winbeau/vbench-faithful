from __future__ import annotations

import json
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from vbench_audit_core.inputs import sha256_file

from .backends.vbench import UPSTREAM_PATH, inspect_upstream, official_diagnostics
from .diagnostics import SubjectConsistencyDiagnostics


def _similarity_matrix(features):
    if getattr(features, "ndim", None) != 2:
        raise ValueError("features must have shape [T, D]")
    if int(features.shape[0]) < 2:
        raise ValueError("Subject Consistency requires at least two frames")
    return (features @ features.transpose(0, 1)).clamp_min(0.0)


def _upper_pairs(similarities):
    import torch

    count = int(similarities.shape[0])
    indices = torch.triu_indices(count, count, offset=1, device=similarities.device)
    return similarities[indices[0], indices[1]]


def local_consistency(features) -> float:
    similarities = _similarity_matrix(features)
    return float(similarities.diagonal(offset=1).mean().item())


def global_pairwise_consistency(features) -> float:
    return float(_upper_pairs(_similarity_matrix(features)).mean().item())


def subject_consistency_score(features) -> float:
    return subject_consistency_diagnostics(features).final_score


def subject_consistency_diagnostics(features) -> SubjectConsistencyDiagnostics:
    similarities = _similarity_matrix(features)
    local_pairs = similarities.diagonal(offset=1)
    global_pairs = _upper_pairs(similarities)
    local_score = float(local_pairs.mean().item())
    global_score = float(global_pairs.mean().item())
    return SubjectConsistencyDiagnostics(
        num_frames=int(features.shape[0]),
        local_score=local_score,
        global_score=global_score,
        final_score=0.5 * local_score + 0.5 * global_score,
        local_pair_scores=tuple(float(value) for value in local_pairs.detach().cpu().tolist()),
        global_pair_min=float(global_pairs.min().item()),
        global_pair_mean=global_score,
        global_pair_max=float(global_pairs.max().item()),
    )


def upstream_path() -> Path:
    return Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()


def build_dino_config(repo: str | None = None, weight: str | None = None) -> dict[str, Any]:
    cache_root = Path(os.environ.get("VBENCH_CACHE_DIR", str(Path.home() / ".cache" / "vbench"))).expanduser()
    repo_path = Path(repo or os.environ.get("VBENCH_AUDIT_DINO_REPO", cache_root / "dino_model/facebookresearch_dino_main")).expanduser()
    weight_path = Path(weight or os.environ.get("VBENCH_AUDIT_DINO_WEIGHT", cache_root / "dino_model/dino_vitbase16_pretrain.pth")).expanduser()
    if not repo_path.is_dir():
        raise ValueError(f"local DINO repository not found: {repo_path}")
    if not weight_path.is_file():
        raise ValueError(f"DINO ViT-B/16 weight not found: {weight_path}")
    return {"repo_or_dir": str(repo_path), "path": str(weight_path), "model": "dino_vitb16", "source": "local", "read_frame": False}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _failed_result(backend: str, video: Path, metadata: Mapping[str, Mapping[str, Any]], exc: Exception) -> dict[str, Any]:
    error = f"{type(exc).__name__}: {exc}"
    item = metadata.get(video.name, {})
    return {"video": str(video), "prompt": item.get("prompt", ""), "backend": backend, "score": None, "status": "failed", "failure_reason": error, "error": error, "diagnostics": None}


def evaluate_batch(backend: str, videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], device: Any, dino_config: Mapping[str, Any], *, extractor: Any | None = None) -> list[dict[str, Any]]:
    from .models import OfficialDinoFeatureExtractor

    extractor = extractor or OfficialDinoFeatureExtractor(device, dino_config, upstream_path())
    results = []
    for video in videos:
        try:
            features = extractor.extract(video)
            if backend == "vbench":
                diagnostics = official_diagnostics(features)
            elif backend == "audit":
                diagnostics = subject_consistency_diagnostics(features)
            else:
                raise ValueError(f"unknown Subject Consistency backend: {backend}")
            item = metadata.get(video.name, {})
            results.append({"video": str(video), "prompt": item.get("prompt", ""), "backend": backend, "score": diagnostics.final_score, "status": "succeeded", "failure_reason": None, "error": None, "diagnostics": diagnostics.to_dict()})
        except Exception as exc:
            results.append(_failed_result(backend, video, metadata, exc))
    return results


def _worker(result_path: str, backend: str, gpu_id: int, videos: list[str], metadata: dict[str, dict[str, Any]], dino_config: dict[str, Any]) -> None:
    paths = [Path(video) for video in videos]
    try:
        import torch
        torch.cuda.set_device(gpu_id)
        results = evaluate_batch(backend, paths, metadata, torch.device(f"cuda:{gpu_id}"), dino_config)
    except Exception as exc:
        results = [_failed_result(backend, video, metadata, exc) for video in paths]
    Path(result_path).write_text(json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8")


def evaluate_backend_sharded(backend: str, videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], gpu_ids: list[int], dino_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from vbench_audit_core.devices import round_robin_shards

    shards = round_robin_shards(videos, gpu_ids)
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    config_dict = dict(dino_config)
    if len(shards) == 1:
        gpu_id, shard = next(iter(shards.items()))
        try:
            import torch
            torch.cuda.set_device(gpu_id)
            return evaluate_batch(backend, list(shard), metadata_dict, torch.device(f"cuda:{gpu_id}"), config_dict)
        except Exception as exc:
            return [_failed_result(backend, video, metadata_dict, exc) for video in shard]

    context = multiprocessing.get_context("spawn")
    collected = []
    with tempfile.TemporaryDirectory(prefix=f"subject-consistency-{backend}-workers-") as root:
        processes = []
        result_paths = []
        for gpu_id, shard in shards.items():
            result_path = Path(root) / f"gpu-{gpu_id}.json"
            process = context.Process(target=_worker, args=(str(result_path), backend, gpu_id, [str(video) for video in shard], metadata_dict, config_dict))
            process.start()
            processes.append(process)
            result_paths.append(result_path)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Subject Consistency worker exited with code {process.exitcode}")
        for result_path in result_paths:
            collected.extend(json.loads(result_path.read_text(encoding="utf-8"))["results"])

    by_video = {item["video"]: item for item in collected}
    missing = [str(video) for video in videos if str(video) not in by_video]
    if missing:
        raise RuntimeError(f"missing Subject Consistency worker results: {missing}")
    return [by_video[str(video)] for video in videos]


def repaired_dataset_score(results: list[dict[str, object]]) -> tuple[float | None, int]:
    scores = [float(item["score"]) for item in results if item.get("status") == "succeeded" and item.get("score") is not None]
    return (sum(scores) / len(scores), len(scores)) if scores else (None, 0)


def environment_record(videos: list[Path], metadata_path: Path | None, dino_config: Mapping[str, Any]) -> dict[str, Any]:
    state = inspect_upstream(upstream_path())
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_sha = None
    weight = Path(str(dino_config["path"]))
    return {
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "code_sha": code_sha,
        "upstream_path": state.path, "upstream_remote": state.remote, "upstream_branch": state.branch,
        "upstream_sha": state.sha, "upstream_dirty": state.dirty,
        "input_sha256": {str(video): sha256_file(video) for video in videos},
        "metadata_sha256": sha256_file(metadata_path) if metadata_path else None,
        "model_name": "DINO ViT-B/16", "model_repo": dino_config["repo_or_dir"], "model_weight": str(weight),
        "model_weight_sha256": sha256_file(weight), "read_frame": False,
    }
