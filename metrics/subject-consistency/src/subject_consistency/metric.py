from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.coordinator import run_spawn_coordinator

from .backends.vbench import inspect_upstream, official_diagnostics
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
    from vbench_audit_core.upstream import resolve_upstream_path

    return resolve_upstream_path()


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


def _worker(result_path: str, gpu_id: int, videos: list[str], backend: str, metadata: dict[str, dict[str, Any]], dino_config: dict[str, Any], seed: int = 42) -> None:
    paths = [Path(video) for video in videos]
    try:
        import torch
        set_seed(seed)
        torch.cuda.set_device(gpu_id)
        results = evaluate_batch(backend, paths, metadata, torch.device(f"cuda:{gpu_id}"), dino_config)
    except Exception as exc:
        results = [_failed_result(backend, video, metadata, exc) for video in paths]
    Path(result_path).write_text(json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8")


def evaluate_backend_sharded(backend: str, videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], gpu_ids: list[int], dino_config: Mapping[str, Any], seed: int = 42) -> list[dict[str, Any]]:
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    config_dict = dict(dino_config)
    return run_spawn_coordinator(
        _worker,
        [str(video) for video in videos],
        gpu_ids,
        worker_args=(backend, metadata_dict, config_dict, seed),
        backend=backend,
        label="subject-consistency",
    ).results


def _masked_worker(result_path: str, gpu_id: int, videos: list[str], metadata: dict[str, dict[str, Any]], dino_config: dict[str, Any], mask_config: dict[str, Any], seed: int = 42) -> None:
    from .models import NpzSubjectMaskProvider

    paths = [Path(video) for video in videos]
    try:
        import torch
        set_seed(seed)
        torch.cuda.set_device(gpu_id)
        provider = NpzSubjectMaskProvider(Path(str(mask_config["root"])))
        results = evaluate_masked_batch(
            paths,
            metadata,
            torch.device(f"cuda:{gpu_id}"),
            dino_config,
            provider,
            phrase_field=str(mask_config["phrase_field"]),
            max_frames=mask_config["max_frames"],
            instance_mode=str(mask_config["instance_mode"]),
            missing_policy=str(mask_config["missing_policy"]),
        )
    except Exception as exc:
        results = [_failed_result("audit-masked", video, metadata, exc) for video in paths]
    Path(result_path).write_text(json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8")


def evaluate_masked_sharded(videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], gpu_ids: list[int], dino_config: Mapping[str, Any], mask_config: Mapping[str, Any], seed: int = 42) -> list[dict[str, Any]]:
    """Subject-localised variant: masks come from a frozen offline directory."""
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    config_dict = dict(dino_config)
    mask_dict = dict(mask_config)
    return run_spawn_coordinator(
        _masked_worker,
        [str(video) for video in videos],
        gpu_ids,
        worker_args=(metadata_dict, config_dict, mask_dict, seed),
        backend="audit-masked",
        label="subject-consistency-masked",
    ).results


def repaired_dataset_score(results: list[dict[str, object]]) -> tuple[float | None, int]:
    scores = [float(item["score"]) for item in results if item.get("status") == "succeeded" and item.get("score") is not None]
    return (sum(scores) / len(scores), len(scores)) if scores else (None, 0)


def subject_phrase(metadata_item: Mapping[str, Any], phrase_field: str, prompt: str = "") -> str:
    """Resolve the phrase the localizer is prompted with.

    The prompt is never parsed here: the phrase must come from an explicit field
    (the official annotations carry ``subject_en``), so this repair cannot
    silently re-invent the string-matching failure mode of the semantic
    dimensions.
    """
    for source in (metadata_item, metadata_item.get("dimension_metadata")):
        if isinstance(source, Mapping):
            value = source.get(phrase_field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"metadata is missing a usable '{phrase_field}' subject phrase (prompt={prompt[:60]!r})")


def evaluate_masked_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    dino_config: Mapping[str, Any],
    mask_provider: Any,
    *,
    extractor: Any | None = None,
    phrase_field: str = "subject_en",
    max_frames: int | None = None,
    instance_mode: str = "union",
    missing_policy: str = "zero",
) -> list[dict[str, Any]]:
    """Score subject consistency on subject-localised evidence.

    The decoded frames are loaded once and shared by the localizer and the
    feature extractor, so masks and patch tokens cannot come from two different
    decodes of the same clip.
    """
    from .models import OfficialDinoPatchExtractor
    from .subject_evidence import masked_subject_consistency, resample_masks_to_grid, sample_frame_indices

    import torch

    extractor = extractor or OfficialDinoPatchExtractor(device, dino_config, upstream_path())
    results = []
    for video in videos:
        item = metadata.get(video.name, {})
        prompt = str(item.get("prompt", ""))
        try:
            phrase = subject_phrase(item, phrase_field, prompt)
            frames = extractor.module.load_video(str(video))
            patches, grid = extractor.patches_from_frames(frames)
            masks = mask_provider.masks_for(video, frames, phrase)
            instance_masks, present = masks.instance_masks, masks.instance_present
            if int(instance_masks.shape[0]) != int(patches.shape[0]):
                raise ValueError(
                    f"{instance_masks.shape[0]} mask frames for {patches.shape[0]} feature frames: {video.name}"
                )
            frames_count, instances = int(instance_masks.shape[0]), int(instance_masks.shape[1])
            grid_masks = resample_masks_to_grid(
                instance_masks.reshape(frames_count * instances, *instance_masks.shape[2:]), grid[0], grid[1],
                image_size=getattr(extractor, "transformed_size", None),
            ).reshape(frames_count, instances, -1).to(patches.device)
            present = present.to(patches.device)
            indices = sample_frame_indices(frames_count, max_frames)
            if len(indices) != frames_count:
                index_tensor = torch.as_tensor(list(indices), device=patches.device)
                patches = patches[index_tensor]
                grid_masks = grid_masks[index_tensor]
                present = present[index_tensor]
            diagnostics = masked_subject_consistency(
                patches, grid_masks, present, mode=instance_mode, missing_policy=missing_policy
            )
            payload = diagnostics.to_dict()
            payload.update(
                {
                    "subject_phrase": phrase,
                    "phrase_field": phrase_field,
                    "patch_grid": [int(grid[0]), int(grid[1])],
                    "max_frames": max_frames,
                    "mask_source": masks.source,
                    "mask_source_sha256": masks.source_sha256,
                    "extractor": type(extractor).__name__,
                }
            )
            results.append(
                {
                    "video": str(video),
                    "prompt": prompt,
                    "backend": "audit-masked",
                    "score": diagnostics.score,
                    "status": "succeeded",
                    "failure_reason": None,
                    "error": None,
                    "diagnostics": payload,
                }
            )
        except Exception as exc:
            results.append(_failed_result("audit-masked", video, metadata, exc))
    return results


def environment_record(videos: list[Path], metadata_path: Path | None, dino_config: Mapping[str, Any]) -> dict[str, Any]:
    state = inspect_upstream(upstream_path())
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        code_sha = None
    weight = Path(str(dino_config["path"]))
    return {
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "code_sha": code_sha,
        "upstream_path": state.path, "upstream_remote": state.remote, "upstream_source_type": state.source_type,
        "upstream_branch": state.branch,
        "upstream_sha": state.sha, "upstream_dirty": state.dirty,
        "input_sha256": {str(video): sha256_file(video) for video in videos},
        "metadata_sha256": sha256_file(metadata_path) if metadata_path else None,
        "model_name": "DINO ViT-B/16", "model_repo": dino_config["repo_or_dir"], "model_weight": str(weight),
        "model_weight_sha256": sha256_file(weight), "read_frame": False,
    }
