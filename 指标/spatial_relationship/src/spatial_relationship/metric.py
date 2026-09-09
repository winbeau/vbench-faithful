from __future__ import annotations

import multiprocessing
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from vbench_audit_core.inputs import sha256_file

from .backends.audit import result_payload, score_predictions
from .backends.vbench import DEFAULT_WEIGHT, UPSTREAM_PATH, OfficialGritDetector, OfficialVBenchEvaluator, inspect_upstream, normalize_official_results
from .diagnostics import DiagnosticsLevel
from .relation import normalize_relation
from .schemas import OrderedRelationQuery


def parse_query(metadata_item: Mapping[str, Any]) -> OrderedRelationQuery:
    dimension = metadata_item.get("dimension_metadata")
    if not isinstance(dimension, Mapping):
        raise ValueError("metadata item must contain dimension_metadata")
    # Accept only explicit structures; nested official structure is plumbing,
    # never a filename/prompt inference fallback.
    while "spatial_relationship" in dimension and isinstance(dimension["spatial_relationship"], Mapping):
        dimension = dimension["spatial_relationship"]
    missing = [key for key in ("object_a", "object_b", "relationship") if not isinstance(dimension.get(key), str) or not dimension[key].strip()]
    if missing:
        raise ValueError(f"spatial dimension_metadata missing fields: {', '.join(missing)}")
    return OrderedRelationQuery(
        subject=dimension["object_a"].strip(),
        relation=normalize_relation(dimension["relationship"]),
        object=dimension["object_b"].strip(),
    )


def weight_path() -> Path:
    return Path(os.environ.get("VBENCH_AUDIT_GRIT_WEIGHT", str(DEFAULT_WEIGHT))).expanduser()


def upstream_path() -> Path:
    return Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()


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


def evaluate_audit_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    diagnostics_level: DiagnosticsLevel,
    *,
    detector: Any | None = None,
) -> list[dict[str, Any]]:
    detector = detector or OfficialGritDetector(device, model_weight, upstream_path())
    results = []
    for video in videos:
        meta = metadata[video.name]
        query = parse_query(meta)
        try:
            frame_indices, predictions = detector.detect_video(video)
            scored = score_predictions(str(video), str(meta.get("prompt", "")), query, predictions, sampled_frame_indices=frame_indices)
            payload = result_payload(scored, diagnostics_level)
            results.append({"video": str(video), "score": scored.video_score, "status": "succeeded", "error": None, **payload})
        except Exception as exc:
            results.append(
                {
                    "video": str(video), "prompt": str(meta.get("prompt", "")),
                    "subject": query.subject, "relation": query.relation, "object": query.object,
                    "backend": "audit", "score": None, "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}", "error": f"{type(exc).__name__}: {exc}",
                    "diagnostics": None,
                }
            )
    return results


def evaluate_vbench_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    *,
    evaluator: Any | None = None,
) -> list[dict[str, Any]]:
    evaluator = evaluator or OfficialVBenchEvaluator(device, model_weight, upstream_path())
    results: list[dict[str, Any]] = []
    for video in videos:
        meta = metadata[video.name]
        query = parse_query(meta)
        try:
            raw = evaluator.evaluate_video(video, meta, query)
            results.extend(normalize_official_results(raw, metadata, parse_query))
        except Exception as exc:
            results.extend(_failed_results("vbench", [video], metadata, exc))
    return results


def _failed_results(backend: str, videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], exc: Exception) -> list[dict[str, Any]]:
    error = f"{type(exc).__name__}: {exc}"
    results = []
    for video in videos:
        meta = metadata[video.name]
        query = parse_query(meta)
        results.append(
            {
                "video": str(video), "prompt": str(meta.get("prompt", "")),
                "subject": query.subject, "relation": query.relation, "object": query.object,
                "backend": backend, "score": None, "status": "failed",
                "failure_reason": error, "error": error, "diagnostics": None,
            }
        )
    return results


def _worker(
    result_path: str,
    backend: str,
    gpu_id: int,
    videos: list[str],
    metadata: dict[str, dict[str, Any]],
    model_weight: str,
    diagnostics_level: str,
) -> None:
    paths = [Path(video) for video in videos]
    try:
        import torch
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        if backend == "vbench":
            results = evaluate_vbench_batch(paths, metadata, device, Path(model_weight))
        else:
            results = evaluate_audit_batch(paths, metadata, device, Path(model_weight), DiagnosticsLevel(diagnostics_level))
    except Exception as exc:
        results = _failed_results(backend, paths, metadata, exc)
    Path(result_path).write_text(json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8")


def evaluate_backend_sharded(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    gpu_ids: list[int],
    model_weight: Path,
    diagnostics_level: DiagnosticsLevel,
) -> list[dict[str, Any]]:
    from vbench_audit_core.devices import round_robin_shards

    shards = round_robin_shards(videos, gpu_ids)
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    if len(shards) == 1:
        gpu_id, shard = next(iter(shards.items()))
        try:
            import torch
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            if backend == "vbench":
                return evaluate_vbench_batch(list(shard), metadata_dict, device, model_weight)
            return evaluate_audit_batch(list(shard), metadata_dict, device, model_weight, diagnostics_level)
        except Exception as exc:
            return _failed_results(backend, list(shard), metadata_dict, exc)

    context = multiprocessing.get_context("spawn")
    collected: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"spatial-{backend}-workers-") as worker_root:
        processes = []
        result_paths: list[Path] = []
        for gpu_id, shard in shards.items():
            result_path = Path(worker_root) / f"gpu-{gpu_id}.json"
            process = context.Process(
                target=_worker,
                args=(str(result_path), backend, gpu_id, [str(video) for video in shard], metadata_dict, str(model_weight), diagnostics_level.value),
            )
            process.start()
            processes.append(process)
            result_paths.append(result_path)
        try:
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"spatial worker exited with code {process.exitcode}")
            for result_path in result_paths:
                if not result_path.is_file():
                    raise RuntimeError(f"spatial worker did not write result: {result_path.name}")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                collected.extend(payload["results"])
        except KeyboardInterrupt:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise

    expected = [str(video) for video in videos]
    by_video: dict[str, dict[str, Any]] = {}
    for result in collected:
        video = result["video"]
        if video in by_video:
            raise RuntimeError(f"duplicate worker result: {video}")
        by_video[video] = result
    missing = [video for video in expected if video not in by_video]
    if missing:
        raise RuntimeError(f"missing worker results: {missing}")
    return [by_video[video] for video in expected]


def environment_record(videos: list[Path], metadata_path: Path, model_weight: Path) -> dict[str, Any]:
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        code_sha = None
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        uv_version = None
    try:
        driver_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        driver_version = None
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = None
        cuda_version = None
    upstream = inspect_upstream(upstream_path())
    workspace_root = Path(__file__).resolve().parents[4]
    lock_path = workspace_root / "uv.lock"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "uv": uv_version,
        "torch": torch_version,
        "torch_cuda": cuda_version,
        "nvidia_driver": driver_version,
        "code_sha": code_sha,
        "code_dirty": dirty,
        "upstream_path": upstream.path,
        "upstream_remote": upstream.remote,
        "upstream_branch": upstream.branch,
        "upstream_sha": upstream.sha,
        "upstream_dirty": upstream.dirty,
        "upstream_submodules": list(upstream.submodules),
        "input_sha256": {str(video): sha256_file(video) for video in videos},
        "metadata_sha256": sha256_file(metadata_path),
        "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "model_name": "GRiT ObjectDet",
        "model_source": "locked upstream vbench/third_party/grit_model.py",
        "model_weight": str(model_weight),
        "model_weight_sha256": sha256_file(model_weight) if model_weight.is_file() else None,
    }
