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

from .diagnostics import DiagnosticsLevel


def upstream_path() -> Path:
    from .backends.vbench import UPSTREAM_PATH

    return Path(os.environ.get("VBENCH_AUDIT_UPSTREAM", str(UPSTREAM_PATH))).expanduser()


def weight_path() -> Path:
    from .backends.vbench import DEFAULT_WEIGHT

    return Path(os.environ.get("VBENCH_AUDIT_RAFT_WEIGHT", str(DEFAULT_WEIGHT))).expanduser()


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


def metadata_for_videos(
    videos: list[Path], metadata: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    source = metadata or {}
    return {
        video.name: dict(source.get(video.name, {"video": video.name, "prompt": ""}))
        for video in videos
    }


def prompt_and_override(metadata_item: Mapping[str, Any]) -> tuple[str, str | None]:
    prompt = metadata_item.get("prompt", "")
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise ValueError("metadata prompt must be a string")
    override = metadata_item.get("motion_target")
    dimension = metadata_item.get("dimension_metadata")
    if isinstance(dimension, Mapping):
        while "dynamic_degree" in dimension and isinstance(dimension["dynamic_degree"], Mapping):
            dimension = dimension["dynamic_degree"]
        override = dimension.get("motion_target", override)
    if override is not None and not isinstance(override, str):
        raise ValueError("motion_target override must be a string")
    return prompt, override


def evaluate_vbench_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    *,
    evaluator: Any | None = None,
) -> list[dict[str, Any]]:
    from .backends.vbench import OfficialDynamicEvaluator, official_result_payload

    evaluator = evaluator or OfficialDynamicEvaluator(device, model_weight, upstream_path())
    results = []
    for video in videos:
        prompt, _ = prompt_and_override(metadata[video.name])
        try:
            official = evaluator.evaluate_video(video)
            results.append(
                {
                    "video": str(video),
                    "prompt": prompt,
                    "backend": "vbench",
                    "score": float(official.official_video_boolean),
                    "status": "succeeded",
                    "failure_reason": None,
                    "error": None,
                    "official_video_boolean": official.official_video_boolean,
                    "diagnostics": official_result_payload(official),
                }
            )
        except Exception as exc:
            results.extend(_failed_results("vbench", [video], metadata, exc))
    return results


def evaluate_audit_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    diagnostics_level: DiagnosticsLevel,
    *,
    flow_model: Any | None = None,
    decoder: Any | None = None,
    config: Any | None = None,
) -> list[dict[str, Any]]:
    from .backends.audit import analyze_timed_flow_sequence, audit_result_payload
    from .models import RaftFlowModel, decode_timed_frames

    flow_model = flow_model or RaftFlowModel(device, model_weight, upstream_path())
    decoder = decoder or decode_timed_frames
    results = []
    for video in videos:
        prompt, override = prompt_and_override(metadata[video.name])
        try:
            sequence = decoder(video)
            audit = analyze_timed_flow_sequence(
                str(video), prompt, sequence, flow_model, motion_target_override=override, config=config
            )
            results.append(audit_result_payload(audit, diagnostics_level))
        except Exception as exc:
            results.extend(_failed_results("audit", [video], metadata, exc))
    return results


def _failed_results(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    exc: Exception,
) -> list[dict[str, Any]]:
    error = f"{type(exc).__name__}: {exc}"
    return [
        {
            "video": str(video),
            "prompt": prompt_and_override(metadata[video.name])[0],
            "backend": backend,
            "score": None,
            "status": "failed",
            "failure_reason": error,
            "error": error,
            "diagnostics": None,
        }
        for video in videos
    ]


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
            results = evaluate_audit_batch(
                paths, metadata, device, Path(model_weight), DiagnosticsLevel(diagnostics_level)
            )
    except Exception as exc:
        results = _failed_results(backend, paths, metadata, exc)
    Path(result_path).write_text(
        json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8"
    )


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
            return evaluate_audit_batch(
                list(shard), metadata_dict, device, model_weight, diagnostics_level
            )
        except Exception as exc:
            return _failed_results(backend, list(shard), metadata_dict, exc)

    context = multiprocessing.get_context("spawn")
    collected: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"dynamic-{backend}-workers-") as worker_root:
        processes = []
        result_paths: list[Path] = []
        for gpu_id, shard in shards.items():
            result_path = Path(worker_root) / f"gpu-{gpu_id}.json"
            process = context.Process(
                target=_worker,
                args=(
                    str(result_path), backend, gpu_id, [str(video) for video in shard], metadata_dict,
                    str(model_weight), diagnostics_level.value,
                ),
            )
            process.start()
            processes.append(process)
            result_paths.append(result_path)
        try:
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"dynamic worker exited with code {process.exitcode}")
            for result_path in result_paths:
                if not result_path.is_file():
                    raise RuntimeError(f"dynamic worker did not write result: {result_path.name}")
                collected.extend(json.loads(result_path.read_text(encoding="utf-8"))["results"])
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


def release_cuda_resources() -> None:
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def environment_record(videos: list[Path], metadata_path: Path | None, model_weight: Path) -> dict[str, Any]:
    from .backends.vbench import inspect_upstream

    def command_output(command: list[str]) -> str | None:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    code_sha = command_output(["git", "rev-parse", "HEAD"])
    code_status = command_output(["git", "status", "--porcelain"])
    driver = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
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
        "uv": command_output(["uv", "--version"]),
        "torch": torch_version,
        "torch_cuda": cuda_version,
        "nvidia_driver": driver.splitlines() if driver else None,
        "code_sha": code_sha,
        "code_dirty": bool(code_status) if code_status is not None else None,
        "upstream_path": upstream.path,
        "upstream_remote": upstream.remote,
        "upstream_branch": upstream.branch,
        "upstream_sha": upstream.sha,
        "upstream_dirty": upstream.dirty,
        "upstream_submodules": list(upstream.submodules),
        "input_sha256": {str(video): sha256_file(video) for video in videos},
        "metadata_sha256": sha256_file(metadata_path) if metadata_path is not None else None,
        "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "model_name": "RAFT Things",
        "model_source": "locked upstream vbench/third_party/RAFT",
        "model_weight": str(model_weight),
        "model_weight_sha256": sha256_file(model_weight) if model_weight.is_file() else None,
    }
