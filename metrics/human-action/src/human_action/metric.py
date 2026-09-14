from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.coordinator import run_spawn_coordinator

from .diagnostics import DiagnosticsLevel, audit_result_payload, official_diagnostics


def upstream_path() -> Path:
    from vbench_audit_core.upstream import resolve_upstream_path

    return resolve_upstream_path()


def weight_path() -> Path:
    from .backends.vbench import DEFAULT_WEIGHT

    return Path(os.environ.get("VBENCH_AUDIT_UMT_WEIGHT", str(DEFAULT_WEIGHT))).expanduser()


def load_categories(path: Path) -> tuple[str, ...]:
    category_path = path / "vbench/third_party/umt/kinetics_400_categories.txt"
    indexed: dict[int, str] = {}
    for line in category_path.read_text(encoding="utf-8").splitlines():
        action, index = line.rsplit("\t", 1)
        indexed[int(index)] = action.strip().lower()
    if set(indexed) != set(range(400)):
        raise ValueError("locked Kinetics category file must contain indices 0..399 exactly once")
    return tuple(indexed[index] for index in range(400))


def set_seed(seed: int) -> None:
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


def evaluate_vbench_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    categories: tuple[str, ...],
    *,
    evaluator: Any | None = None,
) -> list[dict[str, Any]]:
    from .backends.vbench import OfficialHumanActionEvaluator

    evaluator = evaluator or OfficialHumanActionEvaluator(device, model_weight, upstream_path())
    results = []
    for video in videos:
        # Official VBench semantics derive the Kinetics target from the
        # filename; metadata is intentionally not consulted here.
        prompt = str(metadata.get(video.name, {}).get("prompt", ""))
        try:
            official = evaluator.evaluate_video(video)
            results.append(
                {
                    "video": str(video),
                    "prompt": prompt,
                    "target_action": official.target_action,
                    "target_source": official.target_source,
                    "backend": "vbench",
                    "score": float(official.matched),
                    "status": "succeeded",
                    "failure_reason": None,
                    "error": None,
                    "official_video_boolean": official.matched,
                    "diagnostics": official_diagnostics(official),
                }
            )
        except Exception as exc:
            results.extend(_failed_results("vbench", [video], metadata, categories, exc))
    return results


def evaluate_audit_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    categories: tuple[str, ...],
    diagnostics_level: DiagnosticsLevel,
    *,
    evaluator: Any | None = None,
) -> list[dict[str, Any]]:
    from .backends.audit import AuditHumanActionEvaluator, parse_action_query
    from .backends.vbench import import_official_module
    from .models import LockedUmtClassifier

    if evaluator is None:
        module, _ = import_official_module(upstream_path())
        classifier = LockedUmtClassifier(device, model_weight, module)
        evaluator = AuditHumanActionEvaluator(classifier)
    results = []
    for video in videos:
        query = parse_action_query(metadata[video.name], categories)
        try:
            result = evaluator.evaluate_video(video, query)
            results.append(audit_result_payload(result, diagnostics_level))
        except Exception as exc:
            results.extend(_failed_results("audit", [video], metadata, categories, exc))
    return results


def _failed_results(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    categories: tuple[str, ...],
    exc: Exception,
) -> list[dict[str, Any]]:
    error = f"{type(exc).__name__}: {exc}"
    return [
        {
            "video": str(video),
            "prompt": str(metadata.get(video.name, {}).get("prompt", "")),
            "target_action": None,
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
    gpu_id: int,
    videos: list[str],
    backend: str,
    metadata: dict[str, dict[str, Any]],
    model_weight: str,
    categories: tuple[str, ...],
    diagnostics_level: str,
    seed: int = 42,
) -> None:
    paths = [Path(video) for video in videos]
    try:
        import torch

        set_seed(seed)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        if backend == "vbench":
            results = evaluate_vbench_batch(paths, metadata, device, Path(model_weight), categories)
        else:
            results = evaluate_audit_batch(
                paths,
                metadata,
                device,
                Path(model_weight),
                categories,
                DiagnosticsLevel(diagnostics_level),
            )
    except Exception as exc:
        results = _failed_results(backend, paths, metadata, categories, exc)
    Path(result_path).write_text(
        json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False),
        encoding="utf-8",
    )


def evaluate_backend_sharded(
    backend: str,
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    gpu_ids: list[int],
    model_weight: Path,
    categories: tuple[str, ...],
    diagnostics_level: DiagnosticsLevel,
    seed: int = 42,
) -> list[dict[str, Any]]:
    metadata_dict = {key: dict(value) for key, value in metadata.items()}
    coordinated = run_spawn_coordinator(
        _worker,
        [str(video) for video in videos],
        gpu_ids,
        worker_args=(backend, metadata_dict, str(model_weight), categories, diagnostics_level.value, seed),
        backend=backend,
        label="human-action",
    )
    return coordinated.results


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

    try:
        import torch

        torch_version, cuda_version = torch.__version__, torch.version.cuda
    except ImportError:
        torch_version = cuda_version = None
    driver = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
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
        "code_sha": command_output(["git", "rev-parse", "HEAD"]),
        "code_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "upstream_path": upstream.path,
        "upstream_remote": upstream.remote,
        "upstream_branch": upstream.branch,
        "upstream_sha": upstream.sha,
        "upstream_dirty": upstream.dirty,
        "upstream_submodules": list(upstream.submodules),
        "input_sha256": {str(video): sha256_file(video) for video in videos},
        "metadata_sha256": sha256_file(metadata_path) if metadata_path is not None and metadata_path.is_file() else None,
        "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "model_name": "UMT ViT-L/16 Kinetics-400 f16 res224",
        "model_source": "locked upstream vbench/third_party/umt",
        "model_weight": str(model_weight),
        "model_weight_sha256": sha256_file(model_weight) if model_weight.is_file() else None,
    }
