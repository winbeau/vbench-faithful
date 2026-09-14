from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..schemas import OfficialVideoResult

UPSTREAM_PATH = Path(os.environ.get("VBENCH1_ROOT", "/root/vbench1"))
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"
DEFAULT_WEIGHT = Path.home() / ".cache/vbench/raft_model/models/raft-things.pth"


@dataclass(frozen=True)
class UpstreamState:
    path: str
    remote: str
    branch: str
    sha: str
    dirty: bool
    submodules: tuple[str, ...]


def inspect_upstream(path: Path = UPSTREAM_PATH) -> UpstreamState:
    path = Path(path)
    source = path / "vbench/dynamic_degree.py"
    if not path.is_dir():
        raise FileNotFoundError(f"official upstream root does not exist: {path}")
    if not source.is_file():
        raise FileNotFoundError(f"official Dynamic Degree source is missing: {source}")
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()

    return UpstreamState(
        path=str(path.resolve()),
        remote=git("remote", "get-url", "origin"),
        branch=git("branch", "--show-current"),
        sha=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        submodules=tuple(line for line in git("submodule", "status").splitlines() if line),
    )


def verify_upstream(path: Path = UPSTREAM_PATH) -> UpstreamState:
    state = inspect_upstream(path)
    if state.sha != UPSTREAM_SHA:
        raise RuntimeError(f"upstream SHA mismatch: expected={UPSTREAM_SHA}, actual={state.sha}")
    if state.dirty:
        raise RuntimeError("upstream worktree is dirty; refusing Dynamic Degree evaluation")
    return state


def import_official_module(path: Path = UPSTREAM_PATH):
    state = verify_upstream(path)
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.dynamic_degree")
    module_path = Path(module.__file__).resolve()
    if path.resolve() not in module_path.parents:
        raise RuntimeError(f"wrong vbench import: {module_path}")
    return module, state


def official_top5_mean(flow: np.ndarray) -> float:
    array = np.asarray(flow)
    magnitude = np.sqrt(np.square(array[..., 0]) + np.square(array[..., 1]))
    height, width = magnitude.shape
    cut_index = int(height * width * 0.05)
    return float(np.mean(abs(np.sort(-magnitude.reshape(-1)))[:cut_index]))


def official_parameters(frame_shape: tuple[int, ...], sampled_frame_count: int) -> tuple[float, int]:
    scale = min(frame_shape[-2:])
    return 6.0 * (scale / 256.0), round(4 * (sampled_frame_count / 16.0))


def official_check_move(scores: list[float], threshold: float, count_num: int) -> bool:
    count = 0
    for score in scores:
        if score > threshold:
            count += 1
        if count >= count_num:
            return True
    return False


class OfficialDynamicEvaluator:
    """One-model adapter that reproduces the locked DynamicDegree.infer path."""

    def __init__(self, device: Any, model_weight: Path, upstream_path: Path = UPSTREAM_PATH):
        if not model_weight.is_file():
            raise FileNotFoundError(f"official RAFT weight not found: {model_weight}")
        self.module, self.upstream_state = import_official_module(upstream_path)
        args = self.module.edict({"model": str(model_weight), "small": False, "mixed_precision": False, "alternate_corr": False})
        self.dynamic = self.module.DynamicDegree(args, device)
        self.device = device

    def evaluate_video(self, video: Path) -> OfficialVideoResult:
        torch = self.module.torch
        with torch.no_grad():
            frames = self.dynamic.get_frames(str(video))
            self.dynamic.set_params(frame=frames[0], count=len(frames))
            scores: list[float] = []
            for image_a, image_b in zip(frames[:-1], frames[1:]):
                padder = self.module.InputPadder(image_a.shape)
                padded_a, padded_b = padder.pad(image_a, image_b)
                _, flow_up = self.dynamic.model(padded_a, padded_b, iters=20, test_mode=True)
                scores.append(self.dynamic.get_score(padded_a, flow_up))
            result = self.dynamic.check_move(scores)
        capture = self.module.cv2.VideoCapture(str(video))
        source_fps = float(capture.get(self.module.cv2.CAP_PROP_FPS))
        capture.release()
        sampling_interval = max(1, round(source_fps / 8.0))
        source_indices = tuple(index * sampling_interval for index in range(len(frames)))
        timestamps = tuple(index / source_fps for index in source_indices) if source_fps > 0 else ()
        threshold = float(self.dynamic.params["thres"])
        count_num = int(self.dynamic.params["count_num"])
        flags = tuple(score > threshold for score in scores)
        return OfficialVideoResult(
            video=str(video),
            sampled_source_frame_indices=source_indices,
            timestamps=timestamps,
            timestamp_source="nominal_fps_diagnostic_only",
            source_fps=source_fps,
            sampling_interval=sampling_interval,
            frame_shape=(int(frames[0].shape[-2]), int(frames[0].shape[-1])),
            sampled_frame_count=len(frames),
            raw_flow_top5_mean=tuple(float(score) for score in scores),
            official_threshold=threshold,
            official_moving_flags=flags,
            official_moving_count=sum(flags),
            official_count_num=count_num,
            official_video_boolean=bool(result),
        )


def metadata_to_official_entries(videos: list[Path], metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_en": str(metadata.get(video.name, {}).get("prompt", "")),
            "dimension": ["dynamic_degree"],
            "video_list": [str(video.resolve())],
        }
        for video in videos
    ]


def evaluate_official_reference(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
    *,
    compute: Callable[..., Any] | None = None,
    upstream_path: Path = UPSTREAM_PATH,
) -> tuple[Any, UpstreamState | None]:
    if not model_weight.is_file() and compute is None:
        raise FileNotFoundError(f"official RAFT weight not found: {model_weight}")
    state = None
    if compute is None:
        module, state = import_official_module(upstream_path)
        compute = module.compute_dynamic_degree
    with tempfile.TemporaryDirectory(prefix="dynamic-official-") as temporary_root:
        metadata_path = Path(temporary_root) / "official_input.json"
        metadata_path.write_text(json.dumps(metadata_to_official_entries(videos, metadata), ensure_ascii=False), encoding="utf-8")
        raw = compute(str(metadata_path), device, {"model": str(model_weight)})
    return raw, state


def official_result_payload(result: OfficialVideoResult) -> dict[str, Any]:
    return {
        "sampling": {
            "sampled_source_frame_indices": list(result.sampled_source_frame_indices),
            "timestamps": list(result.timestamps),
            "timestamp_source": result.timestamp_source,
            "source_fps": result.source_fps,
            "sampling_interval": result.sampling_interval,
            "frame_shape": list(result.frame_shape),
            "sampled_frame_count": result.sampled_frame_count,
        },
        "official": {
            "raw_flow_top5_mean": list(result.raw_flow_top5_mean),
            "official_threshold": result.official_threshold,
            "official_moving_flags": list(result.official_moving_flags),
            "official_moving_count": result.official_moving_count,
            "official_count_num": result.official_count_num,
            "official_video_boolean": result.official_video_boolean,
        },
        "official_entrypoint": "vbench.dynamic_degree.DynamicDegree.infer",
    }
