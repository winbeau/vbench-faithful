from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

UPSTREAM_PATH = Path("/home/msy625/vbench1")
UPSTREAM_REMOTE = "https://github.com/msy625/VBench.git"
UPSTREAM_BRANCH = "master"
UPSTREAM_SHA = "13dee903cc97e2633ed6e8f50dea61bc90717935"
DEFAULT_WEIGHT = Path.home() / ".cache/vbench/grit_model/grit_b_densecap_objectdet.pth"


@dataclass(frozen=True)
class UpstreamState:
    path: str
    remote: str
    branch: str
    sha: str
    dirty: bool
    submodules: tuple[str, ...]


def inspect_upstream(path: Path = UPSTREAM_PATH) -> UpstreamState:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()

    remote = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current")
    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    submodule_text = git("submodule", "status")
    return UpstreamState(str(path.resolve()), remote, branch, sha, dirty, tuple(line for line in submodule_text.splitlines() if line))


def verify_upstream(path: Path = UPSTREAM_PATH) -> UpstreamState:
    state = inspect_upstream(path)
    expected = (UPSTREAM_REMOTE, UPSTREAM_BRANCH, UPSTREAM_SHA)
    actual = (state.remote, state.branch, state.sha)
    if actual != expected:
        raise RuntimeError(f"upstream identity mismatch: expected={expected}, actual={actual}")
    if state.dirty:
        raise RuntimeError("upstream worktree is dirty; refusing official evaluation")
    return state


def import_official_module(path: Path = UPSTREAM_PATH):
    state = verify_upstream(path)
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    module = importlib.import_module("vbench.spatial_relationship")
    module_path = Path(module.__file__).resolve()
    if path.resolve() not in module_path.parents:
        raise RuntimeError(f"wrong vbench import: {module_path}")
    return module, state


def metadata_to_official_entries(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    query_parser: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for video in videos:
        item = metadata[video.name]
        query = query_parser(item)
        entries.append(
            {
                "prompt_en": item.get("prompt", ""),
                "dimension": ["spatial_relationship"],
                "video_list": [str(video.resolve())],
                "auxiliary_info": {
                    "spatial_relationship": {
                        "spatial_relationship": {
                            "object_a": query.subject,
                            "object_b": query.object,
                            "relationship": query.relation,
                        }
                    }
                },
            }
        )
    return entries


def evaluate_official(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    query_parser: Callable[[Mapping[str, Any]], Any],
    device: Any,
    model_weight: Path,
    *,
    compute: Callable[..., Any] | None = None,
    upstream_path: Path = UPSTREAM_PATH,
) -> tuple[Any, UpstreamState | None]:
    if not model_weight.is_file() and compute is None:
        raise FileNotFoundError(f"official GRiT weight not found: {model_weight}")
    state = None
    if compute is None:
        module, state = import_official_module(upstream_path)
        compute = module.compute_spatial_relationship
    entries = metadata_to_official_entries(videos, metadata, query_parser)
    with tempfile.TemporaryDirectory(prefix="spatial-official-") as temp_root:
        metadata_path = Path(temp_root) / "official_input.json"
        metadata_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        raw = compute(str(metadata_path), device, {"model_weight": str(model_weight)})
    return raw, state


def normalize_official_results(raw: Any, metadata: Mapping[str, Mapping[str, Any]], query_parser: Callable[[Mapping[str, Any]], Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("official result must be (dataset_score, video_results)")
    dataset_score, video_results = raw
    normalized = []
    for item in video_results:
        video = Path(item["video_path"])
        meta = metadata[video.name]
        query = query_parser(meta)
        normalized.append(
            {
                "video": str(video),
                "prompt": meta.get("prompt", ""),
                "subject": query.subject,
                "relation": query.relation,
                "object": query.object,
                "backend": "vbench",
                "score": float(item["video_results"]),
                "status": "succeeded",
                "failure_reason": None,
                "diagnostics": {
                    "frame_results": item.get("frame_results"),
                    "video_results": item.get("video_results"),
                    "dataset_score": dataset_score,
                    "official_entrypoint": "vbench.spatial_relationship.compute_spatial_relationship",
                },
            }
        )
    return normalized


def detections_from_instances(instances: Any) -> list[tuple[str, list[int], float | None]]:
    labels = instances.pred_object_descriptions.data
    boxes = instances.pred_boxes.tensor.detach().cpu()
    scores = instances.scores.detach().cpu() if instances.has("scores") else None
    detections = []
    for index, label in enumerate(labels):
        box = [int(value) for value in boxes[index].tolist()]
        confidence = float(scores[index].item()) if scores is not None else None
        detections.append((label, box, confidence))
    return detections


class OfficialGritDetector:
    """Reuse the locked fork's frame sampling, resize, GRiT model and wrapper."""

    def __init__(self, device: Any, model_weight: Path, upstream_path: Path = UPSTREAM_PATH):
        if not model_weight.is_file():
            raise FileNotFoundError(f"official GRiT weight not found: {model_weight}")
        self.module, self.upstream_state = import_official_module(upstream_path)
        self.device = device
        self.model = self.module.DenseCaptioning(device)
        self.model.initialize_model_det(model_weight=str(model_weight))

    def detect_video(self, video: Path) -> tuple[list[int], list[list[Any]]]:
        video_tensor = self.module.load_video(str(video), num_frames=16)
        _, _, height, width = video_tensor.size()
        if min(height, width) > 768:
            scale = 720.0 / min(height, width)
            video_tensor = self.module.transforms.Resize(size=(int(scale * height), int(scale * width)))(video_tensor)
        predictions = []
        with self.module.torch.no_grad():
            for frame in video_tensor.permute(0, 2, 3, 1):
                raw_predictions, _ = self.model.run_det_tensor(frame)
                instances = raw_predictions["instances"]
                predictions.append(detections_from_instances(instances))

        # Recompute indices through the exact upstream helper for diagnostics;
        # the frames themselves are still loaded by upstream load_video().
        utils = importlib.import_module("vbench.utils")
        reader = utils.VideoReader(str(video), num_threads=1)
        indices = list(utils.get_frame_indices(16, len(reader), sample="middle"))
        return indices, predictions


class OfficialVBenchEvaluator:
    """Load one official model and isolate failures at the video boundary."""

    def __init__(self, device: Any, model_weight: Path, upstream_path: Path = UPSTREAM_PATH):
        if not model_weight.is_file():
            raise FileNotFoundError(f"official GRiT weight not found: {model_weight}")
        self.module, self.upstream_state = import_official_module(upstream_path)
        self.device = device
        self.model = self.module.DenseCaptioning(device)
        self.model.initialize_model_det(model_weight=str(model_weight))

    def evaluate_video(self, video: Path, metadata_item: Mapping[str, Any], query: Any) -> Any:
        # This is the structure produced by upstream load_dimension_info().
        # Calling the locked spatial_relationship() function directly lets one
        # model serve the worker while a corrupt video remains an isolated error.
        video_dict = [
            {
                "prompt": metadata_item.get("prompt", ""),
                "video_list": [str(video.resolve())],
                "auxiliary_info": {
                    "spatial_relationship": {
                        "object_a": query.subject,
                        "object_b": query.object,
                        "relationship": query.relation,
                    }
                },
            }
        ]
        return self.module.spatial_relationship(self.model, video_dict, self.device)
