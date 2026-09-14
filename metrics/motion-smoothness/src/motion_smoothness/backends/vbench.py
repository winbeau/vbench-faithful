from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from vbench_audit_core.upstream import (
    import_official_module as _import_official_module,
    inspect_upstream as _inspect_upstream,
    resolve_upstream_path,
)

# Kept as None for source compatibility; callers must resolve at call time so
# changing VBENCH_AUDIT_UPSTREAM between runs cannot be silently ignored.
UPSTREAM_PATH = None
DEFAULT_CONFIG = None
DEFAULT_WEIGHT = Path.home() / ".cache/vbench/amt_model/amt-s.pth"


def inspect_upstream(path: Path | None = None):
    return _inspect_upstream(path)


def import_official_module(path: Path | None = None):
    return _import_official_module("motion_smoothness", path)


def metadata_to_official_entries(videos: list[Path]) -> list[dict[str, Any]]:
    return [{"prompt_en": "", "dimension": ["motion_smoothness"], "video_list": [str(video.resolve())]} for video in videos]


class OfficialMotionSmoothnessEvaluator:
    """Cache one official AMT model and expose isolated per-video scoring."""

    def __init__(
        self,
        device: Any,
        config: Path | None = None,
        checkpoint: Path = DEFAULT_WEIGHT,
        upstream_path: Path | None = None,
    ) -> None:
        config = config or (resolve_upstream_path() / "vbench/third_party/amt/cfgs/AMT-S.yaml")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"official AMT checkpoint not found: {checkpoint}")
        module, state = import_official_module(upstream_path)
        self.module = module
        self.upstream_state = state
        self.motion = module.MotionSmoothness(str(config), str(checkpoint), device)

    def evaluate_video(self, video: Path) -> float:
        value = float(self.motion.motion_score(str(video)))
        if not math.isfinite(value):
            raise ValueError("official motion evaluator returned a non-finite score")
        return value


def evaluate_official_batch(
    videos: list[Path],
    device: Any,
    config: Path | None = None,
    checkpoint: Path = DEFAULT_WEIGHT,
    upstream_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a shard with one AMT model while isolating bad videos."""
    evaluator = OfficialMotionSmoothnessEvaluator(device, config, checkpoint, upstream_path)
    results: list[dict[str, Any]] = []
    for video in videos:
        try:
            results.append({"video": str(video), "backend": "vbench", "score": evaluator.evaluate_video(video), "status": "succeeded"})
        except Exception as exc:
            results.append({"video": str(video), "backend": "vbench", "score": None, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return results


def evaluate_official_reference(
    videos: list[Path], device: Any, config: Path | None = None, checkpoint: Path = DEFAULT_WEIGHT,
    *, compute: Callable[..., Any] | None = None, upstream_path: Path | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    if compute is None and not checkpoint.is_file():
        raise FileNotFoundError(f"official AMT checkpoint not found: {checkpoint}")
    state = None
    if compute is None:
        module, state = import_official_module(upstream_path)
        compute = module.compute_motion_smoothness
    config = config or (resolve_upstream_path() / "vbench/third_party/amt/cfgs/AMT-S.yaml")
    with tempfile.TemporaryDirectory(prefix="motion-smooth-official-") as root:
        metadata = Path(root) / "official_input.json"
        metadata.write_text(json.dumps(metadata_to_official_entries(videos)), encoding="utf-8")
        result = compute(str(metadata), device, {"config": str(config), "ckpt": str(checkpoint)})
    return result, state
