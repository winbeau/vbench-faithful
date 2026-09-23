"""Local-only CoTracker2 adapter. No scores, downloads, or dataset metadata."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


class CoTracker2Model:
    def __init__(self, source_root: Path, checkpoint: Path, device: str):
        import torch

        self.source_root = Path(source_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not (self.source_root / "cotracker/predictor.py").is_file():
            raise FileNotFoundError("tracker source root must contain cotracker/predictor.py")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        loaded = sys.modules.get("cotracker.predictor")
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(self.source_root):
            raise RuntimeError("a different cotracker checkout is already imported")
        sys.path.insert(0, str(self.source_root))
        module = importlib.import_module("cotracker.predictor")
        if not Path(module.__file__).resolve().is_relative_to(self.source_root):
            raise RuntimeError("tracker import escaped configured source root")
        self.device = torch.device(device)
        # v2 has an eight-frame learned time embedding, not v3's 60-frame one.
        self.model = module.CoTrackerPredictor(
            checkpoint=str(self.checkpoint), v2=True, window_len=8
        ).eval().to(self.device)

    def track_queries(self, frames: np.ndarray, queries_txy: np.ndarray) -> dict[str, np.ndarray]:
        """Explicit native-time point queries, with pre-query backward tracking.

        This is NOT the independent reverse endpoint check used by ``track``.
        Upstream forces query-frame coordinates/visibility to the supplied
        observations. Those forced values cannot certify tracking accuracy.
        Predictions outside the image are retained, never clamped to a border.
        """
        import torch

        frames, q = np.asarray(frames), np.asarray(queries_txy, np.float32)
        if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
                or frames.shape[0] < 2 or min(frames.shape[1:3]) < 2):
            raise ValueError("native T,H,W,3 uint8 frames required")
        if (q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all()
                or (q[:, 0] != np.floor(q[:, 0])).any() or (q < 0).any()
                or (q[:, 0] >= len(frames)).any() or (q[:, 1] > frames.shape[2] - 1).any()
                or (q[:, 2] > frames.shape[1] - 1).any()):
            raise ValueError("finite observed (integer frame,x,y) queries within native video required")
        with torch.inference_mode():
            video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
            video = video[None].to(self.device, dtype=torch.float32)
            queries = torch.tensor(q, device=self.device, dtype=torch.float32)[None]
            tracks, visible = self.model(video, queries=queries, backward_tracking=True)
        result = {"tracks": tracks[0].cpu().numpy(), "visible": visible[0].cpu().numpy()}
        if (result["tracks"].shape != (len(frames), len(q), 2)
                or result["visible"].shape != (len(frames), len(q))
                or not np.isfinite(result["tracks"]).all()):
            raise ValueError("tracker returned invalid query geometry")
        return result

    def track(self, frames: np.ndarray, queries_xy: np.ndarray) -> dict[str, np.ndarray]:
        """Forward tracks and an independent reverse endpoint-query cycle."""
        import torch

        with torch.inference_mode():
            video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
            video = video[None].to(self.device, dtype=torch.float32)
            xy = torch.as_tensor(queries_xy, device=self.device, dtype=torch.float32)
            queries = torch.cat((torch.zeros_like(xy[:, :1]), xy), dim=-1)[None]
            forward, visible = self.model(video, queries=queries)
            end = forward[:, -1].clone()
            height, width = frames.shape[1:3]
            end[..., 0].clamp_(0, width - 1)
            end[..., 1].clamp_(0, height - 1)
            reverse_queries = torch.cat((torch.zeros_like(end[..., :1]), end), dim=-1)
            reverse, reverse_visible = self.model(video.flip(1), queries=reverse_queries)
        return {
            "tracks": forward[0].cpu().numpy(),
            "visible": visible[0].cpu().numpy(),
            "reverse_tracks": reverse[0].flip(0).cpu().numpy(),
            "reverse_visible": reverse_visible[0].flip(0).cpu().numpy(),
        }
