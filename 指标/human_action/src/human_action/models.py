from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_NUM_FRAMES = 16
OFFICIAL_TOP_K = 5
OFFICIAL_THRESHOLD = 0.85
AUDIT_MAX_WINDOWS = 4


def middle_sample_indices(sample_count: int, frame_count: int) -> tuple[int, ...]:
    if sample_count <= 0 or frame_count <= 0:
        raise ValueError("sample_count and frame_count must be positive")
    actual = min(sample_count, frame_count)
    intervals = np.linspace(0, frame_count, num=actual + 1).astype(int)
    indices = [(int(left) + int(right) - 1) // 2 for left, right in zip(intervals[:-1], intervals[1:])]
    if len(indices) < sample_count:
        indices.extend([indices[-1]] * (sample_count - len(indices)))
    return tuple(indices)


def temporal_window_indices(
    frame_count: int,
    *,
    max_windows: int = AUDIT_MAX_WINDOWS,
    samples_per_window: int = OFFICIAL_NUM_FRAMES,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    window_count = min(max_windows, frame_count)
    boundaries = np.linspace(0, frame_count, num=window_count + 1).astype(int)
    windows = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        start, stop = int(start), int(stop)
        local = middle_sample_indices(samples_per_window, stop - start)
        windows.append((start, stop, tuple(start + index for index in local)))
    return tuple(windows)


class LockedUmtClassifier:
    """One locked upstream UMT/K400 model shared by Official and Audit adapters."""

    def __init__(self, device: Any, model_weight: Path, upstream_module: Any):
        if not model_weight.is_file():
            raise FileNotFoundError(f"UMT weight not found: {model_weight}")
        torch = upstream_module.torch
        state_dict = torch.load(model_weight, map_location="cpu")
        model = upstream_module.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=400,
            all_frames=16,
            tubelet_size=1,
            use_learnable_pos_emb=False,
            fc_drop_rate=0.0,
            drop_rate=0.0,
            drop_path_rate=0.2,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=False,
            checkpoint_num=16,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        self.transform = upstream_module.Compose(
            [
                upstream_module.Resize(256, interpolation="bilinear"),
                upstream_module.CenterCrop(size=(224, 224)),
                upstream_module.ClipToTensor(),
                upstream_module.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.module = upstream_module
        self.device = device
        self.model = model.to(device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.categories_by_index = {
            int(index): action for index, action in upstream_module.build_dict().items()
        }

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self.categories_by_index[index] for index in range(400))

    def predict_transformed_tensor(self, clip: Any) -> Any:
        torch = self.module.torch
        images = clip.unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(images))
        return probabilities[0]

    def predict_transformed(self, clip: Any) -> np.ndarray:
        return self.predict_transformed_tensor(clip).detach().cpu().numpy().astype(np.float64)

    def predict_official_video(self, video: Path) -> np.ndarray:
        clip = self.module.load_video(str(video), self.transform, num_frames=OFFICIAL_NUM_FRAMES)
        return self.predict_transformed(clip)

    def predict_official_video_tensor(self, video: Path) -> Any:
        clip = self.module.load_video(str(video), self.transform, num_frames=OFFICIAL_NUM_FRAMES)
        return self.predict_transformed_tensor(clip)

    def decode_video(self, video: Path) -> tuple[np.ndarray, float | None]:
        import decord

        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(video), num_threads=1)
        if len(reader) == 0:
            raise ValueError(f"video contains no decodable frames: {video}")
        frames = reader.get_batch(range(len(reader))).asnumpy().astype(np.uint8)
        fps = float(reader.get_avg_fps())
        return frames, fps if np.isfinite(fps) and fps > 0 else None

    def predict_frame_indices(self, frames: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
        clip = self.transform(frames[list(indices)])
        return self.predict_transformed(clip)
