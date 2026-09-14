from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol, Sequence


class VideoTextEncoder(Protocol):
    def encode_video(self, video_path: Path) -> Sequence[float]:
        ...

    def encode_text(self, text: str) -> Sequence[float]:
        ...


class LockedViCLIPEncoder:
    """The locked VBench 1.0 ViCLIP representation and preprocessing."""

    def __init__(self, official_module: Any, device: Any, checkpoint: Path):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"official ViCLIP checkpoint not found: {checkpoint}")
        self.module = official_module
        self.device = device
        tokenizer_path = Path(official_module.CACHE_DIR) / "ViCLIP/bpe_simple_vocab_16e6.txt.gz"
        self.tokenizer = official_module.SimpleTokenizer(str(tokenizer_path))
        self.model = official_module.ViCLIP(tokenizer=self.tokenizer, pretrain=str(checkpoint)).to(device)
        self.model.eval()
        self._text_cache: dict[str, tuple[float, ...]] = {}
        self._tensor_text_cache: dict[str, Any] = {}

    @staticmethod
    def _normalized_tuple(tensor: Any, name: str) -> tuple[float, ...]:
        import torch

        vector = tensor.float().reshape(-1)
        norm = vector.norm()
        if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(norm)) or float(norm) == 0.0:
            raise ValueError(f"{name} must be finite with non-zero norm")
        vector = vector / norm
        values = tuple(float(value) for value in vector.detach().cpu().tolist())
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} normalization produced non-finite values")
        return values

    def encode_video(self, video_path: Path) -> tuple[float, ...]:
        import torch

        with torch.no_grad():
            frames = self.module.read_frames_decord_by_fps(str(video_path), num_frames=8, sample="middle")
            frames = self.module.clip_transform(224)(frames).to(self.device)
            feature = self.module.get_vid_features(self.model, frames.unsqueeze(0))
        return self._normalized_tuple(feature, "video feature")

    def encode_text(self, text: str) -> tuple[float, ...]:
        if text in self._text_cache:
            return self._text_cache[text]
        import torch

        with torch.no_grad():
            feature = self.module.get_text_features(
                self.model, text, self.tokenizer, self._tensor_text_cache
            )
        normalized = self._normalized_tuple(feature, "text feature")
        self._text_cache[text] = normalized
        return normalized
