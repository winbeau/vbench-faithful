"""Scene evidence scorers.

Model-specific code lives here so the metric and spatial aggregation remain
independent of a particular image-text implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SceneEvidenceScorer(Protocol):
    """Return a continuous scene-support score in the inclusive range [0, 1]."""

    def score(self, image: Any, scene_text: str) -> float:
        ...


@dataclass
class CallableSceneScorer:
    """Small adapter useful for tests and controlled ablations."""

    function: Any

    def score(self, image: Any, scene_text: str) -> float:
        value = float(self.function(image, scene_text))
        if not 0.0 <= value <= 1.0:
            raise ValueError("scene evidence scorer must return a value in [0, 1]")
        return value


class OpenClipSceneScorer:
    """Optional OpenCLIP image-text scorer.

    OpenCLIP is intentionally optional: importing this class does not import
    torch or download weights. A caller must provide a model name and may use
    the repository's normal checkpoint/cache configuration.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Any = "cuda"):
        import torch

        self._torch = torch
        self.device = device
        try:
            import open_clip
        except ImportError:
            # The uploaded OpenAI ViT-B/32 checkpoint is also natively supported
            # by the installed OpenAI CLIP package.  This preserves the same
            # image/text encoder and cosine-to-support mapping without any download.
            try:
                import clip
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Scene scorer requires open_clip_torch or the OpenAI clip package"
                ) from exc
            if pretrained == "openai":
                raise RuntimeError(
                    "open_clip_torch is unavailable; pass a local OpenAI CLIP checkpoint path"
                )
            self.model, self.preprocess = clip.load(pretrained, device=device, jit=False)
            self.tokenizer = clip.tokenize
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def score(self, image: Any, scene_text: str) -> float:
        with self._torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_input = self.tokenizer([scene_text]).to(self.device)
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosine = float((image_features * text_features).sum().item())
        # Map cosine [-1, 1] to a documented bounded support score.
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def build_scorer(kind: str, device: Any = "cuda", model_name: str = "ViT-B-32", pretrained: str = "openai") -> SceneEvidenceScorer:
    if kind == "openclip":
        return OpenClipSceneScorer(model_name=model_name, pretrained=pretrained, device=device)
    raise ValueError(f"unknown scene scorer: {kind}")
