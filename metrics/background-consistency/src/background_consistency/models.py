"""Pinned, local-only CLIP encoder using the exact VBench video transform."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

MODEL_ASSETS = ("clip_vit_b32",)


def required_assets():
    return MODEL_ASSETS


class OfficialClipEncoder:
    def __init__(self, checkpoint, *, device, expected_sha256=None, upstream=None):
        from vbench_audit_core.inputs import sha256_file
        from vbench_audit_core.upstream import import_official_module
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError("an existing local CLIP checkpoint is required; downloads are disabled")
        weight_hash = sha256_file(checkpoint)
        if expected_sha256 is not None and weight_hash != expected_sha256:
            raise ValueError("CLIP checkpoint SHA256 mismatch")
        self.module, state = import_official_module("background_consistency", upstream)
        self.model, self.preprocess = self.module.clip.load(str(checkpoint), device=device, jit=False)
        self.model.eval()
        if self.model.visual.input_resolution != 224 or self.model.visual.conv1.kernel_size != (32, 32):
            raise ValueError("background origin requires CLIP ViT-B/32 at 224px")
        self.transform = self.module.clip_transform(224)
        self.device = device
        self.provenance = {"upstream": asdict(state), "checkpoint": str(checkpoint), "sha256": weight_hash,
                           "preprocess": "VBench clip_transform(224), float32 native RGB input, all frames",
                           "model_dtype": str(self.model.dtype)}

    def decode(self, video):
        return self.module.load_video(str(video))

    def features(self, frames):
        import torch
        from torch.nn import functional as F
        if frames.ndim != 4 or frames.shape[1] != 3 or len(frames) < 2:
            raise ValueError("native RGB frames require [T,3,H,W], T>=2")
        if not bool(torch.isfinite(frames).all()) or frames.min() < 0 or frames.max() > 255:
            raise ValueError("native RGB values must be finite and in [0,255]")
        # VBench returns float32 unnormalized bytes. Resize before casting
        # uint8 would round bicubic interpolation and break official parity.
        images = self.transform(frames.float()).to(self.device)
        with torch.inference_mode():
            return F.normalize(self.model.encode_image(images), dim=-1, p=2)

    def upstream_score(self, video):
        import torch
        with torch.inference_mode():
            _, values = self.module.background_consistency(self.model, self.preprocess, [str(video)], self.device, False)
        return float(values[0]["video_results"])


def build_model(config, *, device):
    return OfficialClipEncoder(config["checkpoint"], device=device,
                               expected_sha256=config.get("sha256"), upstream=config.get("upstream"))
