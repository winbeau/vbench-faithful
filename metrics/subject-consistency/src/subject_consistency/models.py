from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class OfficialDinoFeatureExtractor:
    """Locked VBench whole-frame DINO ViT-B/16 representation."""

    def __init__(self, device: Any, config: Mapping[str, Any], upstream: Path):
        from .backends.vbench import import_official_module

        self.module, self.upstream_state = import_official_module(upstream)
        self.device = device
        self.config = dict(config)
        if self.config.get("read_frame") is not False:
            raise ValueError("ordinary mp4 evaluation requires read_frame=false")
        self.model = self.module.torch.hub.load(
            repo_or_dir=self.config["repo_or_dir"], model=self.config["model"],
            source=self.config["source"], pretrained=False,
        )
        state_dict = self.module.torch.load(self.config["path"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device)
        self.model.eval()

    def extract(self, video: Path):
        images = self.module.load_video(str(video))
        images = self.module.dino_transform(224)(images)
        features = []
        with self.module.torch.no_grad():
            for image in images:
                feature = self.model(image.unsqueeze(0).to(self.device))
                features.append(self.module.F.normalize(feature, dim=-1, p=2))
        if not features:
            raise ValueError(f"video contains no decoded frames: {video}")
        return self.module.torch.cat(features, dim=0)
