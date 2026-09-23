"""Local-checkpoint torchvision RAFT-large; no metric formula or downloads."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np


def source_identity(inference_scale=1.):
    import torchvision

    module = importlib.import_module("torchvision.models.optical_flow.raft")
    directory = Path(module.__file__).resolve().parent
    return {"backend": "torchvision_raft_large_local", "torchvision": torchvision.__version__,
            "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(directory.glob("*.py"))},
            "normalization": "uint8_rgb / 127.5 - 1; replicate pad to >=128 and multiple of 8",
            "inference_scale": inference_scale,
            "resize": "bilinear, align_corners=False; predicted vectors returned in original input pixels",
            "updates": 20}


class TorchvisionRaftFlowModel:
    def __init__(self, device, model_weight, *, inference_scale=1., model=None):
        import torch

        weight = Path(model_weight)
        if not weight.is_file():
            raise FileNotFoundError(weight)
        if not np.isfinite(inference_scale) or not 1 <= inference_scale <= 4:
            raise ValueError("inference scale must be finite in [1,4]")
        self.inference_scale = float(inference_scale)
        if model is None:
            from torchvision.models.optical_flow import raft_large

            self.identity = source_identity(self.inference_scale)
            model = raft_large(weights=None, progress=False)
        else:
            self.identity = {"backend": "injected_test_model", "updates": 20, "inference_scale": self.inference_scale}
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)

    def compute_flow(self, frame_a, frame_b):
        import torch
        from torch.nn.functional import interpolate, pad

        a, b = np.asarray(frame_a), np.asarray(frame_b)
        if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 1:
            raise ValueError("matching nonempty H,W,3 RGB frames required")
        if a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError("explicit uint8 input required; do not infer the intensity range")
        height, width = a.shape[:2]
        ih, iw = round(height * self.inference_scale), round(width * self.inference_scale)
        ph, pw = max(128, (ih + 7) // 8 * 8) - ih, max(128, (iw + 7) // 8 * 8) - iw
        frames = torch.from_numpy(np.stack((a, b))).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        frames = frames / 127.5 - 1
        if (ih, iw) != (height, width):
            frames = interpolate(frames, size=(ih, iw), mode="bilinear", align_corners=False)
        frames = pad(frames, (0, pw, 0, ph), mode="replicate")
        with torch.inference_mode():
            predictions = self.model(frames[:1].contiguous(), frames[1:].contiguous(), num_flow_updates=20)
            flow = predictions[-1][:, :, :ih, :iw]
            if (ih, iw) != (height, width):
                flow = interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
                flow = flow / flow.new_tensor([iw / width, ih / height])[None, :, None, None]
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
        if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
            raise ValueError("invalid flow prediction")
        return flow
