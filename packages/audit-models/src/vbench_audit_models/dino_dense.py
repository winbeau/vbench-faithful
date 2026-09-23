"""Local-only DINO ViT-B/16 descriptors; no metric, downloads or segmentation.

Block-9 keys and final normalized patch tokens are separate facets. A key is
not an object identity; CLS attention is not a semantic mask. The adapter only
returns model evidence and explicit pixel/patch geometry.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_vit_base(source_root):
    """Load the two inspected source files without leaking generic 'utils'.

    Used in isolated inference workers, not a thread-safe plugin loader.
    No source file is modified and hub/pretrained URL constructors are avoided.
    """
    root = Path(source_root).resolve()
    for name in ("utils.py", "vision_transformer.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    utils = _load_file("_audit_dino_utils", root / "utils.py")
    previous = sys.modules.get("utils")
    sys.modules["utils"] = utils
    try:
        module = _load_file("_audit_dino_vit", root / "vision_transformer.py")
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous
    return module.vit_base(patch_size=16, num_classes=0)


def model_geometry(height, width, max_side):
    if (not isinstance(max_side, int) or max_side < 32 or max_side % 16
            or min(height, width) < 1):
        raise ValueError("positive image size and max_side a multiple of 16 >=32 required")
    scale = max_side / max(height, width)
    ih, iw = max(16, round(height * scale / 16) * 16), max(16, round(width * scale / 16) * 16)
    y, x = np.mgrid[:ih // 16, :iw // 16]
    # Pixel-center convention agrees with align_corners=False interpolation.
    q = np.stack(((16 * x + 8) * width / iw - .5,
                  (16 * y + 8) * height / ih - .5), axis=-1).reshape(-1, 2)
    return (ih, iw), q.astype(np.float32)


class DinoDenseModel:
    def __init__(self, source_root, checkpoint, device, *, max_side=512, model=None):
        import torch

        weight = Path(checkpoint)
        if not weight.is_file():
            raise FileNotFoundError(weight)
        model_geometry(32, 32, max_side)
        self.max_side = max_side
        if model is None:
            model = local_vit_base(source_root)
            root = Path(source_root).resolve()
            self.identity = {"backend": "local_dino_vit_base16", "source_root": str(root),
                             "source_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                              for name in ("utils.py", "vision_transformer.py")}}
        else:
            self.identity = {"backend": "injected_test_model"}
        self.identity.update(checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                             max_side=max_side, key_block_zero_based=9, token_block_zero_based=11,
                             normalization="RGB/255 then ImageNet mean/std; bilinear resize, align_corners=False",
                             descriptor_storage="L2 normalized float32 model outputs; cache rounded to float16",
                             attention="last block mean-head CLS-to-patch, not a semantic mask")
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        if len(self.model.blocks) != 12 or self.model.patch_embed.patch_size != 16:
            raise ValueError("this adapter requires twelve-block DINO with patch size 16")

    def extract(self, frames):
        import torch
        from torch.nn import functional as F

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or min(frames.shape[:3]) < 1:
            raise ValueError("nonempty T,H,W,3 uint8 RGB required")
        _, height, width, _ = frames.shape
        (ih, iw), queries = model_geometry(height, width, self.max_side)
        mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        outputs = {"key9": [], "token11": [], "attention": []}
        captured = {}

        def key_hook(_module, _inputs, value):
            # Linear QKV channels are [query, key, value], each concatenated heads.
            captured["key9"] = value[..., value.shape[-1] // 3:2 * value.shape[-1] // 3]

        def attention_hook(_module, _inputs, value):
            captured["attention"] = value[1][:, :, 0, 1:].mean(dim=1)

        hooks = [self.model.blocks[9].attn.qkv.register_forward_hook(key_hook),
                 self.model.blocks[11].attn.register_forward_hook(attention_hook)]
        try:
            with torch.inference_mode():
                for frame in frames:
                    x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None].to(self.device, dtype=torch.float32) / 255
                    x = F.interpolate(x, (ih, iw), mode="bilinear", align_corners=False)
                    tokens = self.model.get_intermediate_layers((x - mean) / std, n=1)[0]
                    for name, value in (("key9", captured["key9"][:, 1:]), ("token11", tokens[:, 1:])):
                        if value.shape[1] != len(queries) or not torch.isfinite(value).all() or torch.any(value.norm(dim=-1) == 0):
                            raise ValueError("invalid model descriptors/patch geometry")
                        outputs[name].append(F.normalize(value, dim=-1)[0].cpu().numpy().astype(np.float16))
                    outputs["attention"].append(captured["attention"][0].cpu().numpy().astype(np.float32))
                    captured.clear()
        finally:
            for hook in hooks:
                hook.remove()
        result = {name: np.stack(value) for name, value in outputs.items()}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("nonfinite feature evidence")
        return {**result, "queries": queries, "grid_shape": np.array([ih // 16, iw // 16]),
                "model_shape": np.array([ih, iw]), "input_shape": np.asarray(frames.shape)}
