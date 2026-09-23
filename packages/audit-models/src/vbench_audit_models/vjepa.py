"""Pinned, local-only frozen V-JEPA 2.1 adapter. No scoring or downloads."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean_state(state):
    result = {}
    for key, value in state.items():
        clean = key.replace('module.', '').replace('backbone.', '')
        if clean in result:
            raise ValueError('colliding checkpoint keys')
        result[clean] = value
    return result


def check_namespace(root):
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in {'src', 'app'} or module is None:
            continue
        filename = getattr(module, '__file__', None)
        paths = [filename] if filename else list(getattr(module, '__path__', []))
        if not paths or any(not Path(p).resolve().is_relative_to(root) for p in paths):
            raise RuntimeError(f'foreign upstream namespace: {name}')


class FrozenVJEPA:
    def __init__(self, source_root, checkpoint, config, device='cuda:0'):
        import torch

        root, checkpoint = Path(source_root).resolve(), Path(checkpoint).resolve()
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
        if commit != config['source_commit'] or dirty:
            raise ValueError('upstream must be the clean pinned checkout')
        if (checkpoint.stat().st_size != config['checkpoint_size_bytes']
                or sha256(checkpoint) != config['checkpoint_sha256']):
            raise ValueError('checkpoint size/SHA mismatch')
        check_namespace(root)
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module('src.hub.backbones')
            check_namespace(root)
            # Never use pretrained=True: this pin's automatic URL is localhost.
            encoder, predictor = getattr(module, config['model_factory'])(pretrained=False)
            del predictor
        finally:
            sys.path.remove(str(root))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        clean = clean_state(state[config['checkpoint_key']])
        encoder.load_state_dict(clean, strict=True)
        del state, clean
        self.model = encoder.eval().requires_grad_(False).to(device)
        self.device, self.config = device, config
        self.identity = {'source_commit': commit, 'checkpoint_sha256': sha256(checkpoint),
                         'checkpoint_size_bytes': checkpoint.stat().st_size,
                         'strict_state_load': True, 'checkpoint_key': config['checkpoint_key'],
                         'torch': torch.__version__, 'cuda': torch.version.cuda,
                         'encoder_parameters': sum(p.numel() for p in encoder.parameters()),
                         'trainable_encoder_parameters': sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                         'model_factory': config['model_factory'], 'autodownload': False,
                         'preprocessing': config['input'], 'source_root': str(root)}

    def preprocess(self, frames):
        import torch
        import torch.nn.functional as F

        c = self.config['input']
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != c['frames']
                or frames.shape[-1] != 3 or frames.shape[1] != frames.shape[2]):
            raise ValueError('all 16 native square RGB uint8 frames required')
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device, torch.float32).permute(0, 3, 1, 2) / 255
        x = F.interpolate(x, size=(c['size'], c['size']), mode='bilinear', align_corners=False, antialias=True)
        mean, std = (torch.tensor(c[key], device=self.device)[None, :, None, None] for key in ('mean', 'std'))
        return ((x - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)

    def encode(self, frames):
        import torch

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            features = self.model(self.preprocess(frames))
        expected = (1, self.config['input']['frames'] // 2 * 24 * 24, 768)
        if tuple(features.shape) != expected or not torch.isfinite(features).all():
            raise ValueError(f'invalid dense features: {tuple(features.shape)}')
        return features[0].to('cpu', torch.float16).numpy()
