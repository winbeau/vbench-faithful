import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_smoothness.accelerated import score_frames


@pytest.mark.parametrize("count", [5, 6, 19])
def test_pair_coverage_native_uint8_conversion_and_unpaired_tail(monkeypatch, count):
    calls = []
    class Padder:
        def __init__(self, *args):
            pass
        def pad(self, *args):
            return args
        def unpad(self, *args):
            return args
    def img2tensor(frame):
        return torch.tensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.
    def tensor2img(value):
        return (value * 255).squeeze(0).permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    monkeypatch.setitem(sys.modules, "vbench.third_party.amt.utils.utils", SimpleNamespace(
        InputPadder=Padder, img2tensor=img2tensor, tensor2img=tensor2img, check_dim_and_resize=lambda x: x))
    def predict(left, right, embt, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs == {"scale_factor": 1., "eval": True}
        assert len(embt) == len(left)
        calls.append(len(left))
        return {"imgt_pred": (left + right) / 2}
    motion = SimpleNamespace(device="cpu", anchor_resolution=8192**2, vram_avail=1,
        anchor_memory_bias=0, anchor_memory=1, embt=torch.tensor(.5).view(1,1,1,1), model=predict,
        fp=SimpleNamespace(extract_frame=lambda frames, start_from: frames[start_from::2]),
        get_diff=lambda a,b: np.abs(a.astype(int) - b.astype(int)).mean())
    frames = [np.full((2, 2, 3), i*10 + (3 if i%2 else 0), dtype=np.uint8) for i in range(count)]
    if count % 2 == 0:
        frames[-1][:] = 255  # Native scoring excludes this unpaired tail.
    assert score_frames(motion, frames, batch_size=3) == pytest.approx((255-3)/255)
    assert sum(calls) == len(frames[::2]) - 1
    assert max(calls) <= 3
