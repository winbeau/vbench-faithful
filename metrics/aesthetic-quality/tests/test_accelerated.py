"""Retain normalized embeddings, native /10 scale and equal-video aggregation."""
import sys
from types import SimpleNamespace

import pytest
import torch

from aesthetic_quality.accelerated import compute


def test_mixed_video_frame_batches_keep_original_normalization_and_reducer(monkeypatch):
    calls = []
    head = torch.nn.Linear(2, 1, bias=False)
    head.weight.data[:] = torch.tensor([[0., 10.]])
    encoder = SimpleNamespace(eval=lambda: None)
    def encode(frames):
        calls.append(len(frames))
        assert not torch.is_grad_enabled()
        return frames.mean((2, 3))[:, :2].half()
    encoder.encode_image = encode
    def load(path):
        values = [3., 4., 0.] if path == "short" else [4., 3., 0.]
        return torch.tensor(values).reshape(1, 3, 1, 1).expand(2 if path == "short" else 3, -1, 2, 2)
    module = SimpleNamespace(get_aesthetic_model=lambda path: head,
        clip=SimpleNamespace(load=lambda *a, **k: (encoder, None)),
        clip_transform=lambda size: lambda frames: frames, load_video=load)
    monkeypatch.setitem(sys.modules, "vbench.aesthetic_quality", module)
    aggregate, returned = compute([{"video": "short"}, {"video": "long"}], "cpu", ["clip", "head"], batch_size=3)
    assert calls == [3, 2]
    assert [r["video_results"] for r in returned] == pytest.approx([.8, .6])
    assert aggregate == pytest.approx(.7)
