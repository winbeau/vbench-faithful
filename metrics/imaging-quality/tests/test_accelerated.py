"""MUSIQ preserves per-video averaging, original scales and all native shapes."""
import sys
from types import SimpleNamespace

import pytest
import torch

from imaging_quality.accelerated import compute


def test_native_transform_scale_unequal_frame_counts_and_short_batch(monkeypatch):
    calls = []

    class Model:
        training = True
        def __init__(self, pretrained_model_path):
            assert pretrained_model_path == "local-model"
        def to(self, device):
            return self
        def __call__(self, frames):
            assert not self.training and not torch.is_grad_enabled()
            calls.append(tuple(frames.shape))
            return frames.mean((1, 2, 3)).reshape(-1, 1) * 100

    def transform(images, mode):
        assert mode == "longer"
        return images / 255

    module = SimpleNamespace(MUSIQ=Model, transform=transform,
        load_video=lambda path: torch.full((3, 3, 2, 2), 255.) if path == "long" else torch.full((1, 3, 2, 4), 127.5))
    monkeypatch.setitem(sys.modules, "vbench.imaging_quality", module)
    aggregate, returned = compute([{"video": "long"}, {"video": "wide"}], "cpu", {"model_path": "local-model"}, batch_size=2)
    assert calls == [(2, 3, 2, 2), (1, 3, 2, 2), (1, 3, 2, 4)]
    assert [r["video_results"] for r in returned] == pytest.approx([100., 50.])
    assert aggregate == pytest.approx(.75)
