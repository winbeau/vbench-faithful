"""Guard dtype-preserving loading and fail closed on incomplete visual weights."""
import sys
from types import SimpleNamespace

import pytest
import torch

from vbench_audit_models.clip_vision import load_clip_vision


class TinyVision(torch.nn.Module):
    before_cpu_float = None

    def __init__(self, resolution, patch, width, layers, heads, output):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, width, patch, bias=False)
        self.positional_embedding = torch.nn.Parameter(torch.empty((resolution // patch) ** 2 + 1, width))
        self.proj = torch.nn.Parameter(torch.empty(width, output))
        self.ln = torch.nn.LayerNorm(width)

    def float(self):
        type(self).before_cpu_float = {k: v.dtype for k, v in self.state_dict().items()}
        return super().float()


def checkpoint(monkeypatch):
    source = TinyVision(4, 2, 64, 0, 1, 3).half()
    for value in source.parameters():
        torch.nn.init.constant_(value, .25)
    state = {"visual." + k: v for k, v in source.state_dict().items()}
    state["unused_text.weight"] = torch.zeros(2)
    monkeypatch.setattr(torch.jit, "load", lambda *a, **k: SimpleNamespace(state_dict=lambda: state))
    def convert(model):
        model.conv1.half()
        model.proj.data = model.proj.data.half()
    monkeypatch.setitem(sys.modules, "clip.model", SimpleNamespace(VisionTransformer=TinyVision, convert_weights=convert))
    return state


def test_load_copies_values_without_adopting_checkpoint_norm_dtype(monkeypatch):
    state = checkpoint(monkeypatch)
    model = load_clip_vision("verified-local.pt", "cpu")
    assert not model.training
    assert TinyVision.before_cpu_float["ln.weight"] == torch.float32
    assert TinyVision.before_cpu_float["positional_embedding"] == torch.float32
    assert TinyVision.before_cpu_float["conv1.weight"] == torch.float16
    for key, value in model.visual.state_dict().items():
        assert not value.is_meta
        assert torch.equal(value, state["visual." + key].float())


def test_incomplete_visual_checkpoint_is_rejected(monkeypatch):
    state = checkpoint(monkeypatch)
    del state["visual.ln.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_clip_vision("incomplete.pt", "cpu")
