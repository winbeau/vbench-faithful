"""Vary frame counts and queries to verify the original frame-weighted reducer."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from appearance_style.accelerated import compute


def test_all_frames_text_reuse_logit_scale_and_frame_weighting(monkeypatch):
    calls = []
    model = SimpleNamespace(logit_scale=torch.tensor(50.).log())
    model.encode_text = lambda tokens: calls.append(tokens.tolist()) or tokens.float()
    model.encode_image = lambda frames: frames
    module = SimpleNamespace(clip=SimpleNamespace(load=lambda **k: (model, None),
        tokenize=lambda words: torch.tensor([[1., 0.] if word == "a" else [0., 1.] for word in words])),
        clip_transform_Image=lambda size: lambda image: torch.tensor(np.asarray(image).reshape(-1)[:2].copy(), dtype=torch.float32),
        load_video=lambda video, **kwargs: [np.array([[[1, 0, 0]]], dtype=np.uint8)] * (1 if video == "short" else 3))
    monkeypatch.setitem(sys.modules, "vbench.appearance_style", module)
    rows = [{"video": name, "auxiliary_info": {"appearance_style": {"appearance_style": query}}}
            for name, query in [("short", "a"), ("long", "b"), ("second", "a")]]
    aggregate, returned = compute(rows, "cpu", {}, batch_size=2)
    assert len(calls) == 2  # Three videos, two distinct query embeddings.
    assert [len(r["frame_results"]) for r in returned] == [1, 3, 3]
    assert [r["video_path"] for r in returned] == ["short", "long", "second"]
    assert [r["video_results"] for r in returned] == pytest.approx([.5, 0., .5])
    assert aggregate == pytest.approx(2 / 7)  # Not the per-video mean 1/3.
