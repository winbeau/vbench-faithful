import pytest
import torch

from temporal_style.accelerated import compute


def test_native_normalization_sign_and_video_mean(monkeypatch):
    monkeypatch.setattr("vbench_audit_models.viclip.embeddings", lambda *a: (
        torch.tensor([[3., 4.], [-2., 0.]]), torch.tensor([[0., 9.], [3., 0.]])))
    aggregate, values = compute([{"video": "one"}, {"video": "two"}], "cpu", {})
    assert [v["video_results"] for v in values] == pytest.approx([.8, -1.])
    assert aggregate == pytest.approx(-.1)
