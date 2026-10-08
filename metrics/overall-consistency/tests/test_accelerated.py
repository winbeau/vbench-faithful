import pytest
import torch

from overall_consistency.accelerated import compute


def test_own_prompt_queries_normalization_and_native_video_mean(monkeypatch):
    seen = []
    def encode(rows, *args, **kwargs):
        seen.extend(r["prompt"] for r in rows)
        return torch.tensor([[0., 5.], [3., 4.]]), torch.tensor([[0., -2.], [2., 0.]])
    monkeypatch.setattr("vbench_audit_models.viclip.embeddings", encode)
    aggregate, values = compute([{"video": "one", "prompt": "first"},
                                 {"video": "two", "prompt": "second"}], "cpu", {})
    assert seen == ["first", "second"]
    assert [v["video_results"] for v in values] == pytest.approx([-1., .6])
    assert aggregate == pytest.approx(-.2)
