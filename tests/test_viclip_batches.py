"""Keep video sampling, tail coverage and prompt-independent feature reuse."""
import sys
from types import SimpleNamespace

import torch

from vbench_audit_models.viclip import embeddings


def test_batches_cache_only_identical_video_inputs_and_keep_new_queries(monkeypatch, tmp_path):
    calls, samples = [], []
    class Model:
        embed_dim = 2
        def __init__(self, **kwargs):
            self.vision_encoder = SimpleNamespace(transformer=SimpleNamespace(checkpoint_num=24))
        def to(self, device):
            return self
        def encode_text(self, texts):
            return torch.tensor([[len(t), 1.] for t in texts])
        def encode_vision(self, videos, test):
            assert test and not torch.is_grad_enabled()
            calls.append(len(videos))
            return torch.stack((videos.mean((1, 2, 3, 4)), torch.ones(len(videos))), dim=1)
    def read(path, **kwargs):
        samples.append(kwargs)
        return torch.full((8, 3, 2, 2), float(path))
    monkeypatch.setitem(sys.modules, "vbench.utils", SimpleNamespace(CACHE_DIR=str(tmp_path),
        clip_transform=lambda size: lambda t: t, read_frames_decord_by_fps=read))
    monkeypatch.setitem(sys.modules, "vbench.third_party.ViCLIP.simple_tokenizer", SimpleNamespace(SimpleTokenizer=lambda p: None))
    monkeypatch.setitem(sys.modules, "vbench.third_party.ViCLIP.viclip", SimpleNamespace(ViCLIP=Model))
    monkeypatch.setenv("VBENCH_EVAL_RUN_INFERENCE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("VBENCH_EVAL_RUN_INFERENCE_CONTEXT", '{"source": "test"}')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"test model")
    rows = [{"video": str(i), "prompt": "x"} for i in range(5)]
    video, text = embeddings(rows, "cpu", {"pretrain": checkpoint}, batch_size=2)
    assert calls == [2, 2, 1]
    assert video[:, 0].tolist() == list(range(5))
    changed = [{**r, "prompt": "different"} for r in rows]
    diagnostics = {}
    again, new_text = embeddings(changed, "cpu", {"pretrain": checkpoint}, batch_size=2, diagnostics=diagnostics)
    assert calls == [2, 2, 1]
    assert torch.equal(again, video) and not torch.equal(new_text, text)
    assert diagnostics["run_shared_inference"]["hits"] == 3
    assert diagnostics["fresh_visual_inference"] is False
    assert all(s == {"num_frames": 8, "sample": "middle"} for s in samples)
    rows[0]["video"] = "9"
    changed_video, _ = embeddings(rows, "cpu", {"pretrain": checkpoint}, batch_size=2)
    assert calls == [2, 2, 1, 2]
    assert changed_video[0, 0] == 9


def test_random_initializers_restored_after_failed_strict_load(monkeypatch, tmp_path):
    import pytest
    from vbench_audit_models.viclip import load_model
    initializers = {name: getattr(torch.nn.init, name) for name in
                    ("uniform_", "normal_", "kaiming_uniform_", "xavier_uniform_")}
    def fail(**kwargs):
        torch.nn.Linear(16, 16)
        raise RuntimeError("strict checkpoint mismatch")
    monkeypatch.setitem(sys.modules, "vbench.third_party.ViCLIP.viclip", SimpleNamespace(ViCLIP=fail))
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="strict checkpoint"):
        load_model(None, "cpu", {"pretrain": checkpoint})
    assert all(getattr(torch.nn.init, name) is fn for name, fn in initializers.items())
