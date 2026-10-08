"""Graph replay must retain independent outputs and fall back on changed state."""
import pytest
import torch

from vbench_audit_models.cuda_image_graph import capture_image_encoder


def test_cpu_and_autograd_keep_original_execution():
    model = torch.nn.Linear(3, 2).eval()
    original = model.forward
    graph = capture_image_encoder(model)
    value = torch.ones(1, 3, requires_grad=True)
    assert torch.equal(model(value), original(value))
    model(value).sum().backward()
    assert value.grad is not None and graph.graph is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA replay required")
def test_cuda_replay_exact_and_independent_with_safe_state_fallback():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU()).cuda().eval()
    original = model.forward
    graph = capture_image_encoder(model)
    with torch.inference_mode():
        a, b = torch.rand(1, 3, 16, 16, device="cuda"), torch.rand(1, 3, 16, 16, device="cuda")
        expected = original(a)
        first = model(a)
        assert torch.equal(first, expected)
        second = model(b)
        assert torch.equal(first, expected) and torch.equal(second, original(b))
        assert graph.replays == 2
        model[0].weight.add_(1)
        assert torch.equal(model(b), original(b)) and graph.replays == 2
        smaller = b[:, :, :8, :8].contiguous()
        assert torch.equal(model(smaller), original(smaller)) and graph.replays == 2
