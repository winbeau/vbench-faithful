import numpy as np
import pytest
import torch

from vbench_audit_models.torchvision_flow import TorchvisionRaftFlowModel


class Fake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, a, b, *, num_flow_updates):
        self.calls.append((a.clone(), b.clone(), num_flow_updates))
        result = torch.zeros((1, 2, *a.shape[-2:]))
        result[:, 0] = 3
        result[:, 1] = -2
        return [result * 0, result]


def make(tmp_path, scale=1.):
    weight = tmp_path / "local.pt"
    torch.save({}, weight)
    model = Fake()
    return TorchvisionRaftFlowModel("cpu", weight, model=model, inference_scale=scale), model


@pytest.mark.parametrize("shape", [(96, 91), (257, 259), (256, 256)])
def test_uint8_normalization_padding_crop_and_last_update(tmp_path, shape):
    adapter, model = make(tmp_path)
    a = np.zeros((*shape, 3), np.uint8)
    b = np.full_like(a, 255)
    flow = adapter.compute_flow(a, b)
    x, y, updates = model.calls[0]
    assert x.shape == y.shape and min(x.shape[-2:]) >= 128
    assert x.shape[-1] % 8 == x.shape[-2] % 8 == 0
    assert torch.all(x == -1) and torch.all(y == 1)
    assert updates == 20 and flow.shape == (*shape, 2)
    assert np.all(flow[..., 0] == 3) and np.all(flow[..., 1] == -2)
    assert adapter.identity["backend"] == "injected_test_model"


def test_missing_weight_never_initializes_or_downloads(tmp_path):
    with pytest.raises(FileNotFoundError):
        TorchvisionRaftFlowModel("cpu", tmp_path / "missing.pth")


def test_invalid_geometry_or_implicit_intensity_range_rejected(tmp_path):
    adapter, _ = make(tmp_path)
    with pytest.raises(ValueError):
        adapter.compute_flow(np.zeros((128, 128, 3)), np.zeros((128, 128, 3)))
    with pytest.raises(ValueError):
        adapter.compute_flow(np.zeros((128, 128, 3), np.uint8), np.zeros((128, 129, 3), np.uint8))


def test_checkpoint_is_loaded_strictly(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"unrecognized.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError):
        TorchvisionRaftFlowModel("cpu", path, model=Fake())


@pytest.mark.parametrize("scale", [1.5, 2.])
def test_resized_inference_returns_vectors_in_original_pixels(tmp_path, scale):
    adapter, _ = make(tmp_path, scale)
    image = np.zeros((129, 133, 3), np.uint8)
    flow = adapter.compute_flow(image, image)
    assert np.allclose(flow[..., 0], 3 * 133 / round(133 * scale))
    assert np.allclose(flow[..., 1], -2 * 129 / round(129 * scale))


@pytest.mark.parametrize("scale", [0., .5, float('nan'), float('inf'), 5.])
def test_invalid_scale_rejected(tmp_path, scale):
    with pytest.raises(ValueError):
        make(tmp_path, scale)
