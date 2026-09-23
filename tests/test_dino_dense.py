from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from vbench_audit_models.dino_dense import DinoDenseModel, local_vit_base, model_geometry


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(6, 18)

    def forward(self, x):
        value = self.qkv(x)
        n = x.shape[1]
        attention = torch.ones((len(x), 2, n, n)) / n
        return value[..., :6], attention


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()

    def forward(self, x):
        return x + .01 * self.attn(x)[0]


class FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(12)])
        self.patch_embed = SimpleNamespace(patch_size=16)
        self.last_input = None

    def get_intermediate_layers(self, x, n=1):
        self.last_input = x.clone()
        n = (x.shape[2] // 16) * (x.shape[3] // 16) + 1
        features = torch.arange(n * 6, dtype=torch.float32).reshape(1, n, 6) / 20 + 1
        for block in self.blocks: features = block(features)
        return [features]


def test_geometry_centers_no_crop_and_original_pixel_units():
    shape, q = model_geometry(100, 200, 512)
    assert shape == (256, 512)
    assert q.shape == (16 * 32, 2)
    assert q[0].tolist() == pytest.approx([2.625, 2.625])
    assert q[-1].tolist() == pytest.approx([196.375, 96.375])


@pytest.mark.parametrize("side", [0, 31, 511, float('nan'), 512.])
def test_invalid_resize(side):
    with pytest.raises(ValueError): model_geometry(100, 200, side)


def test_missing_checkpoint_is_checked_without_loading_source(tmp_path):
    with pytest.raises(FileNotFoundError): DinoDenseModel(tmp_path, tmp_path / "missing.pth", "cpu")


def test_adapter_rgb_normalization_facets_hook_cleanup_and_storage(tmp_path):
    fake = FakeDino()
    weight = tmp_path / "fake.pth"; torch.save(fake.state_dict(), weight)
    model = DinoDenseModel(tmp_path, weight, "cpu", max_side=32, model=fake)
    frames = np.full((2, 16, 32, 3), 255, np.uint8)
    r = model.extract(frames)
    assert r["key9"].shape == r["token11"].shape == (2, 2, 6)
    assert r["key9"].dtype == np.float16
    assert np.allclose(np.linalg.norm(r["key9"], axis=-1), 1, atol=.001)
    assert np.array_equal(r["key9"][0], r["key9"][1])
    assert not np.array_equal(r["key9"], r["token11"])
    assert np.allclose(fake.last_input[0, :, 0, 0], (1 - np.array([.485,.456,.406])) / [.229,.224,.225])
    assert not fake.blocks[9].attn.qkv._forward_hooks
    assert not fake.blocks[11].attn._forward_hooks
    assert model.identity["backend"] == "injected_test_model"


def test_bad_inputs_and_weights_are_rejected(tmp_path):
    fake = FakeDino()
    weight = tmp_path / "fake.pth"; torch.save(fake.state_dict(), weight)
    model = DinoDenseModel(tmp_path, weight, "cpu", max_side=32, model=fake)
    for frames in (np.zeros((2, 16, 32, 3)), np.zeros((0, 16, 32, 3), np.uint8), np.zeros((16, 32, 3), np.uint8)):
        with pytest.raises(ValueError): model.extract(frames)
    torch.save({}, weight)
    with pytest.raises(RuntimeError): DinoDenseModel(tmp_path, weight, "cpu", max_side=32, model=FakeDino())


def test_source_files_required(tmp_path):
    with pytest.raises(FileNotFoundError): local_vit_base(tmp_path)


def test_local_source_does_not_use_or_leak_generic_utils(tmp_path, monkeypatch):
    previous = SimpleNamespace(marker="unrelated")
    monkeypatch.setitem(sys.modules, "utils", previous)
    (tmp_path / "utils.py").write_text("marker = 'configured-local-dino'\n")
    (tmp_path / "vision_transformer.py").write_text(
        "from utils import marker\ndef vit_base(**kwargs): return marker, kwargs\n")
    label, kwargs = local_vit_base(tmp_path)
    assert label == "configured-local-dino"
    assert kwargs == {"patch_size": 16, "num_classes": 0}
    assert sys.modules["utils"] is previous


def test_hooks_removed_after_failed_feature_extraction(tmp_path):
    fake = FakeDino()
    with torch.no_grad(): fake.blocks[9].attn.qkv.weight.fill_(float('nan'))
    weight = tmp_path / "fake.pth"; torch.save(fake.state_dict(), weight)
    model = DinoDenseModel(tmp_path, weight, "cpu", max_side=32, model=fake)
    with pytest.raises(ValueError, match="invalid model descriptors"):
        model.extract(np.ones((2, 16, 32, 3), np.uint8))
    assert not fake.blocks[9].attn.qkv._forward_hooks
    assert not fake.blocks[11].attn._forward_hooks
