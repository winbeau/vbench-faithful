from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from subject_consistency.models import OfficialDinoFeatureExtractor, OfficialDinoPatchExtractor, native_rgb_uint8


def test_bytes_and_upstream_float_take_the_same_unrounded_resize_path():
    seen = []
    def transform(size):
        def resize(x):
            seen.append(x.dtype)
            assert x.dtype == torch.float32
            return F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        return resize
    class Model:
        patch_embed = SimpleNamespace(patch_size=16)
        def __call__(self, x):
            return x.flatten(1)
        def get_intermediate_layers(self, x, n):
            return [torch.stack([x.mean((2, 3)), x[:, :, 0, 0]], dim=1)]
    model = OfficialDinoPatchExtractor.__new__(OfficialDinoPatchExtractor)
    model.module = SimpleNamespace(torch=torch, F=F, dino_transform=transform)
    model.model = Model(); model.device = 'cpu'
    x = torch.arange(3*19*19).reshape(1, 3, 19, 19).remainder(256).to(torch.uint8)
    assert torch.equal(model.features_from_frames(x), model.features_from_frames(x.float()))
    a, ga = model.patches_from_frames(x)
    b, gb = model.patches_from_frames(x.float())
    assert ga == gb and torch.equal(a, b)
    assert seen == [torch.float32]*4


def test_decoded_rgb_cannot_silently_wrap_or_round_invalid_values():
    x = torch.zeros((1, 3, 2, 2))
    x[0, 0, 0, 0] = 255
    assert native_rgb_uint8(x)[0, 0, 0, 0] == 255
    for value in (-1., 256., float('nan'), .3):
        x[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match='exact byte values'):
            native_rgb_uint8(x)
