"""Batched patch extraction preserves the final short batch and frame order."""
from types import SimpleNamespace

import torch

from subject_consistency.models import OfficialDinoPatchExtractor


def test_rectangular_patch_grid_and_order_match_single_frame_path():
    calls = []

    def intermediate(images, n):
        assert n == 1 and images.dtype == torch.float32
        calls.append(len(images))
        count = (images.shape[-2] // 16) * (images.shape[-1] // 16)
        frame_values = images.mean((1, 2, 3)).reshape(-1, 1, 1)
        return [frame_values.expand(-1, count + 1, 4)]

    extractor = OfficialDinoPatchExtractor.__new__(OfficialDinoPatchExtractor)
    extractor.module = SimpleNamespace(torch=torch, dino_transform=lambda _: lambda image: image)
    extractor.model = SimpleNamespace(patch_embed=SimpleNamespace(patch_size=16), get_intermediate_layers=intermediate)
    extractor.device = "cpu"
    frames = torch.arange(19).reshape(-1, 1, 1, 1).expand(-1, 3, 32, 48).to(torch.uint8)
    expected, grid = extractor.patches_from_frames(frames)
    assert calls == [1] * 19 and grid == (2, 3)
    calls.clear()
    extractor.frame_batch_size = 16
    actual, actual_grid = extractor.patches_from_frames(frames)
    assert calls == [16, 3]
    assert torch.equal(actual, expected) and actual_grid == grid
    assert actual.shape == (19, 6, 4) and extractor.transformed_size == (32, 48)
