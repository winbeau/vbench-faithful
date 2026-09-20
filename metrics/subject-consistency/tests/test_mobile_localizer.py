import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from subject_consistency.localizer import (MobileSamSubjectMaskProvider, deterministic_dense_position_encoding,
                                          prompt_sha256, validate_prompt)
from subject_consistency.subject_evidence import resample_masks_to_grid, subject_consistency_from_vectors


def prompt():
    value = {"video_uid": "v0", "phrase": "person", "role": "scoring", "prompt_source": "human_clean_frame",
             "human_confirmed": True, "reviewer": "fixture-reviewer", "confirmed_at": "2026-09-20T00:00:00Z",
             "source_frame_sha256": "a" * 64, "image_size": [32, 48], "box": [5, 6, 20, 25],
             "points": [], "point_labels": []}
    return {**value, "sha256": prompt_sha256(value)}


def test_prompt_must_be_frozen_human_confirmation_independent_of_construction():
    value = prompt()
    validate_prompt(value)
    value["box"][0] = 6
    with pytest.raises(ValueError, match="hash"):
        validate_prompt(value)
    for key, bad, error in (("human_confirmed", False, "human confirmation"),
                             ("prompt_source", "segformer_box", "human-reviewed")):
        value = prompt()
        value[key] = bad
        value["sha256"] = prompt_sha256(value)
        with pytest.raises(ValueError, match=error):
            validate_prompt(value)


def test_every_variant_is_relocalized_with_the_same_frozen_prompt():
    calls = []

    class FakePredictor:
        def set_image(self, image, image_format):
            self.image = image
            assert image_format == "RGB"

        def predict(self, **kwargs):
            calls.append(kwargs)
            mask = np.zeros((1, 32, 48), dtype=bool)
            if self.image.any():
                mask[:, 6:25, 5:20] = True
            return mask, np.ones(1), None

    provider = MobileSamSubjectMaskProvider({"v0": prompt()}, clip_to_video_uid={"clean": "v0", "corrupt": "v0"}, predictor=FakePredictor())
    clean = provider.masks_for(Path("clean"), torch.ones(3, 3, 32, 48, dtype=torch.uint8), "person")
    corrupt = provider.masks_for(Path("corrupt"), torch.zeros(3, 3, 32, 48, dtype=torch.uint8), "person")
    assert len(calls) == 6
    assert clean.instance_present.all()
    assert not corrupt.instance_present.any()
    assert all(np.array_equal(call["box"], [5, 6, 20, 25]) for call in calls)
    assert all(call["multimask_output"] is False for call in calls)
    assert clean.source_sha256 == corrupt.source_sha256


def test_mask_projection_excludes_the_unrepresented_dino_border():
    masks = torch.zeros(1, 224, 239)
    masks[:, :, 224:] = 1
    result = resample_masks_to_grid(masks, 14, 14, image_size=(224, 239))
    assert torch.count_nonzero(result) == 0


def test_one_present_frame_zero_scores_zero_without_dropping_the_clip():
    result = subject_consistency_from_vectors(torch.ones(4, 1), torch.tensor([True, False, False, False]))
    assert result.score == 0
    assert result.pair_denominator == 6


@pytest.mark.parametrize("size", [(64, 64), (7, 11), (1, 1)])
def test_dense_position_encoding_matches_upstream_cumsum_coordinates(size):
    layer = SimpleNamespace(positional_encoding_gaussian_matrix=torch.zeros(2, 2), _pe_encoding=lambda x: x)
    actual = deterministic_dense_position_encoding(layer, size)
    grid = torch.ones(size)
    expected = torch.stack([(grid.cumsum(1) - .5) / size[1],
                            (grid.cumsum(0) - .5) / size[0]], dim=-1).permute(2, 0, 1)
    assert torch.equal(actual, expected)
