import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from subject_consistency.metric import evaluate_masked_batch, subject_phrase
from subject_consistency.models import NpzSubjectMaskProvider, SubjectMasks


class FakeExtractor:
    """Injected backbone: identical interface, no weights and no video decode."""

    def __init__(self, frames, patches, grid):
        self.frames = frames
        self.patches = patches
        self.grid = grid
        self.module = types.SimpleNamespace(load_video=lambda path: self.frames)

    def patches_from_frames(self, frames):
        if frames is not self.frames:
            raise AssertionError("extractor did not receive the shared decode")
        return self.patches, self.grid


class StaticProvider:
    def __init__(self, masks, present, source="static"):
        self.masks = masks
        self.present = present
        self.source = source
        self.calls = []

    def masks_for(self, video, frames, phrase):
        self.calls.append((str(video), phrase))
        return SubjectMasks(self.masks, self.present, phrase, self.source)


def make_features(subject, background):
    """[T, 4, 2] patches: tokens 0-1 are subject, tokens 2-3 are background."""
    return torch.tensor(
        [[subject[t], subject[t], background[t], background[t]] for t in range(len(subject))],
        dtype=torch.float32,
    )


class SubjectPhraseTests(unittest.TestCase):
    def test_top_level_field_wins(self):
        item = {"subject_en": "person", "dimension_metadata": {"subject_en": "other"}}
        self.assertEqual(subject_phrase(item, "subject_en"), "person")

    def test_nested_dimension_metadata_is_used(self):
        self.assertEqual(subject_phrase({"dimension_metadata": {"subject_en": "bicycle"}}, "subject_en"), "bicycle")

    def test_missing_phrase_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "subject_en"):
            subject_phrase({"prompt": "a person walking"}, "subject_en")


class MaskedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.video = Path("/tmp/clip.mp4")
        # frame 0/1: subject constant, background changes; frame 2: subject changes
        self.patches = make_features(
            [[1.0, 0.0]] * 3, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        self.frames = torch.zeros((3, 3, 8, 8), dtype=torch.uint8)
        self.grid = (2, 2)
        # grid row 0 holds the two subject tokens, row 1 the two background tokens
        self.masks = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]] * 3, dtype=torch.float32)
        self.metadata = {"clip.mp4": {"prompt": "a person walking", "subject_en": "person"}}

    def run_batch(self, provider):
        return evaluate_masked_batch(
            [self.video], self.metadata, "cpu", {"path": "/tmp/dino.pth"},
            provider, extractor=FakeExtractor(self.frames, self.patches, self.grid),
        )

    def test_masked_score_ignores_the_background_change(self):
        provider = StaticProvider(self.masks, torch.ones((3, 1), dtype=torch.bool))
        result = self.run_batch(provider)[0]
        self.assertEqual(result["status"], "succeeded")
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(result["diagnostics"]["patch_grid"], [2, 2])
        self.assertEqual(result["diagnostics"]["subject_phrase"], "person")

    def test_provider_receives_the_shared_decode_and_the_phrase(self):
        provider = StaticProvider(self.masks, torch.ones((3, 1), dtype=torch.bool))
        self.run_batch(provider)
        self.assertEqual(provider.calls, [("/tmp/clip.mp4", "person")])

    def test_mask_frame_mismatch_fails_the_sample(self):
        provider = StaticProvider(self.masks[:2], torch.ones((2, 1), dtype=torch.bool))
        result = self.run_batch(provider)[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("mask frames", result["error"])

    def test_missing_phrase_fails_the_sample(self):
        self.metadata = {"clip.mp4": {"prompt": "a person walking"}}
        result = self.run_batch(StaticProvider(self.masks, torch.ones((3, 1), dtype=torch.bool)))[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("subject_en", result["error"])

    def test_frame_cap_selects_uniform_indices(self):
        provider = StaticProvider(self.masks, torch.ones((3, 1), dtype=torch.bool))
        result = evaluate_masked_batch(
            [self.video], self.metadata, "cpu", {"path": "/tmp/dino.pth"}, provider,
            extractor=FakeExtractor(self.frames, self.patches, self.grid), max_frames=2,
        )[0]
        self.assertEqual(result["diagnostics"]["num_frames"], 2)
        self.assertEqual(result["diagnostics"]["max_frames"], 2)

    def test_npz_provider_round_trip_is_hashable(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.npz"
            np.savez(path, masks=self.masks.numpy(), present=np.ones((3, 1), dtype=bool))
            provider = NpzSubjectMaskProvider(Path(tmp))
            result = self.run_batch(provider)[0]
            diagnostics = result["diagnostics"]
        self.assertEqual(result["status"], "succeeded")
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(diagnostics["mask_source"], str(path))
        self.assertEqual(len(diagnostics["mask_source_sha256"]), 64)

    def test_npz_provider_rejects_a_frame_count_mismatch(self):
        with TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "clip.npz", masks=self.masks[:2].numpy())
            provider = NpzSubjectMaskProvider(Path(tmp))
            result = self.run_batch(provider)[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("mask frames", result["error"])


class MissingEvidencePolicyTests(unittest.TestCase):
    def test_zero_policy_is_the_scored_default_and_penalises_missing(self):
        video = Path("/tmp/clip.mp4")
        patches = make_features([[1.0, 0.0]] * 3, [[1.0, 0.0]] * 3)
        frames = torch.zeros((3, 3, 8, 8), dtype=torch.uint8)
        masks = torch.zeros((3, 1, 2, 2), dtype=torch.float32)
        masks[0, 0, 0, 0] = 1.0
        masks[1, 0, 0, 0] = 1.0
        metadata = {"clip.mp4": {"prompt": "x", "subject_en": "person"}}
        result = evaluate_masked_batch(
            [video], metadata, "cpu", {"path": "/tmp/dino.pth"},
            StaticProvider(masks, torch.ones((3, 1), dtype=torch.bool)),
            extractor=FakeExtractor(frames, patches, (2, 2)),
        )[0]
        self.assertEqual(result["diagnostics"]["missing_policy"], "zero")
        self.assertAlmostEqual(result["diagnostics"]["missing_fraction"], 1 / 3)
        self.assertAlmostEqual(result["score"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
