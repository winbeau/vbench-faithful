import unittest

import torch

from subject_consistency.subject_evidence import (
    masked_subject_consistency,
    resample_masks_to_grid,
    sample_frame_indices,
    subject_consistency_from_vectors,
    subject_vectors,
)


def frame_features(subject, background):
    """[T, 2, 2] patches: token 0 is subject, token 1 is background."""
    return torch.tensor([[subject[t], background[t]] for t in range(len(subject))], dtype=torch.float64)


def subject_mask(frames, present=None):
    """[T, 1, 2] mask selecting token 0 (the subject) only."""
    weights = [1.0 if (present is None or present[t]) else 0.0 for t in range(frames)]
    return torch.tensor([[[1.0, 0.0]] if w else [[0.0, 0.0]] for w in weights], dtype=torch.float64)


class SamplingTests(unittest.TestCase):
    def test_short_clip_keeps_every_frame(self):
        self.assertEqual(sample_frame_indices(16, 32), tuple(range(16)))
        self.assertEqual(sample_frame_indices(16, None), tuple(range(16)))

    def test_capped_clip_uses_uniform_midpoints(self):
        self.assertEqual(sample_frame_indices(100, 4), (12, 37, 62, 87))

    def test_cap_below_two_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            sample_frame_indices(10, 1)


class MaskGridTests(unittest.TestCase):
    def test_full_mask_stays_full(self):
        masks = torch.ones((2, 4, 4))
        pooled = resample_masks_to_grid(masks, 2, 2)
        self.assertEqual(pooled.shape, (2, 4))
        self.assertTrue(torch.allclose(pooled, torch.ones((2, 4), dtype=pooled.dtype)))

    def test_partial_mask_becomes_coverage(self):
        masks = torch.zeros((1, 4, 4))
        masks[0, :2, :2] = 1.0
        pooled = resample_masks_to_grid(masks, 2, 2)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=pooled.dtype)))

    def test_out_of_range_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            resample_masks_to_grid(torch.full((1, 2, 2), 2.0), 1, 1)


class MaskedEvidenceTests(unittest.TestCase):
    def test_background_change_outside_the_mask_leaves_the_score_unchanged(self):
        """The discrimination contract: the score must be blind to background."""
        clean = frame_features([[1.0, 0.0]] * 4, [[1.0, 0.0]] * 4)
        changed = frame_features([[1.0, 0.0]] * 4, [[0.0, 1.0]] * 4)
        masks = subject_mask(4)
        self.assertAlmostEqual(masked_subject_consistency(clean, masks).score, 1.0)
        self.assertAlmostEqual(masked_subject_consistency(changed, masks).score, 1.0)

    def test_subject_change_inside_the_mask_lowers_the_score(self):
        clean = frame_features([[1.0, 0.0]] * 4, [[1.0, 0.0]] * 4)
        corrupted = frame_features([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0]] * 4)
        masks = subject_mask(4)
        self.assertLess(masked_subject_consistency(corrupted, masks).score, masked_subject_consistency(clean, masks).score)

    def test_whole_frame_evidence_would_fail_the_background_control(self):
        """Without the mask the same background edit moves the score."""
        clean = frame_features([[1.0, 0.0]] * 4, [[1.0, 0.0]] * 4)
        changed = frame_features([[1.0, 0.0]] * 4, [[0.0, 1.0]] * 4)
        everything = torch.ones((4, 1, 2), dtype=torch.float64)
        self.assertLess(masked_subject_consistency(changed, everything).score, masked_subject_consistency(clean, everything).score)

    def test_score_is_permutation_invariant(self):
        features = frame_features([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]] * 4)
        masks = subject_mask(4)
        order = [2, 0, 3, 1]
        self.assertAlmostEqual(
            masked_subject_consistency(features, masks).score,
            masked_subject_consistency(features[order], masks[order]).score,
        )

    def test_fewer_than_two_frames_is_explicit_error(self):
        features = frame_features([[1.0, 0.0]], [[1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            masked_subject_consistency(features, subject_mask(1))


class InstanceModeTests(unittest.TestCase):
    def test_union_mode_matches_amax_mask_pooling(self):
        features = frame_features([[1.0, 0.0]] * 3, [[0.0, 1.0]] * 3)
        masks = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 3, dtype=torch.float64)
        union_vectors, _ = subject_vectors(features, masks, mode="union")
        amax_vectors, _ = subject_vectors(features, masks.amax(dim=1, keepdim=True), mode="union")
        self.assertTrue(torch.allclose(union_vectors, amax_vectors))

    def test_mean_mode_ignores_absent_instances(self):
        features = frame_features([[1.0, 0.0]] * 3, [[0.0, 1.0]] * 3)
        masks = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 3, dtype=torch.float64)
        present = torch.tensor([[True, False]] * 3)
        mean_vectors, _ = subject_vectors(features, masks, present, mode="mean")
        single_vectors, _ = subject_vectors(features, masks[:, :1, :], mode="mean")
        self.assertTrue(torch.allclose(mean_vectors, single_vectors))

    def test_mean_mode_coverage_uses_present_instances_only(self):
        features = frame_features([[1.0, 0.0]] * 2, [[0.0, 1.0]] * 2)
        masks = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]] * 2, dtype=torch.float64)
        present = torch.tensor([[True, False]] * 2)
        _, coverage = subject_vectors(features, masks, present, mode="mean")
        self.assertTrue(torch.allclose(coverage, torch.full((2,), 0.5, dtype=torch.float64)))


class MissingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.vectors = torch.tensor([[1.0, 0.0]] * 4, dtype=torch.float64)
        self.present = torch.tensor([True, True, False, True])

    def test_zero_policy_penalises_and_keeps_the_denominator(self):
        diagnostics = subject_consistency_from_vectors(self.vectors, self.present, missing_policy="zero")
        self.assertEqual(diagnostics.pair_count, 6)
        self.assertEqual(diagnostics.pair_denominator, 6)
        self.assertAlmostEqual(diagnostics.score, 0.5)
        self.assertAlmostEqual(diagnostics.missing_fraction, 0.25)

    def test_exclude_policy_is_conditioned_on_present_frames(self):
        diagnostics = subject_consistency_from_vectors(self.vectors, self.present, missing_policy="exclude")
        self.assertEqual(diagnostics.pair_count, 3)
        self.assertAlmostEqual(diagnostics.score, 1.0)

    def test_carry_policy_fills_from_neighbours(self):
        vectors = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        diagnostics = subject_consistency_from_vectors(vectors, self.present, missing_policy="carry")
        self.assertEqual(diagnostics.pair_count, 6)
        self.assertAlmostEqual(diagnostics.score, 0.5)

    def test_all_frames_missing_zero_keeps_the_fixed_denominator(self):
        diagnostics = subject_consistency_from_vectors(self.vectors, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(diagnostics.score, 0.0)
        self.assertEqual(diagnostics.pair_denominator, 6)
        self.assertEqual(diagnostics.num_missing_frames, 4)

    def test_all_frames_missing_exclude_is_undefined(self):
        with self.assertRaisesRegex(ValueError, "fewer than two frames"):
            subject_consistency_from_vectors(self.vectors, torch.zeros(4, dtype=torch.bool), missing_policy="exclude")

    def test_negative_cosine_is_clamped(self):
        vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
        self.assertEqual(subject_consistency_from_vectors(vectors).score, 0.0)

    def test_coverage_length_must_match(self):
        with self.assertRaisesRegex(ValueError, "one entry per frame"):
            subject_consistency_from_vectors(self.vectors, self.present, coverage=(1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
