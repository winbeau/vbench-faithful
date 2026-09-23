from inspect import signature

import cv2
import numpy as np
import pytest

from dynamic_degree import dense_correspondence as dense
from dynamic_degree.dense_correspondence import score_dense_evidence as score_evidence


def fixture_translation(delta=2):
    image = np.random.default_rng(7).integers(30, 210, (96, 96, 3), dtype=np.uint8)
    frames = np.stack([cv2.warpAffine(image, np.float32([[1, 0, t * delta], [0, 1, 0]]),
                                     (96, 96), borderMode=cv2.BORDER_REFLECT) for t in range(4)])
    forward = np.zeros((3, 96, 96, 2), np.float32)
    forward[..., 0] = delta
    return frames, forward, -forward


def test_true_translation_is_verified_with_bidirectional_flow():
    frames, forward, backward = fixture_translation()
    config = dense.DenseCorrespondenceConfig(grid_size=4, blur_sigma=2, illumination_sigma=6)
    evidence = dense.dense_evidence(frames, forward, backward, config)
    result = score_evidence(evidence, np.arange(4) / 8, 96, 96, config)
    assert result["status"] == "succeeded"
    assert result["score"] == pytest.approx(16 / 96)
    assert result["coverage"] == 1


def test_wrong_reverse_direction_fails_correspondence_not_magic_zero():
    frames, forward, backward = fixture_translation()
    config = dense.DenseCorrespondenceConfig(grid_size=4, blur_sigma=2, illumination_sigma=6)
    evidence = dense.dense_evidence(frames, forward, -backward, config)
    assert np.all(evidence["cycle_error"] == 4)
    result = score_evidence(evidence, np.arange(4) / 8, 96, 96, config)
    assert result["score"] is None and result["status"] == "insufficient_evidence"


def test_real_static_evidence_counts_zero_without_any_motion_label():
    frames, forward, backward = fixture_translation(delta=0)
    config = dense.DenseCorrespondenceConfig(grid_size=4)
    evidence = dense.dense_evidence(frames, forward, backward, config)
    result = score_evidence(evidence, np.arange(4) / 8, 96, 96, config)
    assert result["score"] == 0 and result["coverage"] == 1


def test_out_of_frame_warp_has_no_fabricated_patch_support():
    frames, forward, backward = fixture_translation(delta=100)
    evidence = dense.dense_evidence(frames, forward, backward, dense.DenseCorrespondenceConfig(grid_size=4))
    assert not evidence["inside_pair"].any() and not evidence["visible_pair"].any()


def test_scalar_and_vector_field_sampling_shapes():
    field = np.zeros((4, 5, 2), np.float32)
    field[..., 0] = 2
    xy = np.array([[1, 1], [2, 2]], np.float32)
    assert dense.sample_field(field, xy).shape == (2, 2)
    assert dense.sample_field(field[..., 0], xy).tolist() == [2, 2]


def test_nonfinite_or_wrong_geometry_is_not_evidence():
    frames, forward, backward = fixture_translation()
    with pytest.raises(ValueError, match="shape"):
        dense.dense_evidence(frames, forward[:1], backward, dense.DenseCorrespondenceConfig())
    forward[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        dense.dense_evidence(frames, forward, backward, dense.DenseCorrespondenceConfig())


def test_filename_invariant_and_no_construction_inputs(monkeypatch, tmp_path):
    frames, _, _ = fixture_translation(delta=0)
    monkeypatch.setattr(dense, "decode_video", lambda *a: (frames, np.arange(4) / 8, {}))

    class Flow:
        def compute_flow(self, a, b):
            return np.zeros((*a.shape[:2], 2), np.float32)

    evaluator = dense.DenseCorrespondenceEvaluator(dense.DenseCorrespondenceConfig(grid_size=4),
                                                    tmp_path, tmp_path, "cpu", flow_model=Flow())
    a = evaluator.evaluate_video(tmp_path / "base.mp4")
    b = evaluator.evaluate_video(tmp_path / "8px_1701.mp4", evidence_path=tmp_path / "e.npz")
    assert a.pop("video") != b.pop("video") and a == b
    assert set(signature(evaluator.evaluate_video).parameters) == {"video", "evidence_path"}
    with np.load(tmp_path / "e.npz") as evidence:
        assert evidence["forward"].shape == (3, 96, 96, 2)
        assert "correspondence_ablation_not_jitter_repair" in a["interpretation"]
