import copy
import hashlib

import numpy as np
import pytest

from dynamic_degree.appearance_paths import triplet_ncc
from dynamic_degree.local_appearance import feature_support
from scripts.counterfactual.audit_appearance_paths import common_check, verify_detected, verify_reference, verify_rankings
from scripts.counterfactual.probe_appearance_paths import inspect_triplet, inspect_reference_triplet


def pair(start, xy, displacement, shape):
    support = feature_support(np.ones(shape, bool), xy, 2., 4.)
    key = hashlib.sha256(np.packbits(support).tobytes()).hexdigest()
    return {"start": start, "lag": 1, "source_points": [{"xy": xy, "size": 2., "regional_supports": [
        {"region": 0, "factor": 4, "whole_frame_control": True, "support_sha256": key}]}],
        "supports": {key: {"hypotheses": [{"displacement_pixels": displacement, "correlation": 1.}]}}}


def test_detection_audit_rejects_changed_geometry_or_ranking():
    frame = np.random.default_rng(73).integers(0, 255, (40, 48, 3), np.uint8)
    first, second = pair(0, [20., 20.], [0, 0], (40, 48)), pair(1, [20., 20.], [0, 0], (40, 48))
    row, arrays = inspect_triplet(np.stack([frame] * 3), np.zeros((0, 40, 48), bool), first, second)
    verify_detected(arrays, row, first, second)
    assert verify_rankings(arrays, row, "detected") == [0]
    changed = copy.deepcopy(arrays); changed["first_offset"][0, 0] = 1.
    with pytest.raises(ValueError, match="offset mismatch"):
        verify_detected(changed, row, first, second)
    wrong = copy.deepcopy(row); wrong["rankings"][0]["ranked_path_indices"] = []
    with pytest.raises(ValueError, match="ranking changed"):
        verify_rankings(arrays, wrong, "detected")


def test_reference_audit_catches_cartesian_path_loss():
    frame = np.random.default_rng(12).integers(0, 255, (40, 48, 3), np.uint8)
    first, second = pair(0, [20., 20.], [0, 0], (40, 48)), pair(1, [20., 20.], [0, 0], (40, 48))
    row, arrays = inspect_reference_triplet(np.stack([frame] * 3), np.zeros((0, 40, 48), bool), first, second)
    verify_reference(arrays, row, first)
    assert len(verify_rankings(arrays, row, "reference")) == 1
    changed = copy.deepcopy(arrays); changed["reference_rank"][-1] = 0
    with pytest.raises(ValueError, match="Cartesian"):
        verify_reference(changed, row, first)


def test_literal_common_pixels_independently_match_streamed_covariance():
    frames = np.random.default_rng(73).integers(0, 255, (3, 40, 48, 3), np.uint8)
    support = np.zeros((40, 48), bool); support[10:30, 10:40] = True
    arrays, overlap = triplet_ncc(frames, support, [[2, -1]], [[-1, 2]])
    measured, fraction = common_check(frames.astype(float) / 255., support, [2, -1], [-1, 2])
    np.testing.assert_allclose(arrays[0], measured, atol=1e-12)
    assert overlap[0] == fraction


def test_no_automatic_queries_is_auditable_missing_evidence_not_zero_motion():
    frames = np.full((3, 24, 24, 3), 128, np.uint8)
    first = {"start": 0, "lag": 1, "source_points": [], "supports": {}}
    second = {**first, "start": 1}
    row, arrays = inspect_reference_triplet(frames, np.zeros((0, 24, 24), bool), first, second)
    verify_reference(arrays, row, first)
    assert verify_rankings(arrays, row, "reference") == []
    assert row["score"] is None and row["source_queries"] == row["candidate_paths"] == 0
