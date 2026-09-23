import hashlib

import numpy as np

from dynamic_degree.local_appearance import feature_support
from scripts.counterfactual.probe_appearance_context import inspect_pair


def test_all_region_refs_and_zero_control_survive_deduplicated_geometry():
    frame = np.random.default_rng(5).integers(0, 255, (48, 48, 3), np.uint8)
    mask = feature_support(np.ones((48, 48), bool), [24., 24.], 2., 4.)
    key = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
    refs = [{"region": i, "factor": 4, "support_sha256": key, "whole_frame_control": i == 1} for i in range(2)]
    pair = {"start": 0, "lag": 1, "seconds": .125, "source_points": [{"xy": [24., 24.], "size": 2., "regional_supports": refs}],
            "supports": {key: {"hypotheses": [{"displacement_pixels": [0, 0], "correlation": 1.}]}}}
    record, arrays = inspect_pair(np.stack([frame, frame]), np.ones((1, 48, 48), bool), pair)
    assert len(record["groups"]) == 1 and len(record["groups"][0]["references"]) == 2
    assert record["groups"][0]["zero_control_index"] == 0 and record["groups"][0]["candidate_count"] == 1
    assert arrays["displacements"].tolist() == [[0, 0]] and record["score"] is None


def test_empty_queries_remain_auditable_and_have_no_score():
    frames = np.full((2, 24, 24, 3), 128, np.uint8)
    pair = {"start": 0, "lag": 1, "seconds": .125, "source_points": [], "supports": {}}
    record, arrays = inspect_pair(frames, np.zeros((0, 24, 24), bool), pair)
    assert record["groups"] == [] and record["score"] is None and arrays["displacements"].shape == (0, 2)
