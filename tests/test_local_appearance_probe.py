import json

import numpy as np
import pytest

from scripts.counterfactual.probe_local_appearance import feature_groups, probe_pair, main


def test_orientation_duplicates_share_a_spatial_scale_but_distinct_scales_remain():
    features = {"xy": np.array([[2, 3], [2, 3], [2, 3]]), "scale": np.array([2, 2, 4])}
    groups = feature_groups(features)
    assert [p["source_keys"] for p in groups] == [[0, 1], [2]]


def test_every_region_and_scale_retained_duplicate_supports_reused():
    rng = np.random.default_rng(42)
    source = rng.integers(10, 240, (20, 24, 3), np.uint8)
    masks = np.ones((2, 20, 24), bool)
    features = {"xy": np.array([[12, 10]]), "scale": np.array([2.])}
    result = probe_pair(source, source, masks, features)
    assert len(result["source_points"][0]["regional_supports"]) == 9
    assert len(result["supports"]) == 3
    assert result["whole_frame_control_region"] == 2
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    for support in result["supports"].values():
        assert support["hypotheses"][0]["displacement_pixels"] == [0, 0]
        assert support["score"] is None


def test_invalid_phase_plan_fails_before_inputs():
    args = ["--manifest", "absent", "--review", "absent", "--region-run", "absent", "--output", "absent",
            "--candidates", "x", "--starts", "2", "2"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_ambiguity_variant_keeps_every_region_at_declared_factor_without_reverse_claim():
    rng = np.random.default_rng(12)
    frame = rng.integers(10, 240, (24, 28, 3), np.uint8)
    features = {"xy": np.array([[14, 12]]), "scale": np.array([2.])}
    result = probe_pair(frame, frame, np.ones((2, 24, 28), bool), features, search="ambiguity", factors=(4,))
    refs = result["source_points"][0]["regional_supports"]
    assert len(refs) == 3 and all(r["factor"] == 4 for r in refs)
    assert len(result["supports"]) == 1
    support = next(iter(result["supports"].values()))
    assert support["reverse_check"] == "NOT RUN" and support["score"] is None
    assert support["hypotheses"][0]["displacement_pixels"] == [0, 0]
    assert len(support["self_alternatives"]) == 3
