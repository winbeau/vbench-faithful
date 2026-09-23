import pytest

from scripts.counterfactual.probe_appearance_consensus import inspect_pair


def test_all_declared_regions_and_both_weighting_ablations_are_kept():
    evidence = {"hypotheses": [{"displacement_pixels": [3, 4], "correlation": .9}], "reverse_check": "NOT RUN",
                "self_alternatives": [{"maximum_support_intersection": .5, "hypotheses": [{"correlation": .6}]}]}
    pair = {"start": 2, "lag": 1, "seconds": .125, "supports": {"a": evidence},
            "source_points": [{"xy": [10., 10.], "regional_supports": [
                {"region": 0, "factor": 4, "whole_frame_control": False, "support_sha256": "a"},
                {"region": 1, "factor": 4, "whole_frame_control": True, "support_sha256": "a"}]}]}
    result = inspect_pair(pair, (32, 32))
    assert len(result["regions"]) == 2
    assert [a["power"] for a in result["regions"][0]["ablations"]] == [1, 2]
    assert all(a["score"] is None for region in result["regions"] for a in region["ablations"])
    pair["supports"]["a"]["reverse_check"] = "copied from another proposal"
    with pytest.raises(ValueError, match="reverse"):
        inspect_pair(pair, (32, 32))
