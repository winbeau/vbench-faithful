import numpy as np
import pytest

from scripts.counterfactual.probe_sparse_identity import main, rank_pair
from scripts.counterfactual.summarize_sparse_identity import method_counts


def test_duplicate_methods_fail_before_opening_sources():
    args = []
    for name in ("manifest", "review", "native-run", "region-run", "output"):
        args.extend(["--" + name, "absent"])
    with pytest.raises(SystemExit) as exc:
        main(args + ["--methods", "sift", "sift"])
    assert exc.value.code == 2


def test_raw_and_temporal_witnesses_are_distinct_missing_is_not_zero():
    mask = np.ones((1, 5, 5), bool)
    region = {"region": 0, "area_pixels": 25, "whole_frame_control": False,
              "hypotheses": [{"displacement_pixels": [1, 0], "alignment_target_region": 0}]}
    matches = [{"source_key": 1, "target_key": 7, "source_xy": [2, 2], "target_xy": [3, 2]}]
    support = [{"source_key": 1, "target_key": 7, "supported_without_conflict": False}]
    result = rank_pair({"regions": [region]}, mask, mask, matches, support)[0]
    assert result["raw_ranking"]["best_compatible_hypothesis"] == 0
    assert result["temporal_ranking"]["best_compatible_hypothesis"] is None
    assert result["raw_ranking"]["score"] is result["temporal_ranking"]["score"] is None
    data = {"pairs": [{"matches": matches, "identity_support": [{**support[0], "conflicting_frames": []}],
                        "regions": [result]}]}
    counts = method_counts(data)
    assert counts["region_pairs"] == counts["unknown_matches"] == 1
    assert counts["raw"]["ranked"] == counts["temporal"]["unranked"] == 1
    assert counts["identity_supported_matches"] == 0
    data["pairs"][0]["identity_support"][0]["target_key"] = 9
    with pytest.raises(ValueError):
        method_counts(data)
