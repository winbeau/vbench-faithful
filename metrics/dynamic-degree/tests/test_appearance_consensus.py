import numpy as np

from dynamic_degree.local_appearance import consensus_hypotheses


def appearance(cross, alternative):
    return {"hypotheses": [{"displacement_pixels": d, "correlation": c} for d, c in cross],
            "self_alternatives": [{"maximum_support_intersection": p, "hypotheses": [{"correlation": alternative}]} for p in (.25, .5, .75)]}


def test_ambiguous_repeated_points_do_not_outvote_distinct_structure():
    repeated = [{"xy": [i + 10., 12.], "appearance": appearance([([0, 0], .99), ([-12, -4], .99)], .999)} for i in range(10)]
    markers = [{"xy": xy, "appearance": appearance([([-12, -4], .95), ([0, 0], .7)], .8)}
               for xy in ([20., 20.], [30., 21.], [21., 30.], [32., 32.])]
    result = consensus_hypotheses(repeated + markers, (64, 64), radius=0)
    assert result["zero_positive_margin_locations"] == 10
    assert result["folds"][0]["peaks"][0]["displacement_pixels"] == [-12, -4]
    assert result["folds"][0]["peaks"][0]["supporting_locations"] == 4
    assert result["score"] is None


def test_scales_and_duplicate_feature_orientations_are_not_extra_votes():
    entries = [{"xy": [20., 20.], "appearance": appearance([([3, 4], .95)], .8)},
               {"xy": [30., 30.], "appearance": appearance([([3, 4], .96)], .7)}]
    before = consensus_hypotheses(entries, (48, 48), radius=0)
    after = consensus_hypotheses(entries + [entries[0]], (48, 48), radius=0)
    assert before == after
    assert before["folds"][1]["source_locations"] + before["folds"][2]["source_locations"] == 2


def test_missing_appearance_never_produces_zero_motion_proposal():
    result = consensus_hypotheses([{"xy": [1., 2.], "appearance": {"hypotheses": [], "self_alternatives": []}}], (20, 20))
    assert result["missing_appearance_locations"] == 1
    assert all(not fold["peaks"] for fold in result["folds"])
    assert result["score"] is None
