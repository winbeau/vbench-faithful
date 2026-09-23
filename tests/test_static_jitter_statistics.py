import numpy as np
import pytest

from scripts.counterfactual.analyze_static_jitter import cluster_interval, ladder, get_score, calibrate


def test_cluster_ci_requires_more_than_one_prompt():
    assert cluster_interval([0, 0], ["same", "same"]) is None
    assert cluster_interval([0, 0], ["a", "b"]) == [0, 0]
    # Repeated interventions within one cluster must not imply tiny independent SE.
    ci = cluster_interval([0] * 50 + [1] * 50, ["a"] * 50 + ["b"] * 50)
    assert ci == [0, 1]


def test_missing_motion_scores_count_as_failed_ordering_not_dropped():
    rows = [{"candidate_id": f"video_{i}", "base_id": "base", "prompt_id": "p", "kind": "translation",
             "family": "clean", "amplitude": 0, "motion": i, "seed": 0,
             "status": "qualified", "sha256": str(i)} for i in range(4)]
    scored = {"video_0": {"input_sha256": "0", "repair": {"status": "succeeded", "score": 0}},
              "video_1": {"input_sha256": "1", "repair": {"status": "succeeded", "score": 1}}}
    result = ladder(rows, scored)
    assert result["expected_pairs"] == 3 and result["complete_pairs"] == 1
    assert result["strict_ordering"]["mean"] == pytest.approx(1 / 3)
    assert result["input_qualified"]["strict_ordering"]["mean"] == pytest.approx(1 / 3)


def test_input_rejection_excludes_whole_ladder_but_keeps_full_candidate_denominator():
    rows, scored = [], {}
    for source in ("valid", "rejected"):
        for level in range(4):
            key = f"{source}-{level}"
            rows.append({"candidate_id": key, "base_id": source, "prompt_id": source,
                         "kind": "translation", "family": "clean", "amplitude": 0,
                         "motion": level, "seed": 0, "sha256": key,
                         "status": "rejected" if source == "rejected" and level == 3 else "qualified"})
            scored[key] = {"input_sha256": key, "repair": {"status": "succeeded", "score": level}}
    result = ladder(rows, scored)
    assert result["expected_pairs"] == 6
    assert result["strict_ordering"]["mean"] == pytest.approx(5 / 6)
    assert result["input_qualified"]["expected_pairs"] == 3
    assert result["input_qualified"]["construction_excluded_ladders"] == 1
    assert result["input_qualified"]["strict_ordering"]["mean"] == 1


def test_input_identity_checked_and_null_not_zero():
    row = {"candidate_id": "a", "sha256": "right", "status": "qualified"}
    assert get_score(row, {}, "repair") is None
    with pytest.raises(ValueError, match="hash mismatch"):
        get_score(row, {"a": {"input_sha256": "wrong"}}, "repair")


def test_cannot_calibrate_on_test_set():
    with pytest.raises(ValueError, match="development-only"):
        calibrate([{"split": "test"}], {}, {})
