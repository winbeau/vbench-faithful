import pytest

from vbench_audit_core.score_agreement import compare_scores


def rows(*scores):
    return [dict(id=str(i), video=f"{i}.mp4", video_sha256=str(i), prompt="a", score=score, status="succeeded")
            for i, score in enumerate(scores)]


def test_individual_error_is_not_hidden_by_equal_means():
    report = compare_scores(rows(1, 1), rows(.98, 1.02))
    assert not report["passed"] and all(not r["passed"] for r in report["rows"])
    assert compare_scores(rows(1, -1), rows(1.009, -.991))["passed"]


def test_near_zero_absolute_tolerance_and_order_independence():
    assert compare_scores(rows(0., 1e-8), list(reversed(rows(1e-6, 9e-7))))["passed"]
    assert not compare_scores(rows(0.), rows(1.0001e-6))["passed"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "sha", "prompt", "status", "null", "nan"])
def test_incomplete_or_changed_population_cannot_pass(change):
    reference, candidate = rows(.5, .7), rows(.5, .7)
    if change == "duplicate":
        candidate.append(candidate[0])
    elif change == "missing":
        candidate.pop()
    elif change == "sha":
        candidate[0]["video_sha256"] = "other"
    elif change == "prompt":
        candidate[0]["prompt"] = "other"
    elif change == "status":
        candidate[0]["status"] = "failed"
    elif change == "null":
        candidate[0]["score"] = None
    elif change == "nan":
        candidate[0]["score"] = float("nan")
    with pytest.raises(ValueError):
        compare_scores(reference, candidate)
