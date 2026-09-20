import copy

import pytest

from scripts.counterfactual.region_discrimination import LEVELS, POSITIONS
from scripts.counterfactual.subject_region_statistics import BACKENDS, analyze, paired_bootstrap


def experiment():
    index = [{"base_id": f"b{i}", "source_prompt_id": f"p{i}", "rejection_reasons": []} for i in range(3)]
    scores = []
    for base in index:
        for position in POSITIONS:
            for backend in BACKENDS:
                for level in LEVELS:
                    values = {"clean": .9, "subject_corrupt": .7,
                              "background_corrupt": .89 if backend.startswith("masked") else .7}
                    scores.append({"base_id": base["base_id"], "position": position, "level": level,
                                   "backend": backend, "score": values[level], "status": "succeeded"})
    return index, scores


def test_common_cohort_paired_bootstrap_and_background_changes():
    index, scores = experiment()
    report = analyze(index, scores, resamples=100)
    result = report["primary"]
    assert result["n_bases"] == 3
    assert result["absolute_change"]["masked_zero"]["median"] == pytest.approx(.01)
    assert result["absolute_change"]["official_minus_masked"]["ci95"] == pytest.approx([.19, .19])
    assert result["background_stability_criterion_met"] is True
    ratio = report["positions"]["middle"]["comparisons"]["zero"]
    assert ratio["statistics"]["masked_zero"]["R"] == pytest.approx(.05)
    assert ratio["equal_area_criterion"].startswith("NOT APPLICABLE")


def test_failures_remain_in_fixed_denominator_and_common_ratio_gates():
    index, scores = experiment()
    for row in scores:
        if row["base_id"] == "b0" and row["backend"] == "masked_zero":
            row.update(status="unsupported", score=None)
        if row["base_id"] == "b1" and row["backend"] == "official" and row["level"] == "subject_corrupt":
            row["score"] = .89
    report = analyze(index, scores, resamples=100)
    full = report["positions"]["full"]
    aggregate = full["aggregates"]["masked_zero"]["clean"]
    assert aggregate["fixed_denominator_mean"] == pytest.approx(.6)
    assert aggregate["fixed_denominator"] == 3
    assert aggregate["conditional_mean"] == pytest.approx(.9)
    assert full["comparisons"]["zero"]["common_base_ids"] == ["b2"]
    assert full["comparisons"]["zero"]["statistics"]["official"]["ci95"] is None
    assert full["exclusion_counts"]["official"] == {"subject_drop_below_0.05": 1}


def test_negative_background_drop_is_preserved():
    index, scores = experiment()
    for row in scores:
        if row["backend"] == "masked_zero" and row["level"] == "background_corrupt":
            row["score"] = .95
    report = analyze(index, scores, resamples=100)
    assert report["positions"]["full"]["drops"]["masked_zero"]["b0"]["ratio"] == pytest.approx(-.25)
    assert report["primary"]["absolute_change"]["masked_zero"]["median"] == pytest.approx(.05)


def test_incomplete_duplicate_and_nonfinite_scores_cannot_silently_disappear():
    index, scores = experiment()
    with pytest.raises(ValueError, match="incomplete"):
        analyze(index, scores[:-1], resamples=100)
    with pytest.raises(ValueError, match="duplicate"):
        analyze(index, scores + scores[:1], resamples=100)
    scores[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="invalid succeeded"):
        analyze(index, scores, resamples=100)


def test_single_prompt_is_one_cluster_even_with_many_videos():
    result = paired_bootstrap({"a": {"v1": .1, "v2": .2, "v3": .3}},
                              {"v1": "p1", "v2": "p1", "v3": "p1"}, resamples=100)
    assert result["a"]["n_prompt_clusters"] == 1
    assert result["a"]["ci95"] is None
