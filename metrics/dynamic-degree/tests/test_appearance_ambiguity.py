import cv2
import numpy as np
import pytest

from dynamic_degree.local_appearance import appearance_ambiguity, correlation_peaks, distinct_correlation_peaks


def surface(values):
    y, x = np.indices(values.shape)
    return {"correlation": values, "dx": x, "dy": y, "overlap": np.ones_like(values),
            "source_variance": np.ones_like(values), "target_variance": np.ones_like(values)}


def test_distinct_peaks_do_not_spend_budget_on_a_broad_peak_shoulder():
    y, x = np.indices((21, 61))
    values = np.maximum(np.exp(-((x - 12) ** 2 + (y - 10) ** 2) / 100),
                        .8 * np.exp(-((x - 47) ** 2 + (y - 10) ** 2) / 10))
    old = correlation_peaks(surface(values), 3)
    assert all(p["displacement_pixels"][0] < 25 for p in old)
    new = distinct_correlation_peaks(surface(values), 3)
    assert [p["displacement_pixels"] for p in new] == [[12, 10], [47, 10]]
    assert all(p["reverse_candidates"] == [] for p in new)


def test_plateau_is_one_hypothesis_and_exclusion_does_not_create_a_peak():
    values = np.zeros((20, 20)); values[5:8, 7:11] = 1.
    peaks = distinct_correlation_peaks(surface(values), 2)
    assert peaks[0]["plateau_pixels"] == 12 and peaks[0]["displacement_pixels"] == [7, 5]
    y, x = np.indices((20, 20))
    values = np.exp(-((x - 10) ** 2 + (y - 10) ** 2) / 15)
    assert distinct_correlation_peaks(surface(values), allowed=(x - 10) ** 2 + (y - 10) ** 2 > 20) == []


def test_unique_source_support_relocates_without_hidden_reverse_certification():
    rng = np.random.default_rng(4)
    source = rng.integers(0, 255, (48, 48, 3), np.uint8)
    target = cv2.warpAffine(source, np.array([[1., 0., 7.], [0., 1., -5.]]), (48, 48))
    support = np.zeros((48, 48), bool); support[18:30, 18:30] = True
    result = appearance_ambiguity(source, target, support)
    assert result["hypotheses"][0]["displacement_pixels"] == [7, -5]
    assert result["self_identity_correlation"] == pytest.approx(1.)
    assert all(a["cross_top1_minus_self_alternative"] > .5 for a in result["self_alternatives"])
    assert result["score"] is None and result["reverse_check"] == "NOT RUN"


def test_periodic_source_reveals_distinct_same_frame_lookalikes():
    rng = np.random.default_rng(5)
    tile = rng.integers(0, 255, (8, 8, 3), np.uint8)
    source = np.tile(tile, (6, 6, 1))
    support = np.zeros((48, 48), bool); support[18:30, 18:30] = True
    result = appearance_ambiguity(source, source, support)
    assert all(a["hypotheses"][0]["correlation"] == pytest.approx(1.) for a in result["self_alternatives"])
    assert all(abs(a["cross_top1_minus_self_alternative"]) < 1e-10 for a in result["self_alternatives"])
    assert all(a["hypotheses"][0]["displacement_pixels"] != [0, 0] for a in result["self_alternatives"])


def test_flat_unobservable_patch_is_not_a_unique_static_match():
    frame = np.full((24, 24, 3), 100, np.uint8)
    result = appearance_ambiguity(frame, frame, np.ones((24, 24), bool))
    assert not result["hypotheses"] and result["self_identity_correlation"] is None
    assert result["score"] is None
