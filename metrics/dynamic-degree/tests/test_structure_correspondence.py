import numpy as np
import pytest

from dynamic_degree.structure_correspondence import (
    match_similarity, pair_diagnostics, peak_refinement, transitivity_diagnostics,
)


def grid(h=5, w=5):
    y, x = np.mgrid[:h, :w]
    return np.stack((x, y), axis=-1).reshape(-1, 2).astype(float)


def test_quadratic_peak_recovers_subpixel_not_quantized_zero():
    q = grid()
    peak = np.array([2.3, 1.8])
    s = -np.sum((q - peak) ** 2, axis=-1)[None]
    offset, resolved, curvature = peak_refinement(s, s.argmax(axis=1), [5, 5])
    assert resolved[0]
    assert np.allclose(q[s.argmax()] + offset[0], peak)
    assert curvature[0] == pytest.approx(2.)


def test_flat_and_boundary_peaks_are_unresolved_not_certified_still():
    s = np.zeros((1, 25))
    _, resolved, _ = peak_refinement(s, np.array([12]), [5, 5])
    assert not resolved[0]
    s[0, 0] = 1
    _, resolved, _ = peak_refinement(s, np.array([0]), [5, 5])
    assert not resolved[0]


def test_identical_descriptors_do_not_get_false_mutual_coverage():
    q = grid()
    r = match_similarity(np.ones((25, 25)), q, [5, 5])
    assert not r["mutual"].any()
    d = pair_diagnostics(r, q, np.ones(25), .125, 5.)
    assert d["mutual_conditional_speed"] is None
    assert d["mutual_resolved_fraction"] == 0.
    assert "score" not in d


def test_unique_identity_matches_have_explicit_coarse_zero():
    q = grid()
    r = match_similarity(np.eye(25), q, [5, 5])
    assert r["mutual"].all()
    assert r["subpixel_resolved"].sum() == 9
    d = pair_diagnostics(r, q, np.ones(25), .25, 5.)
    assert d["coarse_zero_fraction"] == 1
    assert d["mutual_conditional_speed"] == 0
    assert d["mutual_resolved_fraction"] == pytest.approx(9 / 25)


def test_rectangular_geometry_and_pixel_scaling():
    q = grid(4, 6) * [2., 3.]
    s = np.eye(24)[:, np.roll(np.arange(24), 1)]
    r = match_similarity(s, q, [4, 6])
    d = pair_diagnostics(r, q, np.ones(24), .125, 12.)
    other = pair_diagnostics(match_similarity(s, q * 4, [4, 6]), q * 4, np.ones(24), .125, 48.)
    assert d["all_refined_or_coarse_speed"] == pytest.approx(other["all_refined_or_coarse_speed"])
    assert r["mutual"].all()


def test_transitivity_failure_and_missingness_are_not_dropped():
    q = grid()
    a = match_similarity(np.eye(25), q, [5, 5])
    b = match_similarity(np.eye(25)[:, ::-1], q, [5, 5])
    r = transitivity_diagnostics(a, a, b, q)
    assert r["all_mean_error_pixels"] > 0
    assert r["mutual_triple_fraction"] == 1
    b["mutual"][:] = False
    assert transitivity_diagnostics(a, a, b, q)["mutual_conditional_mean_error_pixels"] is None


def test_asymmetric_neighbors_cannot_make_identical_features_move():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(25, 8)); f /= np.linalg.norm(f, axis=1, keepdims=True)
    q = grid()
    r = match_similarity(f @ f.T, q, [5, 5])
    assert r["mutual"].all()
    assert np.linalg.norm(r["oneway_target_xy"] - q, axis=-1).mean() > 0
    assert np.array_equal(r["target_xy"], q)


def test_reciprocal_refinement_is_time_reversal_antisymmetric():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(25, 16)); f /= np.linalg.norm(f, axis=1, keepdims=True)
    g = f + .05 * rng.normal(size=f.shape); g /= np.linalg.norm(g, axis=1, keepdims=True)
    q = grid()
    forward = match_similarity(f @ g.T, q, [5, 5])
    reverse = match_similarity(g @ f.T, q, [5, 5])
    keep = forward["reciprocal_subpixel_resolved"]
    j = forward["target_index"][keep]
    assert keep.any()
    assert np.allclose(forward["target_xy"][keep] - q[keep], -(reverse["target_xy"][j] - q[j]))


@pytest.mark.parametrize("kind", ["nonfinite", "unordered_grid", "wrong_geometry"])
def test_bad_evidence_is_rejected(kind):
    q, s = grid(), np.eye(25)
    if kind == "nonfinite": s[0, 0] = np.nan
    if kind == "unordered_grid": q = q[::-1]
    if kind == "wrong_geometry": s = s[:10]
    with pytest.raises(ValueError):
        match_similarity(s, q, [5, 5])
