from scripts.counterfactual.audit_local_appearance import support_statistics


def test_missing_not_zero_and_alternative_reverse_not_best_reverse():
    supports = [{"hypotheses": []}, {"hypotheses": [
        {"displacement_pixels": [3, 4], "correlation": .8,
         "reverse_candidates": [{"closure_error_pixels": 12.}, {"closure_error_pixels": 0.}]},
        {"displacement_pixels": [0, 0], "correlation": .7}]}]
    stats = support_statistics(supports)
    assert stats["source_points"] == 2
    assert stats["missing"] == 1
    assert stats["top1_mean_length_pixels"] == 5
    assert stats["best_reverse_exact_closure"] == 0
    assert stats["any_reverse_exact_closure"] == 1


def test_unobserved_displacements_are_not_static_motion():
    stats = support_statistics([{"hypotheses": []}])
    assert stats["top1_mean_length_pixels"] is None
    assert stats["top1_mean_displacement_pixels"] is None
