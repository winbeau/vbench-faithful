import numpy as np

from scripts.counterfactual.probe_region_time_modes import compare_predictions


def test_comparison_uses_common_observed_pairs_and_never_imputes_missing_zero():
    truth = np.ones((3, 2, 2))
    baseline = np.zeros_like(truth)
    candidate = truth.copy(); candidate[:, 0] = np.nan
    rows = compare_predictions(truth, np.ones((3, 2), bool), {0: baseline, 1: candidate}, np.arange(4.))
    assert rows[0]["testable_point_pairs"] == 6
    assert rows[1]["testable_point_pairs"] == 3
    assert rows[1]["total_point_pairs"] == 6
    assert rows[0]["common_point_pairs"] == 3
    assert rows[1]["improvement_over_heldout_constant"] == 1
    assert rows[0]["squared_error_on_common"] == 6
