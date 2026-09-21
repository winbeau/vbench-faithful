import pytest

from background_consistency.calibration import balanced_patch_score, calibrate_patch_score


def test_gain_expands_both_nuisance_and_background_response_equally():
    clean, foreground, background = .99, .985, .95
    a, b, c = map(calibrate_patch_score, (clean, foreground, background))
    assert a-b == pytest.approx(2*(clean-foreground))
    assert a-c == pytest.approx(2*(clean-background))
    assert (a-b)/(a-c) == pytest.approx((clean-foreground)/(clean-background))
    assert a > b > c


def test_calibration_preserves_zero_evidence_and_bounds():
    assert calibrate_patch_score(0) == 0
    assert calibrate_patch_score(.4) == 0
    assert calibrate_patch_score(1.0001) == 1
    with pytest.raises(ValueError):
        calibrate_patch_score(float('nan'))
    with pytest.raises(ValueError):
        calibrate_patch_score(.5, gain=.5)


def test_fixed_balanced_mix_is_symmetric_in_the_two_temporal_scores():
    assert balanced_patch_score(.9, .7) == pytest.approx(.6)
    assert balanced_patch_score(.7, .9) == pytest.approx(.6)
    assert balanced_patch_score(0, 0) == 0
