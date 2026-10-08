import json
from pathlib import Path

import numpy as np
import pytest

from dynamic_degree.intensity_scale import calibrated_intensity, fit_scale
from scripts.counterfactual.select_intensity_calibration import select_calibration, validate_calibration

ROOT = Path(__file__).resolve().parents[1]


def test_zero_preserving_continuous_bounded_and_monotone():
    x = np.array([0., 1e-10, .1, .2, .20001, 1., 10.])
    score = calibrated_intensity(x, .2)
    assert score[0] == 0 and score[3] == .5 and score[-1] < 1
    assert np.all(np.diff(score) > 0)
    assert score[4]-score[3] < .0001


def test_mean_fit_has_no_offset_or_free_slope_and_preserves_intensity_units():
    x, y = np.array([0., .1, .2, .6]), np.array([0, 0, 1, 1])
    fit = fit_scale(x, y)
    assert fit['calibrated_mean'] == pytest.approx(.5, abs=1e-12)
    assert fit['offset'] == 0 and fit['exponent'] == 1
    other = fit_scale(x*100, y)
    assert other['scale'] == pytest.approx(fit['scale']*100)
    np.testing.assert_allclose(calibrated_intensity(x, fit['scale']), calibrated_intensity(x*100, other['scale']))


@pytest.mark.parametrize('x,y', [([0, 0], [0, 1]), ([1, 2], [1, 1]), ([1, 2], [0, 0]),
                               ([1, None], [0, 1]), ([1, 2], [0, .2]), ([-1, 2], [0, 1])])
def test_invalid_or_degenerate_calibration_never_fakes_a_fitted_score(x, y):
    with pytest.raises(ValueError):
        fit_scale(x, y)


@pytest.mark.parametrize('x,scale', [([np.nan], .1), ([-1], .1), ([1], 0), ([1], np.inf)])
def test_invalid_mapping_rejects_instead_of_zero_fill(x, scale):
    with pytest.raises(ValueError):
        calibrated_intensity(x, scale)




def dev32_pairs():
    return [{'base_id': f'eval{i}', 'prompt_id': f'eval_prompt{i//4}',
             'origin': {'base': i%2, 'cf': [1., 1.]},
             'repair': {'base': .2, 'cf': [.3, .1]},
             'raw_ablation': {'base': .2, 'cf': [.4, .2]}} for i in range(32)]


def test_frozen_mapping_keeps_origin_and_both_seeds_without_refitting():
    from scripts.counterfactual.calibrate_motion_intensity import align_pairs, aligned_statistics
    pairs = dev32_pairs()
    fit = {'scale': .2, 'cohort': [{'video_uid': 'fit0', 'prompt_id': 'fit_prompt0'}]}
    mapped = align_pairs(pairs, fit)
    assert [r['origin'] for r in mapped] == [r['origin'] for r in pairs]
    assert all(r['repair']['base']==.5 for r in mapped)
    assert all(r['repair']['base']==.2 for r in pairs)  # input preserved
    assert mapped[0]['repair']['cf'] == pytest.approx([.6, 1/3])
    assert mapped[0]['repair_intensity'] == pairs[0]['repair']
    result = aligned_statistics(mapped)
    # Mapping precedes within-source averaging; signed negative changes retained.
    assert result['repair']['delta'] == pytest.approx((.6+1/3)/2-.5)
    assert result['criterion']['passes_batch_point_estimate']
    assert result['baseline_alignment']['refitted_on_dev32'] is False


def test_calibration_leakage_or_incomplete_evaluation_fails():
    from scripts.counterfactual.calibrate_motion_intensity import align_pairs
    pairs = dev32_pairs()
    for overlap in ({'video_uid': 'eval0', 'prompt_id': 'fit_prompt'},
                    {'video_uid': 'fit0', 'prompt_id': 'eval_prompt0'}):
        with pytest.raises(ValueError, match='leakage'):
            align_pairs(pairs, {'scale': .2, 'cohort': [overlap]})
    with pytest.raises(ValueError, match='complete'):
        align_pairs(pairs[:-1], {'scale': .2, 'cohort': []})


def test_calibration_cluster_bootstrap_and_degenerate_resamples():
    from scripts.counterfactual.calibrate_motion_intensity import bootstrap_scales
    cohort = [{'prompt_id': str(i), 'intensity': .1*(j+1), 'origin': j%2}
              for i in range(21) for j in range(3)]
    boot = bootstrap_scales(cohort, replicates=30)
    expected = fit_scale([.1, .2, .3], [0, 1, 0])['scale']
    assert boot['ci95'] == pytest.approx([expected, expected])
    for row in cohort:
        row['origin'] = 0
    assert bootstrap_scales(cohort, replicates=30)['status'] == 'NOT ESTIMABLE'
