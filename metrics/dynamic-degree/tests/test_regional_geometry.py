import numpy as np
import pytest

from dynamic_degree.regional_geometry import fit_geometry, regional_geometry, transform_points


@pytest.mark.parametrize('model', ['translation', 'similarity', 'affine'])
@pytest.mark.parametrize('robust', [False, True])
def test_translation_and_subpixel_camera_preserved(model, robust):
    x = np.random.default_rng(1).uniform(0, 200, (30, 2))
    y = x + [.2, -.35]
    result = fit_geometry(x, y, model, robust=robust)
    assert result['status'] == 'proposal_only'
    assert np.allclose(transform_points(result['matrix'], x), y, atol=1e-10)


@pytest.mark.parametrize('model', ['similarity', 'affine'])
def test_stationary_center_does_not_erase_rotation_or_approach(model):
    x = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    matrix = np.array([[1.1, -.3, 0.], [.3, 1.1, 0.]])
    y = transform_points(matrix, x)
    assert np.allclose(x.mean(0), y.mean(0))
    r = fit_geometry(x, y, model, robust=True)
    assert np.allclose(r['matrix'], matrix)
    assert np.linalg.norm(transform_points(r['matrix'], x) - x, axis=1).min() > 0


def test_periodic_return_does_not_cancel_pointwise_path():
    x = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    y = transform_points([[0., -1., 0.], [1., 0., 0.]], x)
    a = fit_geometry(x, y, 'similarity'); b = fit_geometry(y, x, 'similarity')
    first = transform_points(a['matrix'], x); final = transform_points(b['matrix'], first)
    assert np.allclose(final, x)
    assert np.all(np.linalg.norm(first-x, axis=1) + np.linalg.norm(final-first, axis=1) > 0)


def test_affine_handles_shear_but_collinear_geometry_is_insufficient():
    x = np.random.default_rng(3).uniform(0, 100, (20, 2))
    matrix = [[1., .15, -3.], [-.02, .9, 2.]]
    assert np.allclose(fit_geometry(x, transform_points(matrix, x), 'affine')['matrix'], matrix)
    line = np.c_[np.arange(8), np.arange(8)*2]
    r = fit_geometry(line, line+[3, -1], 'affine')
    assert r['matrix'] is None and r['reason'] == 'rank_deficient'
    assert fit_geometry(np.zeros((0, 2)), np.zeros((0, 2)), 'translation')['matrix'] is None


def test_irls_resists_minority_outliers_without_motion_floor():
    x = np.random.default_rng(4).uniform(0, 100, (50, 2)); y = x + [.05, -.02]
    y[:6] += [90, 70]
    raw = fit_geometry(x, y, 'translation')
    robust = fit_geometry(x, y, 'translation', robust=True)
    assert np.linalg.norm(np.asarray(raw['matrix'])[:, 2] - [.05, -.02]) > 5
    assert np.allclose(np.asarray(robust['matrix'])[:, 2], [.05, -.02], atol=1e-7)


def test_reported_weights_are_those_used_for_fit_not_next_iteration():
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = x + [1., 0.]; y[0] += [15., 9.]
    result = fit_geometry(x, y, 'translation', robust=True, max_iterations=1)
    delta = np.average(y-x, axis=0, weights=result['fit_weights'])
    assert np.allclose(np.asarray(result['matrix'])[:, 2], delta)
    assert result['fit_weights'] == [1., 1., 1., 1.]


def test_duplicate_keypoint_orientations_do_not_inflate_votes_or_leak_folds():
    x = np.array([[0., 0.], [0., 0.], [3., 0.], [3., 4.], [8., 4.]])
    y = x + [2., 0.]
    matches = [dict(source_key=i, target_key=i, source_xy=a.tolist(), target_xy=b.tolist()) for i, (a, b) in enumerate(zip(x, y))]
    result = regional_geometry(matches, range(len(x)))
    assert result['score'] is None and result['unique_source_locations'] == 4
    for proposal in result['proposals']:
        for f in proposal['spatial_folds']:
            assert (0 in f['training_keys']) == (1 in f['training_keys'])
            assert set(f['training_keys']).isdisjoint(f['heldout_keys'])
    assert result['proposals'][0]['full']['fit_weights'][:2] == [.5, .5]


def test_crossfit_exposes_nonrigid_mismatch_without_certifying_full_rank():
    rng = np.random.default_rng(8); x = rng.uniform(0, 100, (30, 2)); y = rng.uniform(0, 100, (30, 2))
    matches = [dict(source_key=i, target_key=i, source_xy=a.tolist(), target_xy=b.tolist()) for i, (a, b) in enumerate(zip(x, y))]
    result = regional_geometry(matches, range(len(x)))
    assert result['status'] == 'diagnostic_only'
    assert all(np.mean(p['spatial_folds'][0]['heldout_residual_pixels']) > 10 for p in result['proposals'])
    with pytest.raises(ValueError): regional_geometry(matches, [999])


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_geometry_refused(value):
    with pytest.raises(ValueError): fit_geometry([[value, 0]], [[1, 1]], 'translation')
