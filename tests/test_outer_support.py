import numpy as np
import pytest

from dynamic_degree.image_plane_modes import region_partition
from dynamic_degree.outer_support import outer_support_partition, outer_support_reversal, partition_support


def fixture():
    y, x = np.mgrid[:12, :12]
    xy = np.c_[x.ravel(), y.ravel()].astype(float)
    large = np.stack([x < 4, (x >= 4) & (x < 8), x >= 8])
    small = np.zeros((144, 12, 12), bool)
    small[np.arange(144), y.ravel(), x.ravel()] = True
    masks = np.concatenate([large, small])
    times = np.arange(16.)/8
    return xy, masks, times, np.ones((15, 144), bool)


def test_outer_support_uses_existing_larger_masks_not_new_geometry():
    xy, masks, _, known = fixture()
    outer = outer_support_partition(masks, xy)
    assert set(outer) == {0, 1, 2}
    assert len(set(region_partition(masks, xy))) == 144
    assert partition_support(region_partition(masks, xy), xy, (12, 12), known)['fully_protected_degenerate_point_pairs'] == 2160
    assert partition_support(outer, xy, (12, 12), known)['fully_protected_degenerate_point_pairs'] == 0
    masks[:, 0, 0] = False
    assert outer_support_partition(masks, xy)[0] == -1
    with pytest.raises(ValueError, match='outside'):
        outer_support_partition(masks, xy + 100)


def test_equal_area_ownership_is_stable():
    xy, masks, _, _ = fixture()
    assert outer_support_partition(np.stack([masks[0], masks[0]]), xy)[0] == 0


def test_coherent_large_region_periodic_and_camera_motion_not_removed():
    xy, masks, times, known = fixture()
    owners = outer_support_partition(masks, xy)
    field = owners[:, None]*[3., -2.] + np.c_[xy[:, 1], -xy[:, 0]]
    values = np.sin(np.arange(15.)*2.3)[:, None, None]*field + [4., -1.]
    result, arrays = outer_support_reversal(values, times, xy, masks, known, (12, 12))
    assert not result['subtracted']
    np.testing.assert_array_equal(arrays['corrected_velocity'], values)


def test_local_real_periodic_motion_does_not_pass_cross_region_prediction():
    xy, masks, times, known = fixture()
    field = np.sin(xy) * (xy[:, :1] < 3)
    values = np.sin(np.arange(15.)*2.3)[:, None, None]*field
    result, arrays = outer_support_reversal(values, times, xy, masks, known, (12, 12))
    np.testing.assert_array_equal(arrays['corrected_velocity'], values)


def test_partition_change_does_not_erase_unknown_estimates():
    xy, masks, times, known = fixture()
    known[::3, :25] = False
    values = np.sin(np.arange(15.)*2.3)[:, None, None]*np.sin(xy)[None] + [3., -2.]
    result, arrays = outer_support_reversal(values, times, xy, masks, known, (12, 12))
    np.testing.assert_array_equal(arrays['corrected_velocity'][~known], values[~known])
    np.testing.assert_allclose(np.sum(arrays['removal']*np.diff(times)[:, None, None], axis=0), 0, atol=1e-7)
    assert result['physical_artifact_classification'] == 'NOT VERIFIED'
