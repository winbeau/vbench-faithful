import numpy as np
import pytest

from scripts.counterfactual.probe_feature_tracks import prepare_queries


def test_grid_uses_every_cell_and_native_time_without_image_selection():
    frame = np.zeros((256, 512, 3), np.uint8)
    q, groups, count = prepare_queries(frame, 7, "grid", 12)
    assert q.shape == (144, 3) and count == 144
    assert q.dtype == np.float32 and np.all(q[:, 0] == 7)
    assert groups == [[i] for i in range(144)]
    assert np.unique(q[:, 1]).size == np.unique(q[:, 2]).size == 12
    assert np.all(q[:, 1:] >= 0) and np.all(q[:, 1:] < [512, 256])
    assert np.array_equal(q, prepare_queries(255 - frame, 7, "grid", 12)[0])
    assert np.array_equal(q[:, 1:], prepare_queries(frame, 8, "grid", 12)[0][:, 1:])


@pytest.mark.parametrize("detector,size", [("unknown", 12), ("grid", 1), ("grid", 300), ("grid", 3.5)])
def test_grid_rejects_unknown_or_non_native_query_geometry(detector, size):
    with pytest.raises(ValueError):
        prepare_queries(np.zeros((256, 256, 3), np.uint8), 0, detector, size)
