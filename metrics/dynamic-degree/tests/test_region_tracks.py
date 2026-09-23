import numpy as np
import pytest

from dynamic_degree.region_tracks import region_window, track_regions, transform_xy


def test_native_roundtrip_preserves_translation_and_reversal():
    mask = np.zeros((40, 70), bool)
    mask[10:20, 6:62] = True
    window = region_window(mask)
    path = np.array([[10, 12], [19, 15], [10, 12]], float)
    mapped = transform_xy(path, window["native_to_window"])
    recovered = transform_xy(mapped, window["window_to_native"])
    np.testing.assert_allclose(recovered, path, atol=1e-12)
    assert np.linalg.norm(np.diff(recovered, axis=0), axis=-1).sum() > 18
    assert region_window(np.zeros_like(mask)) is None
    with pytest.raises(ValueError):
        region_window(mask, context=.5)


def test_all_regions_retained_empty_not_zero_and_fixed_crop_not_temporal_stabilization():
    frames = np.zeros((3, 40, 70, 3), np.uint8)
    frames[:, 10:20, 6:62] = 100
    masks = np.zeros((3, 40, 70), bool)
    masks[0, 10:20, 6:62] = True
    masks[1, 2:5, 2:5] = True
    q = np.array([[1, 15, 15], [1, 35, 15]], np.float32)
    calls = []

    class Tracker:
        def track_queries(self, images, queries):
            calls.append(images)
            assert np.array_equal(images[0], images[1])
            tracks = np.broadcast_to(queries[None, :, 1:], (3, len(queries), 2)).copy()
            return {"tracks": tracks, "visible": np.ones((3, len(queries)), bool)}

    records, arrays = track_regions(Tracker(), frames, q, masks)
    assert len(records) == 3 and len(calls) == 1
    assert [r["status"] for r in records] == ["diagnostic_only", "insufficient", "insufficient"]
    assert all(r["score"] is None for r in records)
    np.testing.assert_allclose(arrays["region_0_tracks_native"], np.tile(q[None, :, 1:], (3, 1, 1)), atol=1e-5)
