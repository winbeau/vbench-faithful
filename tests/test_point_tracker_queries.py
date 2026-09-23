import numpy as np
import pytest

from vbench_audit_models.point_tracker import CoTracker2Model


def test_explicit_time_queries_preserved_and_predicted_out_of_view_not_clamped():
    torch = pytest.importorskip("torch")
    model = object.__new__(CoTracker2Model)
    model.device = torch.device("cpu")
    calls = []

    def predict(video, *, queries, backward_tracking):
        calls.append((video.shape, queries.cpu().numpy(), backward_tracking))
        tracks = queries[:, None, :, 1:].repeat(1, video.shape[1], 1, 1)
        tracks[:, -1, :, 0] = 99  # outside native image must stay explicit
        return tracks, torch.ones(tracks.shape[:-1], dtype=torch.bool)

    model.model = predict
    frames = np.zeros((4, 12, 16, 3), np.uint8)
    q = np.array([[2, 3, 4], [1, 8, 2]], np.float32)
    result = model.track_queries(frames, q)
    assert calls[0][0] == (1, 4, 3, 12, 16)
    np.testing.assert_array_equal(calls[0][1][0], q)
    assert calls[0][2] is True
    assert (result["tracks"][-1, :, 0] == 99).all()
    assert q[0, 1] == 3


@pytest.mark.parametrize("query", [[[.5, 2, 3]], [[4, 2, 3]], [[1, 17, 3]], [[1, 2, -1]], [[1, np.nan, 3]], []])
def test_invalid_queries_rejected_before_model_call(query):
    pytest.importorskip("torch")
    model = object.__new__(CoTracker2Model)
    with pytest.raises(ValueError):
        model.track_queries(np.zeros((4, 12, 16, 3), np.uint8), query)
