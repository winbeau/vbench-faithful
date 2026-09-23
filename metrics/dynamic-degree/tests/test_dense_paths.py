import cv2
import numpy as np
import pytest

from dynamic_degree.dense_paths import local_dense_paths
from dynamic_degree.dense_correspondence import dense_evidence
from dynamic_degree.local_trajectory import LocalTrajectoryConfig


def fixture():
    cfg = LocalTrajectoryConfig(blur_sigma=2., illumination_sigma=6., tracker_blur_sigma=0)
    image = np.random.default_rng(8).integers(30, 220, (96, 96, 3), dtype=np.uint8)
    frames = np.stack([cv2.warpAffine(image, np.float32([[1, 0, t], [0, 1, 0]]),
                                     (96, 96), borderMode=cv2.BORDER_REFLECT) for t in range(16)])
    forward = np.zeros((15, 96, 96, 2), np.float32)
    forward[..., 0] = 1
    queries = np.array([[25, 25], [50, 25], [25, 50], [50, 50]], np.float32)
    return frames, forward, queries, cfg


def test_local_advection_uses_path_locations_and_unique_ownership():
    frames, flow, queries, cfg = fixture()
    cache = local_dense_paths(frames, np.arange(16) / 8, flow, -flow, queries, cfg)
    for k, (a, b) in enumerate(cache['windows']):
        assert np.allclose(cache[f'window_{k}_tracks'], queries[None] + np.arange(b - a)[:, None, None] * [1, 0])
        assert cache[f'window_{k}_reliable_pair'].all()
    assert cache['transition_owner'].shape == (15,)
    assert (cache['transition_owner'] >= 0).all()


def test_wrong_reverse_cannot_certify_paths():
    frames, flow, queries, cfg = fixture()
    cache = local_dense_paths(frames, np.arange(16) / 8, flow, flow * 3, queries, cfg)
    assert all(not cache[f'window_{k}_reliable_pair'].any() for k in range(len(cache['windows'])))


def test_time_varying_query_support_and_geometry_checks():
    frames, flow, queries, cfg = fixture()
    moving_queries = np.tile(queries, (15, 1, 1))
    moving_queries[3] = 200
    evidence = dense_evidence(frames, flow, -flow, cfg, query_positions=moving_queries)
    assert not evidence['inside_pair'][3].any()
    with pytest.raises(ValueError):
        dense_evidence(frames, flow, -flow, cfg, query_positions=moving_queries[:2])
    with pytest.raises(ValueError):
        local_dense_paths(frames, np.arange(16) / 8, flow[:2], -flow, queries, cfg)
