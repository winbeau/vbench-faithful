from copy import deepcopy

import cv2
import numpy as np
import pytest

from scripts.counterfactual.probe_line_structure import inspect_video
from scripts.counterfactual.audit_line_structure import audit_video


def fixture():
    frame = np.zeros((64, 64, 3), np.uint8)
    cv2.rectangle(frame, (8, 8), (27, 32), (140, 120, 240), 2)
    cv2.line(frame, (35, 12), (51, 49), (250, 210, 100), 2)
    cv2.line(frame, (36, 49), (55, 24), (50, 210, 220), 2)
    frames = np.stack([frame] * 3); times = np.arange(3) / 8
    result = inspect_video(frames, times)
    result.update(sampling={"timestamps": times.tolist()}, decoded_shape=list(frames.shape))
    return result, frames, times


def test_independent_pixels_geometry_global_ranking_and_composition():
    result, frames, times = fixture()
    checked = audit_video(result, frames, times)
    assert checked["matched_lines"] > 0 and checked["junctions"] > 0 and checked["matched_junctions"] > 0
    assert checked["line_ncc_rechecks"] == checked["matched_lines"]
    assert checked["full_target_rank_queries"] == 3
    assert checked["adjacent_conditional_junction_speed"]["raw"]["mean"] == 0


def test_tampered_geometry_pixels_and_missing_frame_pairs_are_rejected():
    result, frames, times = fixture()
    bad = deepcopy(result); bad["frames"][0]["junctions"].pop()
    with pytest.raises(ValueError, match="junction completeness"):
        audit_video(bad, frames, times)
    bad = deepcopy(result); bad["pairs"].pop()
    with pytest.raises(ValueError, match="phase/pair"):
        audit_video(bad, frames, times)
    bad = deepcopy(result); bad["pairs"][0]["lines"]["matches"][0]["endpoint_displacements"][0][0] = 2
    with pytest.raises(ValueError, match="endpoint displacement"):
        audit_video(bad, frames, times)


def test_no_detections_does_not_become_zero_speed():
    frames = np.zeros((3, 32, 32, 3), np.uint8); times = np.arange(3) / 8
    result = inspect_video(frames, times)
    result.update(sampling={"timestamps": times.tolist()}, decoded_shape=list(frames.shape))
    checked = audit_video(result, frames, times)
    assert checked["matched_lines"] == 0
    assert checked["adjacent_conditional_junction_speed"]["raw"] == {"samples": 0, "mean": None}
