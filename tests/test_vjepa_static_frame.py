import numpy as np
import pytest

from scripts.counterfactual.probe_vjepa_static_frame import global_pan


def test_no_motion_and_exact_integer_endpoint_translation():
    frame = np.zeros((32, 48, 3), np.uint8)
    frame[10:14, 10:14] = 255
    static = global_pan(frame, 16, 0)
    assert np.array_equal(static, np.repeat(frame[None], 16, axis=0))
    pan = global_pan(frame, 16, 8)
    assert np.array_equal(pan[0], frame)
    assert np.array_equal(pan[-1, 10:14, 18:22], frame[10:14, 10:14])
    assert pan.shape == static.shape and pan.dtype == np.uint8


def test_invalid_pan_input_rejected():
    with pytest.raises(ValueError):
        global_pan(np.zeros((4, 4, 3), np.float32), 16, 8)
    with pytest.raises(ValueError):
        global_pan(np.zeros((4, 4, 3), np.uint8), 1, 8)
