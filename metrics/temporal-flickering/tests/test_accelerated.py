import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from temporal_flickering.accelerated import score_video


def test_streaming_matches_native_all_frame_fp32_and_releases_capture(monkeypatch):
    frames = [np.full((3, 4, 3), v, dtype=np.uint8) for v in (0, 9, 255, 12, 0)]
    remaining, released = iter(frames), []
    def read():
        frame = next(remaining, None)
        return frame is not None, frame
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: SimpleNamespace(
        isOpened=lambda: True, read=read, release=lambda: released.append(True)))
    differences = [np.mean(cv2.absdiff(a.astype(np.float32), b.astype(np.float32)))
                   for a, b in zip(frames, frames[1:])]
    assert score_video("movie") == (255 - np.mean(np.array(differences)).item()) / 255
    assert released == [True]


def test_cpu_path_imports_no_torch():
    code = "from temporal_flickering.accelerated import score_video; import sys; assert score_video('/nonexistent-video.mp4') is None; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_single_frame_rejected_and_capture_closed(monkeypatch):
    released, stream = [], iter([(True, np.zeros((2,2,3), dtype=np.uint8)), (False, None)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: SimpleNamespace(
        isOpened=lambda: True, read=lambda: next(stream), release=lambda: released.append(True)))
    with pytest.raises(ValueError, match="two decoded frames"):
        score_video("one frame")
    assert released == [True]
