import hashlib
import numpy as np
import pytest

from scripts.object_color_construction import occlude, encode_lossless


def test_inward_feather_and_nested_visibility_have_exact_replay(tmp_path):
    frames = np.random.default_rng(8).integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    masks = np.zeros((16, 32, 32), np.uint8)
    masks[:, 4:20, 3:22] = 1
    last_hidden = set()
    for fraction in (1, .75, .5, .25, 0):
        result, proof = occlude(frames, masks, fraction, seed=13)
        replay, same_proof = occlude(frames, masks, fraction, seed=13)
        assert np.array_equal(result[masks == 0], frames[masks == 0])
        assert np.array_equal(result, replay) and proof == same_proof
        assert last_hidden <= set(proof["hidden_indices"])
        assert len(proof["hidden_indices"]) == int(16*(1-fraction))
        last_hidden = set(proof["hidden_indices"])
    first = encode_lossless(result, tmp_path / "a.mp4")
    second = encode_lossless(replay, tmp_path / "b.mp4")
    assert first["sha256"] == second["sha256"]
    import subprocess
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(tmp_path / "a.mp4"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    assert hashlib.sha256(decoded).hexdigest() == proof["rgb_sha256"]
