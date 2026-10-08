import time

import pytest
import torch

from vbench_audit_core.video_batches import frame_batches, prefetch


def test_prefetch_preserves_order_with_out_of_order_completion():
    def load(i):
        time.sleep(.02 if i == 0 else .001)
        return i * 10
    assert list(prefetch(range(5), load, workers=2)) == [(i, i * 10) for i in range(5)]


def test_frame_packing_keeps_short_batches_and_shape_boundaries():
    decoded = [("a", torch.arange(3).reshape(3, 1, 1)),
               ("b", torch.arange(3, 6).reshape(3, 1, 1)),
               ("c", torch.ones(1, 2, 2, dtype=torch.int64))]
    actual = list(frame_batches(decoded, 4))
    assert [owners for owners, _ in actual] == [["a", "a", "a", "b"], ["b", "b"], ["c"]]
    assert torch.cat([batch.flatten() for _, batch in actual[:2]]).tolist() == list(range(6))
    assert actual[-1][1].shape == (1, 2, 2)


def test_prefetch_propagates_decode_failure_and_rejects_empty_videos():
    def load(i):
        if i == 1:
            raise ValueError("decode failed")
        return i
    with pytest.raises(ValueError, match="decode failed"):
        list(prefetch(range(4), load))
    with pytest.raises(ValueError, match="empty video"):
        list(frame_batches([("empty", torch.zeros(0, 3, 2, 2))], 32))
