from __future__ import annotations

import numpy as np

from vbench_audit_models.raft import RaftFlowModel


class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def permute(self, *order):
        return _Tensor(self.array.transpose(order))

    def float(self):
        return self

    def __getitem__(self, index):
        return _Tensor(self.array[index])

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Torch:
    @staticmethod
    def from_numpy(array):
        return _Tensor(array)

    @staticmethod
    def no_grad():
        return _NoGrad()


class _Padder:
    calls = []

    def __init__(self, shape):
        self.shape = shape
        self.__class__.calls.append(("init", shape))

    def pad(self, first, second):
        self.__class__.calls.append(("pad", first.shape, second.shape))
        return first, second

    def unpad(self, flow):
        self.__class__.calls.append(("unpad", flow.shape))
        return flow


class _Module:
    torch = _Torch
    InputPadder = _Padder


class _Model:
    def __init__(self):
        self.calls = []

    def __call__(self, first, second, **kwargs):
        self.calls.append((first.shape, second.shape, kwargs))
        return None, _Tensor(np.zeros((1, 2, 4, 6), dtype=np.float32))


def test_shared_raft_wrapper_keeps_official_call_and_unpads():
    wrapper = RaftFlowModel.__new__(RaftFlowModel)
    wrapper.module = _Module
    wrapper.model = _Model()
    wrapper.device = "cpu"
    _Padder.calls.clear()
    result = wrapper.compute_flow(np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8))
    assert result.shape == (4, 6, 2)
    assert wrapper.model.calls[0][2] == {"iters": 20, "test_mode": True}
    assert [item[0] for item in _Padder.calls] == ["init", "pad", "unpad"]
