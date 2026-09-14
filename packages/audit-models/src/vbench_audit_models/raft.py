"""The VBench 1.0 RAFT adapter shared by Dynamic Degree and Motion Smoothness."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RaftFlowModel:
    """Load the locked official RAFT and return dense, unpadded flow.

    The official model remains lazy with respect to importing VBench and torch:
    importing this class is safe for CPU-only CLI/help and pure algorithm tests.
    Model inference intentionally follows VBench's 20-iteration ``test_mode``
    call and then removes ``InputPadder``'s padding for callers that need a
    frame-shaped flow field.
    """

    def __init__(
        self,
        device: Any,
        model_weight: Path | str,
        upstream_path: Path | str | None = None,
    ) -> None:
        weight = Path(model_weight).expanduser()
        if not weight.is_file():
            raise FileNotFoundError(f"official RAFT weight not found: {weight}")
        from vbench_audit_core.upstream import import_official_module

        module, state = import_official_module("dynamic_degree", upstream_path)
        args = module.edict(
            {
                "model": str(weight),
                "small": False,
                "mixed_precision": False,
                "alternate_corr": False,
            }
        )
        dynamic = module.DynamicDegree(args, device)
        self.module = module
        self.upstream_state = state
        self.device = device
        self.model = dynamic.model

    def compute_flow(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        torch = self.module.torch
        first = torch.from_numpy(np.asarray(frame_a, dtype=np.uint8)).permute(2, 0, 1).float()[None]
        second = torch.from_numpy(np.asarray(frame_b, dtype=np.uint8)).permute(2, 0, 1).float()[None]
        first = first.to(self.device)
        second = second.to(self.device)
        padder = self.module.InputPadder(first.shape)
        padded_first, padded_second = padder.pad(first, second)
        with torch.no_grad():
            _, flow = self.model(padded_first, padded_second, iters=20, test_mode=True)
        flow = padder.unpad(flow)
        return flow[0].permute(1, 2, 0).detach().cpu().numpy()
