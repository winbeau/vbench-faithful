"""Use the same numeric settings in the public CLI and experiment runners."""
from __future__ import annotations

import os


def configure_inference(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"seed": seed, "deterministic_algorithms": True, "matmul_tf32": False,
            "cudnn_tf32": False, "cudnn_benchmark": False,
            "exception": "COCO box forward temporarily enables native CUDA ROIAlign; see localizer provenance"}
