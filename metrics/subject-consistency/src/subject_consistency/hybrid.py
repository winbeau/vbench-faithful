"""Local-weight entry point for the experimentally frozen v5 subject repair."""
from __future__ import annotations

import os
from pathlib import Path

from vbench_audit_core.inputs import sha256_file


DETECTOR_SHA256 = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
MOBILESAM_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
DINO_SHA256 = "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"


def load_hybrid_provider(detector_checkpoint: Path, mobilesam_checkpoint: Path, device):
    # Validate before importing model libraries or constructing any model.
    for path, expected, name in ((detector_checkpoint, DETECTOR_SHA256, "COCO detector"),
                                 (mobilesam_checkpoint, MOBILESAM_SHA256, "MobileSAM")):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{name} requires the pinned local checkpoint")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from vbench_audit_models.foreground import CocoSubjectBoxDetector, load_mobile_sam_predictor
    from .temporal_localizer import TemporalMobileSamSubjectMaskProvider

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    detector = CocoSubjectBoxDetector(detector_checkpoint, expected_sha256=DETECTOR_SHA256,
                                     device=device, threshold=.5, size=512)
    return TemporalMobileSamSubjectMaskProvider(
        detector, load_mobile_sam_predictor(mobilesam_checkpoint, device),
        weights_sha256=MOBILESAM_SHA256, prompt_policy="direct_anchor")
