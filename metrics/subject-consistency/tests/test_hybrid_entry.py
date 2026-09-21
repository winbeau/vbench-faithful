import json
from pathlib import Path

import pytest

from subject_consistency.hybrid import DETECTOR_SHA256, DINO_SHA256, MOBILESAM_SHA256, load_hybrid_provider


def test_hybrid_pins_match_the_measured_protocol():
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads((root / "configs/subject-repair/stability_hybrid_v5_protocol.json").read_text())
    assets = json.loads((root / "configs/subject-repair/assets.lock.json").read_text())
    natural = json.loads((root / "configs/subject-repair/natural1440_protocol_v2.json").read_text())
    assert DETECTOR_SHA256 == protocol["detector"]["weight_sha256"]
    assert MOBILESAM_SHA256 == assets["mobilesam"]["checkpoint_sha256"]
    assert DINO_SHA256 == natural["dino_weight_sha256"]


def test_hybrid_rejects_wrong_local_weights_before_loading_models(tmp_path):
    detector, sam = tmp_path / "detector.pth", tmp_path / "sam.pt"
    detector.write_bytes(b"not the pinned detector")
    sam.write_bytes(b"not the pinned SAM")
    with pytest.raises(ValueError, match="COCO detector requires the pinned local checkpoint"):
        load_hybrid_provider(detector, sam, "cpu")
