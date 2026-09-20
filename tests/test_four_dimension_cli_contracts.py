from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


DIMENSIONS = (
    ("background-consistency", "background_consistency"),
    ("temporal-style", "temporal_style"),
    ("object-class", "object_class"),
    ("color", "color"),
)


@pytest.mark.parametrize("distribution, import_name", DIMENSIONS)
def test_new_cli_has_the_common_mode_and_input_contract(distribution, import_name):
    module = importlib.import_module(f"{import_name}.cli")
    parser = module.build_parser()
    parsed = parser.parse_args(["--both", "--video", "clip.mp4", "--gpu", "--model-config", "models.toml"])
    assert parsed.both is True
    assert parsed.video == "clip.mp4"
    assert parsed.gpu == "0"
    assert parsed.model_config == "models.toml"
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "clip.mp4"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--audit", "--video", "clip.mp4", "--video-dir", "clips"])


@pytest.mark.parametrize("distribution, import_name", DIMENSIONS)
def test_new_cli_emits_null_statuses_for_both_backends(tmp_path: Path, distribution, import_name):
    module = importlib.import_module(f"{import_name}.cli")
    video = tmp_path / "video_000.mp4"
    video.touch()
    output = tmp_path / "output"
    assert module.main(["--both", "--video", str(video), "--output", str(output)]) == 1
    official = next((output / distribution / "vbench").iterdir())
    audit = next((output / distribution / "audit").iterdir())
    assert official.name == audit.name
    for destination in (official, audit):
        rows = json.loads((destination / "results.json").read_text(encoding="utf-8"))
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        assert len(rows) == 1
        assert rows[0]["video"] == str(video)
        assert rows[0]["status"] == "not_implemented"
        assert rows[0]["score"] is None
        assert rows[0]["backend"] in {"vbench", "audit"}
        assert rows[0]["implementation_status"] == "not_implemented"
        assert summary["status"] == "not_implemented"
