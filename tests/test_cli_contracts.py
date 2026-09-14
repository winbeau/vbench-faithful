from __future__ import annotations

import importlib

import pytest

from vbench_audit_core.devices import parse_gpu


CLI_MODULES = (
    "dynamic_degree.cli",
    "motion_smoothness.cli",
    "subject_consistency.cli",
    "scene.cli",
    "human_action.cli",
    "spatial_relationship.cli",
    "overall_consistency.cli",
    "multiple_objects.cli",
)


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_every_cli_has_required_mode_and_input_contract(module_name: str) -> None:
    module = importlib.import_module(module_name)
    parser = module.build_subject_parser() if module_name.startswith("subject_") else module.build_parser()
    parsed = parser.parse_args(["--both", "--video", "clip.mp4", "--gpu"])
    assert parsed.both is True
    assert parsed.video == "clip.mp4"
    assert parsed.gpu == "0"
    with pytest.raises(SystemExit):
        parser.parse_args(["--video", "clip.mp4"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--vbench", "--video", "clip.mp4", "--video-dir", "clips"])


@pytest.mark.parametrize("value, expected", [(None, [0]), ("0", [0]), ("0,2,4", [0, 2, 4])])
def test_gpu_contract(value: str | None, expected: list[int]) -> None:
    assert parse_gpu(value) == expected


@pytest.mark.parametrize("value", ["", "0,0", "-1", "0,x"])
def test_gpu_contract_rejects_invalid(value: str) -> None:
    with pytest.raises(Exception):
        parse_gpu(value)
