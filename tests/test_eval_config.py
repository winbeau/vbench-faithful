"""Configuration and event boundaries, with no model or GPU initialization."""

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from vbench_audit_core.eval_config import OFFICIAL_DIMENSIONS, PAPER_DIMENSIONS, load_config
from vbench_audit_core.eval_logging import EvalLogger
from vbench_audit_core.paths import workspace_root


BASE = "input: ../inputs.jsonl\nassets: ../runtime/assets.json\n"
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def config_file(tmp_path, body=BASE):
    path = tmp_path / "configs/eval.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_are_resolved_serializable_and_do_not_create_outputs(tmp_path):
    path = config_file(tmp_path)
    first, second = load_config(path), load_config(path)
    assert first.version == 1 and first.backend == "repair" and first.reuse is True
    assert first.dimensions == OFFICIAL_DIMENSIONS and first.gpus == (0,)
    assert first.input == tmp_path / "inputs.jsonl"
    assert first.assets == tmp_path / "runtime/assets.json"
    assert first.video_root is None
    assert first.cache_dir == Path("~/.cache/vbench-repair").expanduser().resolve()
    assert first.env_dir == workspace_root() / ".venvs/metrics"
    assert first.output.parent == tmp_path / "output/eval"
    assert first.output.name.endswith("Z") and first.output != second.output
    assert not first.output.parent.exists()
    serialized = json.loads(json.dumps(first.as_dict(), allow_nan=False))
    assert serialized["input"] == str(first.input)
    assert serialized["gpus"] == [0] and serialized["dimensions"] == list(OFFICIAL_DIMENSIONS)


def test_dimensions_match_the_existing_paper_routes():
    from scripts.paper_common import OFFICIAL, PAPER
    assert OFFICIAL_DIMENSIONS == OFFICIAL
    assert PAPER_DIMENSIONS == PAPER
    assert len(OFFICIAL_DIMENSIONS) == 16 and len(PAPER_DIMENSIONS) == 9


@pytest.mark.parametrize("name,canonical", [("ours", "repair"), ("official", "origin"),
                                          ("repair", "repair"), ("origin", "origin")])
def test_backend_aliases_preserve_legacy_names(tmp_path, name, canonical):
    assert load_config(config_file(tmp_path, BASE + f"backend: {name}\n")).backend == canonical


def test_explicit_dimensions_keep_order_and_normalize_historical_names(tmp_path):
    path = config_file(tmp_path, BASE + "dimensions: [dynamics-degree, multiplt_object, color-consistency, scene]\n")
    assert load_config(path).dimensions == ("dynamic_degree", "multiple_objects", "color", "scene")
    assert load_config(path, {"dimensions": "paper"}).dimensions == PAPER_DIMENSIONS


def test_yaml_paths_are_relative_to_config_even_from_another_cwd(tmp_path, monkeypatch):
    path = config_file(tmp_path, BASE + "output: ../fresh\nvideo_root: media\ncache_dir: cache\nenv_dir: envs\n")
    cwd = tmp_path / "caller"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = load_config(path)
    assert config.output == tmp_path / "fresh"
    assert config.video_root == path.parent / "media"
    assert config.cache_dir == path.parent / "cache"
    assert config.env_dir == path.parent / "envs"
    assert config.input == tmp_path / "inputs.jsonl"


def test_cli_paths_are_relative_to_cwd_and_unoverridden_paths_stay_with_yaml(tmp_path, monkeypatch):
    path = config_file(tmp_path, BASE + "output: file-output\nvideo_root: file-media\ncache_dir: file-cache\nenv_dir: file-env\n")
    cwd = tmp_path / "caller"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = load_config(path, {
        "input": Path("cli/inputs.json"), "assets": "cli/assets.json",
        "output": Path("cli/output"), "video_root": Path("cli/media"), "env_dir": "cli/env",
        "backend": "both", "reuse": False,
    })
    assert config.input == cwd / "cli/inputs.json"
    assert config.assets == cwd / "cli/assets.json"
    assert config.output == cwd / "cli/output"
    assert config.video_root == cwd / "cli/media"
    assert config.env_dir == cwd / "cli/env"
    assert config.cache_dir == path.parent / "file-cache"
    assert config.backend == "both" and config.reuse is False


def test_environment_and_home_paths_expand_without_recursive_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_TEST_ROOT", str(tmp_path / "external"))
    monkeypatch.setenv("EVAL_TEST_CACHE", "relative-cache")
    path = config_file(tmp_path, "input: ${EVAL_TEST_ROOT}/inputs.json\nassets: ~/runtime/assets.json\ncache_dir: ${EVAL_TEST_CACHE}\n")
    config = load_config(path)
    assert config.input == tmp_path / "external/inputs.json"
    assert config.assets == Path.home() / "runtime/assets.json"
    assert config.cache_dir == path.parent / "relative-cache"
    monkeypatch.setenv("EVAL_TEST_CACHE", "${EVAL_TEST_ROOT}")
    with pytest.raises(ValueError, match="only.*environment substitution"):
        load_config(path)


def test_unset_environment_and_expression_substitution_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_TEST_UNSET", raising=False)
    path = config_file(tmp_path, BASE + "cache_dir: ${EVAL_TEST_UNSET}/cache\n")
    with pytest.raises(ValueError, match="cache_dir.*EVAL_TEST_UNSET.*not set"):
        load_config(path)
    path.write_text(BASE + "cache_dir: ${1+1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only.*environment substitution"):
        load_config(path)


def test_shell_syntax_in_paths_is_literal_not_executed(tmp_path):
    path = config_file(tmp_path, BASE + "cache_dir: $(touch SHOULD_NOT_EXIST)\n")
    assert load_config(path).cache_dir.name == "$(touch SHOULD_NOT_EXIST)"
    assert not (path.parent / "SHOULD_NOT_EXIST").exists()


@pytest.mark.parametrize("body,match", [
    ("", "expected one YAML mapping"),
    ("[input, assets]", "expected one YAML mapping"),
    (BASE + "unknown_option: true\n", "unknown configuration keys.*unknown_option"),
    (BASE + "reuse: true\nreuse: false\n", "duplicate configuration key: reuse"),
    (BASE + "1: something\n", "keys must be strings"),
    (BASE + "? [a, b]\n: something\n", "keys must be strings"),
    (BASE + "backend: {name: repair}\n", "nested YAML mappings"),
    (BASE + "gpus: [[0]]\n", "not nested collections"),
    (BASE + "gpus: &devices [0]\n", "aliases and anchors"),
    (BASE + "gpus: *devices\n", "aliases and anchors"),
    (BASE + "---\nreuse: false\n", "single document"),
    ("!!map scalar", "expected a YAML mapping"),
])
def test_yaml_structure_rejects_ambiguous_or_unsupported_inputs(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_config(config_file(tmp_path, body))


def test_unsafe_yaml_tag_is_never_executed(tmp_path):
    sentinel = tmp_path / "executed"
    body = BASE + f"reuse: !!python/object/apply:os.system ['touch {sentinel}']\n"
    with pytest.raises(ValueError, match="Invalid evaluation YAML"):
        load_config(config_file(tmp_path, body))
    assert not sentinel.exists()


@pytest.mark.parametrize("field,value", [
    ("version", "true"), ("version", "1.0"), ("version", "2"), ("version", "'1'"),
    ("backend", "[]"), ("backend", "unknown"), ("backend", "null"),
    ("reuse", "1"), ("reuse", "'false'"), ("reuse", "on"), ("reuse", "null"),
    ("dimensions", "[]"), ("dimensions", "scene"), ("dimensions", "[1]"),
    ("dimensions", "[scene, scene]"), ("dimensions", "[spatial-relationship, spatial_relationship]"),
    ("dimensions", "[does_not_exist]"), ("dimensions", "true"),
    ("gpus", "[]"), ("gpus", "[true]"), ("gpus", "[-1]"), ("gpus", "['0']"),
    ("gpus", "[0, 0]"), ("gpus", "[0.0]"), ("gpus", "[null]"), ("gpus", "[GPU-wrong]"),
    ("video_root", "false"), ("cache_dir", "[]"), ("env_dir", "3"), ("output", "null"),
])
def test_fields_reject_coercion_and_invalid_values(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        load_config(config_file(tmp_path, BASE + f"{field}: {value}\n"))


@pytest.mark.parametrize("body,match", [
    ("assets: assets.json\n", "input: required"),
    ("input: inputs.json\n", "assets: required"),
    ("input: video.mp4\nassets: assets.json\n", "input: expected a JSON or JSONL"),
    ("input: inputs.JSONL\nassets: assets.json\n", "input: expected a JSON or JSONL"),
    ("input: false\nassets: assets.json\n", "input: expected a nonempty path"),
    ("input: inputs.json\nassets: ''\n", "assets: expected a nonempty path"),
])
def test_required_manifest_and_path_contracts(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_config(config_file(tmp_path, body))


def test_gpu_uuids_are_preserved_but_case_variants_cannot_duplicate_a_device(tmp_path):
    path = config_file(tmp_path, BASE + f"gpus: [0, '{GPU_UUID}', '{GPU_UUID.replace('GPU-', 'MIG-')}']\n")
    assert load_config(path).gpus == (0, GPU_UUID, GPU_UUID.replace("GPU-", "MIG-"))
    duplicate = GPU_UUID[:4] + GPU_UUID[4:].upper()
    with pytest.raises(ValueError, match="gpus: duplicate"):
        load_config(path, {"gpus": [GPU_UUID, duplicate]})


def test_overrides_are_strict_and_can_explicitly_remove_video_root(tmp_path):
    path = config_file(tmp_path, BASE + "video_root: media\n")
    assert load_config(path, {"video_root": None}).video_root is None
    for overrides, match in [([], "overrides: expected a dictionary"),
                              ({1: "value"}, "keys must be strings"),
                              ({"unknown": 1}, "unknown configuration keys"),
                              ({"reuse": "false"}, "reuse: expected a boolean")]:
        with pytest.raises(ValueError, match=match):
            load_config(path, overrides)


def test_existing_output_is_not_reused_or_created_by_configuration_loading(tmp_path):
    path = config_file(tmp_path, BASE + "output: existing\n")
    output = path.parent / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="output: expected a new directory"):
        load_config(path)
    output.rmdir()
    output.symlink_to(path.parent / "missing")
    with pytest.raises(ValueError, match="output: expected a new directory"):
        load_config(path)


def test_non_tty_events_append_json_and_keep_missing_scores_visible(tmp_path):
    terminal = io.StringIO()
    console = Console(file=terminal, force_terminal=False, color_system=None, width=120)
    for _ in range(2):
        logger = EvalLogger(tmp_path, total=1)
        logger.console = console
        with logger:
            assert logger._progress is None
            logger.event("prepare", "scene", "origin", "worker output is separate", log_path=tmp_path / "worker.log")
            logger.finish_task("scene", "origin", status="failed", message="Missing asset")
            logger.summary([{"dimension": "scene", "backend": "origin", "complete": False, "coverage": .5, "score": None}])
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert records[0]["log_path"] == str(tmp_path / "worker.log")
    assert records[1]["status"] == "failed" and records[1]["completed_tasks"] == 1
    assert records[2]["rows"][0]["score"] is None
    assert "Missing asset" in terminal.getvalue() and "50.0%" in terminal.getvalue()
    assert "\x1b" not in terminal.getvalue() and "\r" not in terminal.getvalue()


def test_logger_reports_and_preserves_an_exception(tmp_path):
    terminal = io.StringIO()
    logger = EvalLogger(tmp_path, total=1)
    logger.console = Console(file=terminal, force_terminal=False, color_system=None)
    with pytest.raises(RuntimeError, match="worker failed"):
        with logger:
            raise RuntimeError("worker failed\nSee the worker log")
    record = json.loads((tmp_path / "events.jsonl").read_text())
    assert record["stage"] == "error" and record["error_type"] == "RuntimeError"
    assert record["message"] == "worker failed\nSee the worker log"
    assert "worker failed" in terminal.getvalue()
