import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_message.py"
_spec = importlib.util.spec_from_file_location("check_commit_message", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_conventional_commit_examples():
    for message in (
        "feat(core): add output provenance",
        "fix(dynamic-degree): handle short videos",
        "refactor(workspace)!: change CLI contract",
        "docs(core): clarify the evaluator's input\n\nKeep the original media hashes.\n",
        "docs(core): clarify the evaluator’s input",
    ):
        assert _module.validate(message) == (True, "ok")


def test_invalid_commit_examples():
    for message in ("update stuff", "feature(core): missing type", "fix: no scope", "docs(UPPER): bad scope"):
        assert _module.validate(message)[0] is False


@pytest.mark.parametrize("message", [
    "feat(core): 增加输出 provenance",
    "fix(core): preserve video ordering\n\n修复排序错误。\n",
    "fix(core): preserve video ordering\n\nValidation: 测试通过\n",
    "fix(core): preserve video ordering\n\nReviewed-by: 中文姓名\n",
    "fix(core): preserve video ordering\n\n\U00020000\n",
    "fix(core): preserve video ordering\n\n\uf900\n",
    "fix(core): preserve video ordering\n\n〇\n",
])
def test_chinese_is_rejected_in_the_complete_message(message):
    valid, reason = _module.validate(message)
    assert valid is False
    assert "English" in reason


@pytest.mark.parametrize("via_file", [False, True])
def test_cli_checks_the_body_from_stdin_and_hook_files(tmp_path, via_file):
    message = "fix(core): preserve video ordering\n\n中文不能出现在正文。\n"
    command = [sys.executable, str(_path)]
    if via_file:
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(message, encoding="utf-8")
        command.append(str(path))
    result = subprocess.run(command, input=message, text=True, capture_output=True)
    assert result.returncode == 1
    assert "Chinese text is not allowed" in result.stderr
