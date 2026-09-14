import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_message.py"
_spec = importlib.util.spec_from_file_location("check_commit_message", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_conventional_commit_examples():
    for message in (
        "feat(core): 增加输出 provenance",
        "fix(dynamic-degree): handle short videos",
        "refactor(workspace)!: change CLI contract",
    ):
        assert _module.validate(message) == (True, "ok")


def test_invalid_commit_examples():
    for message in ("update stuff", "feature(core): missing type", "fix: no scope", "docs(UPPER): bad scope"):
        assert _module.validate(message)[0] is False
