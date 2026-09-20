from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = {
    "background-consistency": "background_consistency",
    "temporal-style": "temporal_style",
    "object-class": "object_class",
    "color": "color",
}


def test_root_registers_all_four_workspace_projects():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = config["project"]["dependencies"]
    sources = config["tool"]["uv"]["sources"]
    for distribution in DIMENSIONS:
        assert distribution in dependencies
        assert sources[distribution] == {"workspace": True}


def test_each_project_is_independent_and_has_the_shared_layout():
    names = set(DIMENSIONS.values())
    for distribution, import_name in DIMENSIONS.items():
        project = ROOT / "metrics" / distribution
        config = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
        assert config["project"]["name"] == distribution
        assert config["project"]["scripts"][distribution] == f"{import_name}.cli:main"
        assert (project / "src" / import_name / "backends" / "vbench.py").is_file()
        assert (project / "src" / import_name / "backends" / "audit.py").is_file()
        assert (project / "src" / import_name / "backends" / "repair.py").is_file()
        assert (project / "tests" / "test_runtime_parity.py").is_file()
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        for other in names - {import_name}:
            assert f'"{other}"' not in text


def test_new_metric_sources_do_not_import_each_other():
    names = set(DIMENSIONS.values())
    for distribution, import_name in DIMENSIONS.items():
        for source in (ROOT / "metrics" / distribution / "src" / import_name).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module.split(".")[0])
            assert not names.intersection(set(imported) - {import_name})
