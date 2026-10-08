"""All sixteen dimensions have independent installed packages and entry points."""
import ast
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tomllib

from vbench_audit_core.eval_config import OFFICIAL_DIMENSIONS


ROOT = Path(__file__).resolve().parents[1]


def test_all_sixteen_projects_are_registered_with_independent_sources():
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    upstream = tomllib.loads((ROOT / "configs/upstream.toml").read_text())
    expected = {name.replace("_", "-") for name in OFFICIAL_DIMENSIONS}
    assert {p.name for p in (ROOT / "metrics").iterdir() if p.is_dir()} == expected
    assert set(upstream["dimensions"]) == set(OFFICIAL_DIMENSIONS)
    for dimension in OFFICIAL_DIMENSIONS:
        distribution = dimension.replace("_", "-")
        folder = ROOT / "metrics" / distribution
        config = tomllib.loads((folder / "pyproject.toml").read_text())
        assert distribution in root["project"]["dependencies"]
        assert root["tool"]["uv"]["sources"][distribution] == {"workspace": True}
        assert config["project"]["scripts"][distribution] == dimension + ".cli:main"
        assert any(e.name == distribution for e in importlib.metadata.distribution(distribution).entry_points)
        assert (folder / "tests").is_dir()
        for path in (folder / "src" / dimension).rglob("*.py"):
            tree = ast.parse(path.read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    imported.add(node.module.split(".")[0])
            assert not imported.intersection(set(OFFICIAL_DIMENSIONS) - {dimension}), path


def test_new_cli_help_needs_no_models_and_is_independent_of_cwd(tmp_path):
    for dimension in ("aesthetic_quality", "imaging_quality", "temporal_flickering", "appearance_style"):
        code = (f"import sys; from {dimension}.cli import main; "
                "sys.argv=['metric', '--help']; "
                "\ntry: main()\nexcept SystemExit as exc: assert exc.code == 0\n"
                "assert not {'torch', 'transformers', 'vbench'}.intersection(sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "--config" in result.stdout and "--no-reuse" in result.stdout
        assert "--dimensions" not in result.stdout
