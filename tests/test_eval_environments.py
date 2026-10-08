"""Real subprocess checks for shared model dependencies without model downloads."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from vbench_audit_core.environments import prepare_environment


def run_python(python, code, *, pythonpath=None):
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    completed = subprocess.run([str(python), "-B", "-c", code], env=env,
                               text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def create_base(folder, python):
    uv = shutil.which("uv")
    assert uv, "environment integration tests require the workspace's uv"
    subprocess.run([uv, "venv", "--python", str(python), "--no-project", "--no-config",
                    "--offline", "--no-python-downloads", str(folder)],
                   check=True, capture_output=True, text=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PYTHON"))})
    executable = folder / "bin/python"
    site = Path(run_python(executable, "import json, site; print(json.dumps(site.getsitepackages()[0]))"))
    (site / "toy_model.py").write_text("value = 'shared model dependency'\n")
    metadata = site / "toy_model-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-model\nVersion: 1.0\n")
    (metadata / "RECORD").write_text("toy_model.py,,\n")
    return executable, site, metadata


@pytest.fixture
def base_runtime(tmp_path):
    return create_base(tmp_path / "base-runtime", sys.executable)


def base_bytes(folder):
    return {str(p.relative_to(folder)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in folder.rglob("*") if p.is_file() and not p.is_symlink()}


def test_shared_dependencies_keep_prefix_and_do_not_leak_metric_editables(tmp_path, base_runtime):
    base, site, _ = base_runtime
    # Plain .pth paths and setuptools-style import finders both occur in
    # editable runtimes. Keep third-party editables while removing metrics.
    selected = tmp_path / "checkout/metrics/scene/src"
    sibling = tmp_path / "checkout/metrics/other/src"
    selected.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (selected / "selected_metric.py").write_text("value = 'selected source'\n")
    (sibling / "sibling_metric.py").write_text("value = 'must not leak'\n")
    outside = tmp_path / "third-party"
    outside.mkdir()
    (outside / "extra_model.py").write_text("value = 'external pth dependency'\n")
    (outside / "shared_editable.py").write_text("value = 'external finder dependency'\n")
    (site / "legacy.pth").write_text(f"{selected}\n{outside}\n")
    (site / "sitecustomize.py").write_text(f"import sys; sys.path.insert(0, {str(sibling)!r})\n")
    (site / "__editable___test_finder.py").write_text(
        "import importlib.util, sys\n"
        f"MAPPING = {{'sibling_metric': {str(sibling / 'sibling_metric.py')!r}, "
        f"'shared_editable': {str(outside / 'shared_editable.py')!r}}}\n"
        "NAMESPACES = {}\n"
        "class Finder:\n"
        "    @classmethod\n"
        "    def find_spec(cls, name, path=None, target=None):\n"
        "        if name in MAPPING:\n"
        "            return importlib.util.spec_from_file_location(name, MAPPING[name])\n"
        "def install():\n"
        "    sys.meta_path.append(Finder)\n"
    )
    (site / "finder.pth").write_text("import __editable___test_finder; __editable___test_finder.install()\n")
    before = base_bytes(base.parent.parent)
    python = prepare_environment("scene", "visual", base, tmp_path / "environments")
    data = run_python(python, """
import importlib.util, json, sys, toy_model, extra_model, shared_editable
print(json.dumps({'prefix': sys.prefix, 'version': sys.version,
                  'model': toy_model.value, 'model_file': toy_model.__file__,
                  'extra': extra_model.value, 'editable': shared_editable.value,
                  'selected': importlib.util.find_spec('selected_metric') is not None,
                  'sibling': importlib.util.find_spec('sibling_metric') is not None}))
""")
    assert Path(data["prefix"]) == python.parent.parent
    assert Path(data["prefix"]) != base.parent.parent
    assert data["version"] == run_python(base, "import json, sys; print(json.dumps(sys.version))")
    assert data["model"] == "shared model dependency"
    assert Path(data["model_file"]).parent == site
    assert data["extra"] == "external pth dependency"
    assert data["editable"] == "external finder dependency"
    assert not data["selected"] and not data["sibling"]
    assert run_python(python, "import json, selected_metric; print(json.dumps(selected_metric.value))",
                      pythonpath=selected) == "selected source"
    assert base_bytes(base.parent.parent) == before
    # A venv consists of links, startup glue and receipts, not copied packages.
    assert sum(p.stat().st_size for p in python.parent.parent.rglob("*")
               if p.is_file() and not p.is_symlink()) < 200_000


def test_package_metadata_changes_create_new_fingerprint_without_removing_old_env(tmp_path, base_runtime):
    base, _, metadata = base_runtime
    root = tmp_path / "environments"
    first = prepare_environment("scene", "visual", base, root)
    assert prepare_environment("scene", "visual", base, root) == first
    receipt = json.loads((first.parent.parent / "environment.json").read_text())
    assert receipt["base_python"] == str(base)
    assert receipt["base"]["soabi"]
    assert any(item["name"] == "toy-model" and item["version"] == "1.0"
               for item in receipt["base"]["distributions"])
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-model\nVersion: 2.0\n")
    second = prepare_environment("scene", "visual", base, root)
    assert second != first and first.is_file() and second.is_file()
    # Even a rebuild at the same version changes identity through RECORD.
    (metadata / "RECORD").write_text("toy_model.py,sha256=changed,37\n")
    third = prepare_environment("scene", "visual", base, root)
    assert third not in {first, second}
    assert first.is_file() and second.is_file()


def test_dimensions_and_roles_use_separate_venvs(tmp_path, base_runtime):
    base, _, _ = base_runtime
    root = tmp_path / "environments"
    paths = {prepare_environment(dim, role, base, root)
             for dim, role in [("scene", "visual"), ("color", "visual"), ("scene", "semantic")]}
    assert len(paths) == 3
    assert {Path(run_python(p, "import json, sys; print(json.dumps(sys.prefix))")) for p in paths} == {
        p.parent.parent for p in paths}


def test_parallel_preparation_publishes_once(tmp_path, base_runtime, monkeypatch):
    base, _, _ = base_runtime
    actual_uv = shutil.which("uv")
    shim = tmp_path / "bin"
    shim.mkdir()
    calls = tmp_path / "uv-calls"
    executable = shim / "uv"
    executable.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        f"fd = os.open({str(calls)!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)\n"
        "os.write(fd, b'create\\n'); os.close(fd)\n"
        f"os.execv({actual_uv!r}, [{actual_uv!r}, *sys.argv[1:]])\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    root = tmp_path / "environments"
    code = ("from pathlib import Path; from vbench_audit_core.environments import prepare_environment; "
            f"print(prepare_environment('scene', 'visual', Path({str(base)!r}), Path({str(root)!r})))")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "packages/audit-core/src"))
    children = [subprocess.Popen([sys.executable, "-B", "-c", code], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    outputs = []
    for child in children:
        stdout, stderr = child.communicate(timeout=45)
        assert child.returncode == 0, stderr
        outputs.append(stdout.strip())
    assert len(set(outputs)) == 1
    assert calls.read_text().splitlines() == ["create"]
    assert [p.name for p in (root / "scene/visual").iterdir() if p.is_dir()] == [Path(outputs[0]).parents[1].name]


def test_failed_build_cleans_only_its_staging_directory(tmp_path, base_runtime, monkeypatch):
    base, _, _ = base_runtime
    root = tmp_path / "environments"
    existing = prepare_environment("color", "visual", base, root)
    import vbench_audit_core.environments as environments
    original_run = environments.subprocess.run

    def failing_venv(command, **kwargs):
        if "venv" in command:
            (Path(command[-1]) / "partial").write_text("incomplete")
            raise subprocess.CalledProcessError(1, command, stderr="deliberate failure")
        return original_run(command, **kwargs)

    monkeypatch.setattr(environments.subprocess, "run", failing_venv)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_environment("scene", "visual", base, root)
    assert existing.is_file()
    assert [p.name for p in (root / "scene/visual").iterdir()] == [".prepare.lock"]
    assert run_python(base, "import json, toy_model; print(json.dumps(toy_model.value))") == "shared model dependency"


def test_corrupted_cache_is_not_silently_replaced(tmp_path, base_runtime):
    base, _, _ = base_runtime
    root = tmp_path / "environments"
    python = prepare_environment("scene", "visual", base, root)
    receipt = python.parent.parent / "environment.json"
    receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="Invalid cached evaluation environment"):
        prepare_environment("scene", "visual", base, root)
    assert receipt.read_text() == "{}"


def test_visual_python_keeps_310_abi_when_controller_uses_311(tmp_path):
    interpreter = shutil.which("python3.10")
    if interpreter is None:
        pytest.skip("Python 3.10 is not installed locally; no interpreter downloads allowed")
    base, _, _ = create_base(tmp_path / "visual-base", interpreter)
    python = prepare_environment("dynamic_degree", "visual", base, tmp_path / "environments")
    assert run_python(python, "import json, sys, toy_model; print(json.dumps(list(sys.version_info[:2])))") == [3, 10]


@pytest.mark.parametrize("dimension,role", [("../scene", "visual"), ("scene/x", "visual"), ("scene", "unknown")])
def test_rejects_invalid_environment_identifiers(tmp_path, dimension, role):
    with pytest.raises(ValueError):
        prepare_environment(dimension, role, Path(sys.executable), tmp_path)


def test_env_root_cannot_mutate_base_runtime(base_runtime):
    base, _, _ = base_runtime
    with pytest.raises(ValueError, match="outside the shared base runtime"):
        prepare_environment("scene", "visual", base, base.parent.parent / "children")
