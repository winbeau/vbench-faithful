"""Protect missing-score semantics and safe restoration of published research."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load("reproduce_main_table")
restore = load("prepare_reproduction")


def record(uid, values, primary=True):
    return {"sample_id": uid, "primary": primary,
            "scores": {key: {"score": value} for key, value in zip(replay.CELLS, values)}}


def test_common_pairs_keep_real_zero_and_never_impute_missing_scores():
    rows = [record("valid", [0, 0.5, 0.8, 0.7]),
            record("undefined", [1, None, 1, 1]),
            record("rejected", [0.9, 0.9, 0.9, 0.9], primary=False)]
    result = replay.summarize(rows)
    assert result["planned_primary"] == 2
    assert result["complete_pairs"] == 1
    assert result["origin_base"] == 0
    assert result["repair_cf"] == 0.7


def test_duplicate_candidates_and_nonfinite_results_fail():
    row = record("a", [1, 1, 1, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        replay.summarize([row, row])
    with pytest.raises(ValueError, match="Invalid score"):
        replay.summarize([record("bad", [1, float("nan"), 1, 1])])


@pytest.mark.parametrize("unsafe", ["../escape.json", "/absolute.json"])
def test_restoration_rejects_paths_outside_the_bundle(tmp_path, unsafe):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tf:
        entry = tarfile.TarInfo(unsafe)
        entry.size = 2
        tf.addfile(entry, io.BytesIO(b"{}"))
    expected = [{"path": unsafe, "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]
    with pytest.raises(ValueError, match="Unsafe"):
        restore.extract_verified(archive, tmp_path / "bundle", expected)
    assert not (tmp_path / "escape.json").exists()


def test_restoration_preserves_existing_modified_work(tmp_path):
    archive = tmp_path / "safe.tar"
    with tarfile.open(archive, "w") as tf:
        entry = tarfile.TarInfo("scores.json")
        entry.size = 2
        tf.addfile(entry, io.BytesIO(b"{}"))
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "scores.json").write_text("user changes")
    expected = [{"path": "scores.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]
    with pytest.raises(ValueError, match="Existing archive member differs"):
        restore.extract_verified(archive, root, expected)
    assert (root / "scores.json").read_text() == "user changes"


def model_releases():
    root = Path(__file__).resolve().parents[1] / "configs/reproduction"
    return (json.loads((root / "release.json").read_text()),
            json.loads((root / "model-release.json").read_text()))


def test_new_model_repository_preserves_all_seven_selected_weights(tmp_path):
    paper, published = model_releases()
    tasks = restore.selected_model_tasks(paper, published, tmp_path)
    assert published["repo_id"] == "winbeau/vbench-faithful"
    assert len(tasks) == 14  # Seven weights, six PEFT configs, and Spatial's schema config.
    weights = [t for t in tasks if t[0]["path"].endswith((".safetensors", ".pt"))]
    assert len(weights) == 7
    assert all(t[2] == published["repo_id"] and t[3] == published["revision"] for t in tasks)
    assert next(t[0] for t in weights if t[0]["path"].endswith("aligned.pt"))["local_path"] == "models/dynamic_degree/aligned.pt"
    schema = next(t[0] for t in tasks if t[0]["path"] == "configs/training/spatial_relationship.json")
    assert schema["local_path"] == "adapters/spatial_relationship/training_config.json"


@pytest.mark.parametrize("damage", ["missing", "missing_schema", "duplicate", "weight_hash", "weight_size", "path", "revision", "source"])
def test_model_migration_cannot_silently_change_the_paper_selection(tmp_path, damage):
    paper, published = model_releases()
    weight = next(e for e in published["files"] if e["path"].endswith(".safetensors"))
    if damage == "missing":
        published["files"].remove(weight)
    elif damage == "missing_schema":
        published["files"] = [e for e in published["files"] if not e["path"].startswith("configs/training/")]
    elif damage == "duplicate":
        published["files"].append(dict(weight))
    elif damage == "weight_hash":
        weight["sha256"] = "0" * 64
    elif damage == "weight_size":
        weight["bytes"] += 1
    elif damage == "path":
        weight["local_path"] = "../escape.safetensors"
    elif damage == "revision":
        published["revision"] = "main"
    else:
        published["source_release"]["revision"] = "0" * 40
    with pytest.raises(ValueError):
        restore.selected_model_tasks(paper, published, tmp_path)


def download_task(root, content=b"selected model", local="models/dynamic_degree/aligned.pt"):
    entry = {"path": "heads/dynamic_degree/aligned.pt", "local_path": local,
             "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    return entry, "model", "winbeau/vbench-faithful", "a" * 40, root


def test_download_uses_hub_path_but_restores_existing_evaluator_layout(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    source.write_bytes(b"selected model")
    calls = []
    def download(repo, path, **kwargs):
        calls.append((repo, path, kwargs))
        return str(source)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    task = download_task(tmp_path / "restored")
    restore.fetch_task(task, "https://huggingface.co")
    restore.fetch_task(task, "https://huggingface.co")
    assert (task[4] / task[0]["local_path"]).read_bytes() == source.read_bytes()
    assert calls == [(task[2], task[0]["path"], {"repo_type": "model", "revision": task[3], "endpoint": "https://huggingface.co"})]


def test_corrupted_remote_model_never_becomes_an_installed_weight(tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    source.write_bytes(b"wrong contents")  # Same length, different hash.
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda *a, **k: str(source)))
    task = download_task(tmp_path / "restored")
    with pytest.raises(ValueError, match="Downloaded content differs"):
        restore.fetch_task(task, "https://huggingface.co")
    assert not (task[4] / task[0]["local_path"]).exists()


def test_changed_local_models_are_preserved_without_downloading(tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("A modified existing model must be rejected before downloading")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=unexpected_download))
    task = download_task(tmp_path)
    target = tmp_path / task[0]["local_path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local training")
    with pytest.raises(ValueError, match="Existing file differs"):
        restore.fetch_task(task, "https://huggingface.co")
    assert target.read_bytes() == b"local training"


def test_restored_spatial_adapter_retains_four_direction_schema(tmp_path, monkeypatch):
    from vbench_prompts_compile.inference import adapter_relation_space

    source = tmp_path / "training_config.json"
    source.write_text('{"spatial_relation_space": "four_directions"}')
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda *a, **k: str(source)))
    task = download_task(tmp_path / "restored", content=source.read_bytes(),
                         local="adapters/spatial_relationship/training_config.json")
    task[0]["path"] = "configs/training/spatial_relationship.json"
    restore.fetch_task(task, "https://huggingface.co")
    assert adapter_relation_space(str(task[4] / "adapters/spatial_relationship")) == "four_directions"


@pytest.mark.parametrize("local", ["../escape.pt", "/absolute.pt", "symlink/escape.pt"])
def test_download_rejects_target_escape_before_network_access(tmp_path, monkeypatch, local):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Unsafe paths must be rejected before downloading")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=unexpected_download))
    root = tmp_path / "restored"
    root.mkdir()
    (root / "symlink").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe restoration path"):
        restore.fetch_task(download_task(root, local=local), "https://huggingface.co")
