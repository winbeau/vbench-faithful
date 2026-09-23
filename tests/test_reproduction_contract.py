"""Protect missing-score semantics and safe restoration of published research."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

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
