import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("publication_archive", Path(__file__).parents[1] / "scripts/archive_counterfactuals.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_deferred_commit_failure_resumes_without_losing_members(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "clip.mp4").write_bytes(b"preserved source fixture")
    config = tmp_path / "spec.json"
    config.write_text(json.dumps({"experiments": [{"dimension": "dynamic_degree", "version": "test-v1",
                      "root": str(source), "include": ["clip.mp4"]}]}))
    args = SimpleNamespace(spec=config, work=tmp_path / "work", defer_commit=True,
                           endpoint="https://example.invalid", repo_id="owner/repo", shard_mib=1, not_before=0)
    uploaded = []
    fail = True

    class Operation:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj
            self._upload_mode = "lfs" if path_in_repo.endswith(".tar") else "regular"
            self._is_uploaded = False

    class API:
        def __init__(self, **kwargs):
            pass

        def preupload_lfs_files(self, repo, additions, **kwargs):
            for op in additions:
                if op._upload_mode == "lfs" and not op._is_uploaded:
                    uploaded.append(archive.digest(op.path_or_fileobj))
                    op._is_uploaded = True
                    op.path_or_fileobj = b""

        def create_commit(self, **kwargs):
            assert not list(args.work.rglob("*.tar"))
            if fail:
                raise RuntimeError("commit temporarily unavailable")
            return SimpleNamespace(oid="verified-test-commit")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, CommitOperationAdd=Operation))
    with pytest.raises(RuntimeError, match="commit temporarily"):
        archive.run(args)
    state_file = args.work / "dynamic_degree/test-v1/state.json"
    pending = json.loads(state_file.read_text())
    assert len(pending["pending_archives"]) == len(pending["pending_members"]) == 1
    assert pending["archives"] == pending["members"] == []
    fail = False
    archive.run(args)
    state = json.loads(state_file.read_text())
    assert len(state["archives"]) == len(state["members"]) == 1
    assert uploaded[0] == uploaded[1]
    assert (source / "clip.mp4").read_bytes() == b"preserved source fixture"
    assert state["commit_sha"] == "verified-test-commit"
    archive.run(args)
    assert len(uploaded) == 2  # Already committed archives need no rebuild.
    assert len(json.loads(state_file.read_text())["members"]) == 1
