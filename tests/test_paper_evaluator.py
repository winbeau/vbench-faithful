"""Contracts that prevent a successful-looking partial or wrong-method evaluation."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_common import PAPER, OFFICIAL, load_inputs, summarize, verify_assets


def test_full_info_preserves_auxiliary_targets_and_each_video(tmp_path):
    for name in ("one.mp4", "two.mp4"):
        (tmp_path / name).write_bytes(name.encode())
    path = tmp_path / "inputs.json"
    aux = {"object_class": {"object": "person"}}
    path.write_text(json.dumps([{"prompt_en": "a human", "dimension": ["object_class"],
                                "video_list": ["one.mp4", "two.mp4"], "auxiliary_info": aux}]))
    rows = load_inputs(path)
    assert len(rows) == 2 and rows[0]["id"] != rows[1]["id"]
    assert all(r["auxiliary_info"] == aux and r["prompt"] == "a human" for r in rows)
    assert all(len(r["video_sha256"]) == 64 for r in rows)


def test_duplicate_ids_and_missing_media_are_rejected(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"video")
    path = tmp_path / "inputs.json"
    row = {"id": "same", "video": "a.mp4", "dimensions": ["dynamic_degree"]}
    path.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="Duplicate input id"):
        load_inputs(path)
    path.write_text(json.dumps([{**row, "video": "missing.mp4"}]))
    with pytest.raises(FileNotFoundError):
        load_inputs(path)


def test_media_substitution_is_rejected_before_scoring(tmp_path):
    import hashlib
    media = tmp_path / "video.mp4"
    media.write_bytes(b"frozen video")
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps([{"video": media.name,
                                    "video_sha256": hashlib.sha256(media.read_bytes()).hexdigest()}]))
    assert len(load_inputs(manifest)) == 1
    media.write_bytes(b"substituted video")
    with pytest.raises(ValueError, match="Input media hash differs"):
        load_inputs(manifest)


def test_missing_score_cannot_change_the_denominator():
    summary = summarize([{"id": "a", "status": "succeeded", "score": .8},
                         {"id": "b", "status": "failed", "score": None}])
    assert summary["input_count"] == 2 and summary["coverage"] == .5
    assert summary["score"] is None and not summary["complete"]
    assert summary["observed_subset_mean"] == .8


@pytest.mark.parametrize("score", [float("nan"), float("inf"), None])
def test_nonfinite_success_is_not_accepted(score):
    with pytest.raises(ValueError, match="finite score"):
        summarize([{"id": "a", "status": "succeeded", "score": score}])


def test_default_routes_bind_all_nine_paper_methods():
    methods = json.loads((ROOT / "configs/reproduction/paper-methods.json").read_text())["methods"]
    assert set(methods) == set(PAPER) and len(PAPER) == 9 and len(OFFICIAL) == 16
    assert methods["dynamic_degree"]["version"] == "aligned-v1"
    assert methods["subject_consistency"]["version"] == "frozen-v5/evaluated-v9"
    assert methods["spatial_relationship"]["step"] == 600
    assert methods["human_action"]["interface"] == "repair-v2.1"


def test_origin_plan_reports_the_official_method(tmp_path):
    import subprocess
    media = tmp_path / "video.mp4"
    media.write_bytes(b"input identity only; plan does not decode media")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"video": media.name, "dimensions": ["scene"]}]))
    assets = tmp_path / "assets.json"
    assets.write_text("{}")
    run = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_vbench.py"),
                          "--input", str(inputs), "--assets", str(assets), "--output", str(tmp_path / "output"),
                          "--dimensions", "scene", "--backend", "origin", "--plan"],
                         capture_output=True, text=True, check=True)
    plan = json.loads(run.stdout)
    assert set(plan["methods"]["scene"]) == {"origin"}
    assert plan["methods"]["scene"]["origin"]["implementation"] == "official VBench 1.0"


def test_changed_adapter_is_rejected_before_loading_any_model(tmp_path):
    folder = tmp_path / "human_action"
    folder.mkdir()
    (folder / "adapter_model.safetensors").write_bytes(b"wrong checkpoint")
    # Replacing the required UMT pin is rejected first, before CUDA is touched.
    with pytest.raises(ValueError, match="frozen paper selection"):
        verify_assets({"umt": folder / "adapter_model.safetensors", "adapters": tmp_path},
                      ["human_action"], "repair")


def test_publication_semantic_source_is_byte_identical():
    # This route needs no vision weights; all integrated runtime sources are pinned.
    checked = verify_assets({}, [], "origin")
    source = json.loads((ROOT / "configs/reproduction/semantic-source.json").read_text())
    assert set(checked) == {str(ROOT / entry["path"]) for entry in source["files"]}
    assert len(checked) == 16


def test_spatial_schema_and_peft_config_are_part_of_verified_cache_identity(tmp_path):
    from paper_common import asset_requirements

    assets = {"grit": tmp_path / "grit", "adapters": tmp_path / "adapters", "base_model": tmp_path / "base"}
    pins = asset_requirements(assets, ["spatial_relationship"], "repair")
    release = json.loads((ROOT / "configs/reproduction/model-release.json").read_text())
    configs = [e for e in release["files"] if e["local_path"].startswith("adapters/spatial_relationship/")
               and e["local_path"].endswith(".json")]
    assert len(configs) == 2
    for entry in configs:
        assert pins[str(tmp_path / entry["local_path"])] == entry["sha256"]


@pytest.mark.parametrize("missing", [True, False])
def test_legacy_or_modified_spatial_schema_is_rejected_before_inference(tmp_path, monkeypatch, missing):
    import paper_common

    assets = {"grit": tmp_path / "grit", "adapters": tmp_path / "adapters", "base_model": tmp_path / "base"}
    pins = paper_common.asset_requirements(assets, ["spatial_relationship"], "repair")
    config = tmp_path / "adapters/spatial_relationship/training_config.json"
    if not missing:
        config.parent.mkdir(parents=True)
        config.write_text('{"spatial_relation_space": "legacy"}')
    original_digest = paper_common.digest
    # Other large model payloads have passed their independent weight checks.
    monkeypatch.setattr(paper_common, "digest", lambda p: original_digest(p) if Path(p) == config else pins[str(p)])
    with pytest.raises(FileNotFoundError if missing else ValueError):
        verify_assets(assets, ["spatial_relationship"], "repair")


def test_parallel_verification_reads_all_bytes_even_if_stat_is_unchanged(tmp_path, monkeypatch):
    import os
    from threading import Barrier
    import paper_common

    files = [tmp_path / str(i) for i in range(2)]
    for path in files:
        path.write_bytes(path.name.encode() * 100)
    pins = [(str(p), paper_common.digest(p)) for p in files]
    monkeypatch.setattr(paper_common, "_asset_requirements", lambda *a: iter(pins))
    assert verify_assets({}, [], "repair") == dict(pins)
    original_digest = paper_common.digest
    barrier = Barrier(2)

    def concurrent_digest(path):
        barrier.wait(timeout=5)
        return original_digest(path)

    monkeypatch.setattr(paper_common, "digest", concurrent_digest)
    assert verify_assets({}, [], "repair", workers=2) == dict(pins)
    stat = files[0].stat()
    files[0].write_bytes(b"x" * stat.st_size)
    os.utime(files[0], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="frozen paper selection"):
        verify_assets({}, [], "repair", workers=2)


@pytest.mark.parametrize("workers", [1, 4])
def test_verification_preserves_first_pin_failure_before_later_config_error(tmp_path, monkeypatch, workers):
    import paper_common
    path = tmp_path / "weight"
    path.write_bytes(b"wrong")
    expected = paper_common.digest(path)

    def requirements(*args):
        yield str(path), expected
        raise KeyError("missing later asset")

    monkeypatch.setattr(paper_common, "_asset_requirements", requirements)
    with pytest.raises(KeyError, match="missing later asset"):
        verify_assets({}, [], "repair", workers=workers)
    path.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="frozen paper selection"):
        verify_assets({}, [], "repair", workers=workers)


def test_recovery_archive_cannot_escape_destination(tmp_path):
    import io
    import tarfile
    from restore_paper_runtime import extract_archive

    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        member = tarfile.TarInfo("visual-env/../../outside")
        member.size = 1
        tf.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unexpected recovery"):
        extract_archive(path, tmp_path / "restore", ["visual-env"])
    assert not (tmp_path / "outside").exists()


def test_recovery_rejects_external_symlinks_but_restores_internal_links(tmp_path):
    import io
    import tarfile
    from restore_paper_runtime import extract_archive

    path = tmp_path / "links.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        item = tarfile.TarInfo("python-3.10/bin/python3.10")
        item.size = 1
        tf.addfile(item, io.BytesIO(b"x"))
        link = tarfile.TarInfo("visual-env/bin/python")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../python-3.10/bin/python3.10"
        tf.addfile(link)
    dest = tmp_path / "restored"
    extract_archive(path, dest, ["python-3.10", "visual-env"])
    assert (dest / "visual-env/bin/python").read_bytes() == b"x"
    with tarfile.open(path, "w:gz") as tf:
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    with pytest.raises(tarfile.FilterError):
        extract_archive(path, tmp_path / "unsafe", ["visual-env"])


def test_source_export_requires_complete_matching_content(tmp_path, monkeypatch):
    import hashlib
    from vbench_audit_core import upstream
    project = tmp_path / "project"
    export = tmp_path / "export"
    export.mkdir()
    source = export / "module.py"
    source.write_text("value = 1\n")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    config = upstream.UpstreamConfig("https://example.test/upstream.git", "a" * 40,
                                    {"scene": upstream.UpstreamSpec("scene", "module", "compute", "module.py", identity)})
    manifest = project / "configs/reproduction/vbench-source.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"repository": config.url, "commit": config.sha,
                                    "files": [{"path": "module.py", "sha256": identity}]}))
    monkeypatch.setattr(upstream, "_workspace_root", lambda: project)
    monkeypatch.setattr(upstream, "load_config", lambda: config)
    state = upstream.verify_upstream(export)
    assert state.source_type == "verified-export" and not state.dirty
    source.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        upstream.verify_upstream(export)
    source.write_text("value = 1\n")
    (export / "injected.py").write_text("value = 3\n")
    with pytest.raises(RuntimeError, match="file coverage differs"):
        upstream.verify_upstream(export)


def test_interrupted_restore_requires_matching_owned_directory(tmp_path, monkeypatch):
    import restore_paper_runtime as restore
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"repo_id": "test/recovery", "archives": []}))
    output = tmp_path / "runtime"
    argv = ["restore", "--manifest", str(manifest), "--output", str(output),
            "--downloads", str(tmp_path / "downloads"), "--archives-only"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(restore, "relocate_environment", lambda *args: None)
    def interrupted(*args):
        raise RuntimeError("interrupted configuration")
    monkeypatch.setattr(restore, "configure_assets", interrupted)
    with pytest.raises(RuntimeError, match="interrupted configuration"):
        restore.main()
    assert (output / "restore-state.json").is_file()
    with pytest.raises(FileExistsError):
        restore.main()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    monkeypatch.setattr(restore, "configure_assets", lambda *args: {})
    restore.main()
    assert json.loads((output / "restore-receipt.json").read_text())["old_container_required"] is False
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="different manifest"):
        restore.main()
