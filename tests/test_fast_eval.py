"""Routing, coverage, cache invalidation and process isolation without model weights."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("fast_eval", ROOT / "scripts/eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from vbench_audit_core.eval_cache import ArtifactCache
from vbench_audit_core.eval_config import load_config


def test_all_dimensions_have_explicit_methods_without_fake_repairs():
    assert len(runner.routes(runner.OFFICIAL, "origin")) == 16
    repaired = runner.routes(runner.OFFICIAL, "repair")
    assert len(repaired) == 16
    assert {d for d, b in repaired if b == "repair"} == set(runner.PAPER)
    both = runner.routes(runner.OFFICIAL, "both")
    assert len(both) == 32 and len(set(both)) == 32
    assert ("motion_smoothness", "origin") in both
    assert ("motion_smoothness", "accelerated") in both
    assert ("motion_smoothness", "repair") not in both
    assert runner.routes(runner.OFFICIAL, "ours") == repaired
    assert runner.routes(runner.OFFICIAL, "official") == runner.routes(runner.OFFICIAL, "origin")
    assert {d for d, b in repaired if b == "accelerated"} == set(runner.OFFICIAL) - set(runner.PAPER)


def test_gpu_selection_respects_inherited_visibility_and_unique_devices():
    assert runner.gpu_tokens([0, 2], "2,4,6") == ["2", "6"]
    assert runner.gpu_tokens([0], None) == ["0"]
    with pytest.raises(ValueError, match="outside"):
        runner.gpu_tokens([1], "3")
    with pytest.raises(ValueError, match="outside"):
        runner.gpu_tokens([0], "")
    with pytest.raises(ValueError, match="duplicate"):
        runner.gpu_tokens([0, 1], "3,3")


def test_worker_import_paths_only_include_selected_metric():
    paths = runner.python_paths("dynamic_degree")
    assert ROOT / "metrics/dynamic-degree/src" in paths
    assert ROOT / "metrics/scene/src" not in paths
    assert not any("metrics" in p.parts for p in runner.python_paths(upstream_only=True))


def test_relative_runtime_launcher_preserves_virtual_environment(tmp_path):
    launcher = tmp_path / "visual-env/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"visual_python": "visual-env/bin/python"}))
    assert runner.load_assets(assets)["visual_python"] == str(launcher)


def test_output_symlink_cannot_reach_frozen_research(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(ROOT / "results", target_is_directory=True)
    with pytest.raises(ValueError, match="frozen"):
        runner.writable_path(alias / "new")


def test_cache_reuses_verified_artifacts_and_keeps_old_generation(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    calls = []
    def compute(folder):
        calls.append(folder)
        (folder / "result.json").write_text('{"score": 0.5}')
        return True
    old, hit, _ = cache.materialize({"media": "A", "model": "M"}, tmp_path / "first", compute)
    assert not hit
    same, hit, _ = cache.materialize({"media": "A", "model": "M"}, tmp_path / "second", compute)
    assert hit and same == old and len(calls) == 1
    (old / "result.json").write_text('{"score": 999}')
    fresh, hit, _ = cache.materialize({"media": "A", "model": "M"}, tmp_path / "third", compute)
    assert not hit and fresh != old and (tmp_path / "first").resolve() == old
    for changed in ({"media": "B", "model": "M"}, {"media": "A", "model": "N"}):
        _, hit, _ = cache.materialize(changed, tmp_path / str(len(calls)), compute)
        assert not hit


def test_failed_stage_is_never_a_cache_hit(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    def incomplete(folder):
        (folder / "result.json").write_text('{"score": null}')
        return False
    for i in range(2):
        _, hit, _ = cache.materialize({"same": True}, tmp_path / str(i), incomplete)
        assert not hit


def test_native_official_aggregate_is_preserved_and_invalid_rows_rejected():
    original = {"id": "a", "video": "a.mp4", "video_sha256": "abc", "prompt": "a"}
    payload = {"rows": [{**original, "score": 70.0, "status": "succeeded"}], "official_aggregate": .7}
    assert runner.validate_payload(payload, [original])["score"] == .7
    payload["rows"].append(payload["rows"][0])
    with pytest.raises(ValueError, match="coverage"):
        runner.validate_payload(payload, [original])
    payload["rows"].pop()
    payload["rows"][0]["video_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        runner.validate_payload(payload, [original])


def test_controller_cache_hit_and_metadata_change(tmp_path, monkeypatch, fake_official_transport):
    fake_official_transport(runner)
    from vbench_audit_core import upstream
    (tmp_path / "video.mp4").write_bytes(b"identity fixture; fake worker does not decode")
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps([{"id": "video", "video": "video.mp4", "prompt": "red car",
                                     "dimensions": ["imaging_quality"]}]))
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"visual_python": sys.executable, "vbench": str(tmp_path),
                                 "vbench_cache": str(tmp_path)}))
    yaml = tmp_path / "eval.yaml"
    yaml.write_text("input: input.json\nassets: assets.json\ndimensions: [imaging_quality]\n"
                    "backend: both\noutput: first\ncache_dir: cache\nenv_dir: envs\n")
    config = load_config(yaml)
    pins = lambda assets, dimensions, backend, **kwargs: {dim: "pin-" + dim for dim in dimensions}
    monkeypatch.setattr(runner, "verify_assets", pins)
    monkeypatch.setattr(runner, "asset_requirements", pins)
    monkeypatch.setattr(runner, "source_identity", lambda *a: "code")
    monkeypatch.setattr(runner, "probe_gpu", lambda *a: {"name": "fake CUDA test"})
    monkeypatch.setattr(runner, "prepare_environment", lambda *a: Path(sys.executable))
    monkeypatch.setattr(upstream, "verify_upstream", lambda *a: SimpleNamespace(sha="pinned"))
    calls = []
    def run(self, command, env, log):
        calls.append(command)
        args = list(map(str, command))
        rows = json.loads(Path(args[args.index("--input") + 1]).read_text())
        destination = Path(args[args.index("--output") + 1])
        aggregate = ({"aggregate": .7, "aggregation": "vbench_native_reducer"}
                     if args[args.index("--mode") + 1] == "accelerated" else {"official_aggregate": .7})
        runner.write_json(destination, {"rows": [runner.result(row, 70.0) for row in rows], **aggregate})
        Path(log).write_text("fake worker\n")
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["VBENCH_EVAL_DIMENSION"] in {"imaging_quality", "temporal_flickering"}
    monkeypatch.setattr(runner.Workers, "run", run)
    assert runner.evaluate(config) == 0
    assert runner.evaluate(replace(config, output=tmp_path / "second")) == 0
    assert len(calls) == 2
    report = json.loads((tmp_path / "second/summary.json").read_text())
    assert report["official_fallback"] == []
    assert report["accelerated_dimensions"] == ["imaging_quality"]
    assert report["dimensions"]["imaging_quality/accelerated"]["score"] == .7
    summary = report["dimensions"]["imaging_quality/origin"]
    assert summary["score"] == .7 and summary["cached"]
    execution = json.loads((tmp_path / "second/imaging_quality/origin.json").read_text())["execution"]
    assert execution["cache"][0]["worker_seconds"] == 0
    assert execution["cache"][0]["elapsed_seconds"] >= execution["cache"][0]["environment_seconds"] >= 0
    assert report["timing_seconds"]["total_before_dispatch"] >= report["timing_seconds"]["asset_verification"] >= 0
    # Adding another dimension changes the verified asset union, but must not
    # invalidate an identical existing task's independently bound cache entry.
    data = json.loads(manifest.read_text())
    data.append({**data[0], "id": "other", "dimensions": ["temporal_flickering"]})
    manifest.write_text(json.dumps(data))
    assert runner.evaluate(replace(config, output=tmp_path / "expanded",
                                   dimensions=("imaging_quality", "temporal_flickering"))) == 0
    assert len(calls) == 4
    expanded = json.loads((tmp_path / "expanded/summary.json").read_text())["dimensions"]
    assert expanded["imaging_quality/origin"]["cached"]
    assert not expanded["temporal_flickering/origin"]["cached"]
    data = json.loads(manifest.read_text()); data[0]["prompt"] = "blue car"
    manifest.write_text(json.dumps(data))
    assert runner.evaluate(replace(config, output=tmp_path / "third")) == 0
    assert len(calls) == 6


def test_accelerated_native_scale_required_and_partial_mean_not_reported():
    original = {"id": "one", "video": "one.mp4", "video_sha256": "hash", "prompt": "p"}
    payload = {"rows": [runner.result(original, 70.)]}
    with pytest.raises(ValueError, match="native reducer"):
        runner.validate_payload(payload, [original], "accelerated")
    payload.update(aggregate=.7, aggregation="vbench_native_reducer")
    assert runner.validate_payload(payload, [original], "accelerated")["score"] == .7
    payload["rows"][0].update(score=None, status="failed")
    assert runner.validate_payload(payload, [original], "accelerated")["score"] is None


def test_interrupted_worker_group_is_reaped(tmp_path):
    import os
    import threading
    import time
    workers = runner.Workers()
    errors = []
    def run():
        try:
            workers.run([sys.executable, "-c", "import time;time.sleep(30)"], os.environ.copy(), tmp_path / "worker.log")
        except RuntimeError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while not workers.processes and time.monotonic() < deadline:
        time.sleep(.01)
    workers.stop(); thread.join(timeout=5)
    assert not thread.is_alive() and not workers.processes and errors
