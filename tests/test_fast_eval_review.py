"""Regression boundaries found while reviewing the cached evaluation controller."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from vbench_audit_core.eval_config import load_config


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("fast_eval_review", ROOT / "scripts/eval.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_relative_model_python_preserves_venv_symlink(tmp_path):
    python = tmp_path / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"visual_python": "runtime/bin/python",
                                 "semantic_python": "runtime/bin/python"}))
    loaded = runner.load_assets(assets)
    assert loaded["visual_python"] == str(python)
    assert loaded["semantic_python"] == str(python)


def test_origin_without_native_aggregate_cannot_succeed_or_cache(tmp_path, monkeypatch):
    from vbench_audit_core import upstream

    (tmp_path / "video.mp4").write_bytes(b"fake worker does not decode")
    (tmp_path / "input.json").write_text(json.dumps([
        {"id": "one", "video": "video.mp4", "dimensions": ["imaging_quality"]}
    ]))
    (tmp_path / "assets.json").write_text(json.dumps({
        "visual_python": sys.executable, "vbench": str(tmp_path), "vbench_cache": str(tmp_path)
    }))
    yaml = tmp_path / "eval.yaml"
    yaml.write_text("input: input.json\nassets: assets.json\ndimensions: [imaging_quality]\n"
                    "backend: origin\noutput: first\ncache_dir: cache\nenv_dir: envs\n")
    config = load_config(yaml)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runner, "verify_assets", lambda *a: {})
    monkeypatch.setattr(runner, "asset_requirements", lambda *a: {})
    monkeypatch.setattr(runner, "source_identity", lambda *a: "review-fixture")
    monkeypatch.setattr(runner, "probe_gpu", lambda *a: {"name": "fake GPU"})
    monkeypatch.setattr(runner, "prepare_environment", lambda *a: Path(sys.executable))
    monkeypatch.setattr(upstream, "verify_upstream", lambda *a: SimpleNamespace(sha="pinned"))
    calls = []

    def run(self, command, env, log):
        calls.append(command)
        args = list(map(str, command))
        rows = json.loads(Path(args[args.index("--input") + 1]).read_text())
        destination = Path(args[args.index("--output") + 1])
        # Imaging quality's native reducer is not the mean of these raw values.
        # A malformed response must not silently switch the aggregation contract.
        runner.write_json(destination, {"rows": [runner.result(row, 70.0) for row in rows]})
        Path(log).write_text("malformed origin response\n")

    monkeypatch.setattr(runner.Workers, "run", run)
    codes = [runner.evaluate(config), runner.evaluate(replace(config, output=tmp_path / "second"))]
    reports = [json.loads((tmp_path / name / "summary.json").read_text())
               for name in ("first", "second")]
    assert codes == [1, 1]
    assert len(calls) == 2
    for report in reports:
        summary = report["dimensions"]["imaging_quality/origin"]
        assert not report["complete"] and summary["score"] is None


def _live_process(pid):
    """Linux zombie descendants no longer execute or own GPU resources."""
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group cleanup contract")
def test_stop_kills_descendant_even_after_worker_leader_exits(tmp_path):
    pidfile = tmp_path / "descendant.pid"
    child_code = ("import os, pathlib, signal, time; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                  "time.sleep(30)")
    parent_code = ("import subprocess, sys, time; "
                   f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                   "time.sleep(30)")
    workers = runner.Workers()
    errors = []

    def run():
        try:
            workers.run([sys.executable, "-c", parent_code], os.environ.copy(), tmp_path / "worker.log")
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=run)
    thread.start()
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pidfile.is_file() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pidfile.is_file(), "worker descendant did not start"
        child_pid = int(pidfile.read_text())
        workers.stop()
        thread.join(timeout=5)
        deadline = time.monotonic() + 1
        while _live_process(child_pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not thread.is_alive() and errors
        assert not _live_process(child_pid), "SIGTERM-resistant descendant survived group shutdown"
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        workers.stop()
        thread.join(timeout=5)
