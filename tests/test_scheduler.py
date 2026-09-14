from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

# ``multiprocessing`` spawn imports this module by its package name. pytest's
# importlib mode does not always put the repository root on child sys.path, so
# make the test package discoverable without changing production scheduling.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ["PYTHONPATH"] = _ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from vbench_audit_core.coordinator import run_spawn_coordinator


def fake_worker(result_path: str, gpu_id: int, videos: list[str], backend: str, delay: float = 0.03) -> None:
    started = time.time()
    time.sleep(delay)
    results = [
        {"video": video, "backend": backend, "status": "succeeded", "score": float(gpu_id)}
        for video in videos
    ]
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results, "pid": os.getpid(), "started": started, "ended": time.time()}, handle)


def malformed_worker(result_path: str, gpu_id: int, videos: list[str], backend: str) -> None:
    if gpu_id == 1:
        videos = videos + ["extra"]
    fake_worker(result_path, gpu_id, videos, backend, delay=0.0)


def crash_worker(result_path: str, gpu_id: int, videos: list[str], backend: str) -> None:
    if gpu_id == 1:
        raise RuntimeError("simulated worker crash")
    fake_worker(result_path, gpu_id, videos, backend, delay=0.0)


def barrier_worker(result_path: str, gpu_id: int, videos: list[str], backend: str, barrier) -> None:
    started = time.time()
    barrier.wait(timeout=5)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"results": [{"video": video, "backend": backend, "status": "succeeded", "score": 1.0} for video in videos], "started": started, "ended": time.time()}, handle)


def test_spawn_coordinator_runs_nonempty_shards_and_restores_order() -> None:
    result = run_spawn_coordinator(fake_worker, ["v3", "v1", "v2", "v4", "v5"], [0, 2, 4], worker_args=("audit",), label="test")
    assert [item["video"] for item in result.results] == ["v3", "v1", "v2", "v4", "v5"]
    assert len(result.workers) == 3
    assert {video for worker in result.workers for video in worker.videos} == {"v1", "v2", "v3", "v4", "v5"}
    assert len({worker.pid for worker in result.workers}) == 3


def test_spawn_workers_reach_barrier_before_any_join() -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    result = run_spawn_coordinator(barrier_worker, ["v1", "v2", "v3"], [0, 1, 2], worker_args=("audit", barrier), label="barrier")
    assert len(result.workers) == 3
    assert all(worker.exitcode == 0 for worker in result.workers)
    assert max(worker.started_at or 0 for worker in result.workers) <= min(worker.ended_at or 0 for worker in result.workers)


def test_bad_shard_is_failed_without_discarding_good_shard() -> None:
    result = run_spawn_coordinator(malformed_worker, ["v1", "v2"], [0, 1], worker_args=("audit",), label="test")
    assert result.results[0]["status"] == "succeeded"
    assert result.results[1]["status"] == "failed"


def test_crashed_worker_is_failed_without_discarding_good_shard() -> None:
    result = run_spawn_coordinator(crash_worker, ["v1", "v2"], [0, 1], worker_args=("audit",), label="test")
    assert result.results[0]["status"] == "succeeded"
    assert result.results[1]["status"] == "failed"


def test_start_failure_terminates_and_joins_started_workers(monkeypatch) -> None:
    import vbench_audit_core.coordinator as coordinator

    class FakeProcess:
        instances = []

        def __init__(self, **kwargs):
            self.exitcode = None
            self.pid = len(self.instances) + 100
            self.started = False
            self.terminated = False
            self.joined = False
            self.instances.append(self)

        def start(self):
            if len(self.instances) == 2:
                raise KeyboardInterrupt()
            self.started = True

        def is_alive(self):
            return self.started and not self.terminated

        def terminate(self):
            self.terminated = True

        def join(self):
            self.joined = True

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(coordinator.multiprocessing, "get_context", lambda name: FakeContext())
    import pytest
    with pytest.raises(KeyboardInterrupt):
        coordinator.run_spawn_coordinator(fake_worker, ["v1", "v2"], [0, 1], worker_args=("audit",), label="start")
    assert FakeProcess.instances[0].terminated
    assert FakeProcess.instances[0].joined


def test_external_torchrun_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    import pytest
    with pytest.raises(RuntimeError, match="external distributed"):
        run_spawn_coordinator(fake_worker, ["v1"], [0], worker_args=("audit",), label="test")


def test_subject_transition_weighting_is_gpu_count_independent() -> None:
    from subject_consistency.cli import _official_dataset_score

    results = [
        {"status": "succeeded", "score": 1.0, "diagnostics": {"num_frames": 2}},
        {"status": "succeeded", "score": 0.0, "diagnostics": {"num_frames": 6}},
    ]
    assert _official_dataset_score(results, 1) == _official_dataset_score(results, 4)
    assert _official_dataset_score(results, 4) == (1 / 6, 6)
