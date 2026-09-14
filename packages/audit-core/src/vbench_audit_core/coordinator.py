"""Small, dependency-free spawn coordinator used by metric runners.

Workers receive only picklable values and write one JSON envelope.  The
parent starts every non-empty shard before joining any process, validates the
complete key set, and restores the caller's input order.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .devices import round_robin_shards

_LAST_SCHEDULE: dict[str, Any] = {"workers": []}


@dataclass(frozen=True)
class WorkerInfo:
    gpu_id: int
    videos: tuple[str, ...]
    pid: int | None
    started_at: float | None
    ended_at: float | None
    exitcode: int | None
    result_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "videos": list(self.videos),
            "pid": self.pid,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exitcode": self.exitcode,
            "result_path": self.result_path,
        }


@dataclass(frozen=True)
class CoordinatorResult:
    results: list[dict[str, Any]]
    workers: tuple[WorkerInfo, ...]

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.results[index]

    def scheduling_record(self) -> dict[str, Any]:
        return {"workers": [worker.to_dict() for worker in self.workers]}


def _failed(video: str, backend: str, reason: str) -> dict[str, Any]:
    return {
        "video": video,
        "backend": backend,
        "score": None,
        "status": "failed",
        "failure_reason": reason,
        "error": reason,
    }


def _timed_worker_entry(
    worker: Callable[..., None], result_path: str, gpu_id: int, shard: list[str],
    worker_args: tuple[Any, ...], worker_kwargs: dict[str, Any], backend: str,
) -> None:
    """Spawn-safe wrapper recording worker-clock timing even on exceptions."""
    started = time.time()
    try:
        worker(result_path, gpu_id, shard, *worker_args, **worker_kwargs)
    except BaseException as exc:
        reason = f"worker {gpu_id} crashed: {type(exc).__name__}: {exc}"
        Path(result_path).write_text(
            json.dumps({"results": [_failed(video, backend, reason) for video in shard],
                        "worker_error": reason, "worker_started_at": started,
                        "worker_ended_at": time.time()}, ensure_ascii=False),
            encoding="utf-8",
        )
        raise
    ended = time.time()
    path = Path(result_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"results": []}
    except (OSError, json.JSONDecodeError):
        payload = {"results": []}
    if isinstance(payload, list):
        payload = {"results": payload}
    if not isinstance(payload, dict):
        payload = {"results": []}
    payload["worker_started_at"] = started
    payload["worker_ended_at"] = ended
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_spawn_coordinator(
    worker: Callable[..., None],
    videos: Sequence[str | Path],
    gpu_ids: Sequence[int],
    worker_args: Iterable[Any] = (),
    *,
    worker_kwargs: Mapping[str, Any] | None = None,
    backend: str = "unknown",
    label: str = "metric",
) -> CoordinatorResult:
    """Run a top-level worker once per non-empty round-robin shard.

    The worker must accept ``(result_path, gpu_id, shard, *worker_args,
    **worker_kwargs)`` and write either a JSON list or ``{"results": list}``.
    A bad worker shard becomes explicit failed records; unrelated shards are
    retained.  Duplicate, extra, or missing video keys raise ``RuntimeError``
    because those results cannot be trusted.
    """

    expected = [str(video) for video in videos]
    if not expected:
        return CoordinatorResult([], ())
    world_size = os.environ.get("WORLD_SIZE")
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if (world_size and world_size not in {"", "1"}) or rank not in (None, "", "0") or local_rank not in (None, "", "0"):
        raise RuntimeError(
            "external distributed environment detected (WORLD_SIZE>1); "
            "invoke the metric outside torchrun so the coordinator owns sharding"
        )
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique")
    shards = round_robin_shards(expected, list(gpu_ids))
    context = multiprocessing.get_context("spawn")
    kwargs = dict(worker_kwargs or {})
    args = tuple(worker_args)
    processes: list[tuple[Any, int, list[str], Path, float]] = []
    infos: list[WorkerInfo] = []
    collected: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"{label}-{backend}-workers-") as root:
        root_path = Path(root)
        try:
            # Start all workers first: joining while starting would accidentally
            # serialize scene/overall/motion evaluations.
            for gpu_id, shard_items in shards.items():
                shard = [str(item) for item in shard_items]
                result_path = root_path / f"gpu-{gpu_id}.json"
                started = time.time()
                process = context.Process(
                    target=_timed_worker_entry,
                    args=(worker, str(result_path), int(gpu_id), shard, args, kwargs, backend),
                )
                process.start()
                processes.append((process, int(gpu_id), shard, result_path, started))
            # All processes have been started above before the first join.
            for process, gpu_id, shard, result_path, started in processes:
                process.join()
                ended = time.time()
                payload: Any = None
                parse_reason: str | None = None
                if result_path.is_file():
                    try:
                        payload = json.loads(result_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        parse_reason = f"{label} worker gpu={gpu_id} produced invalid JSON: {exc}"
                else:
                    parse_reason = f"{label} worker gpu={gpu_id} exited with code {process.exitcode} without a result"
                if isinstance(payload, list):
                    payload = {"results": payload}
                payload_results = payload.get("results") if isinstance(payload, dict) else None
                worker_error = payload.get("worker_error") if isinstance(payload, dict) else None
                actual_started = payload.get("worker_started_at", started) if isinstance(payload, dict) else started
                actual_ended = payload.get("worker_ended_at", ended) if isinstance(payload, dict) else ended
                infos.append(WorkerInfo(gpu_id, tuple(shard), process.pid, actual_started, actual_ended, process.exitcode, str(result_path)))
                reason = parse_reason or worker_error or (
                    f"{label} worker gpu={gpu_id} exited with code {process.exitcode}"
                    if process.exitcode not in (0, None) else None
                )
                valid = isinstance(payload_results, list) and all(isinstance(item, dict) for item in payload_results)
                keys = [item.get("video") for item in payload_results] if valid else []
                valid = valid and all(isinstance(key, str) for key in keys)
                valid = valid and all(isinstance(item.get("status"), str) for item in payload_results)
                duplicate = valid and len(keys) != len(set(keys))
                wrong_keys = valid and set(keys) != set(shard)
                if reason or not valid or duplicate or wrong_keys:
                    detail = reason or f"{label} worker gpu={gpu_id} returned invalid shard keys"
                    collected.extend(_failed(video, backend, detail) for video in shard)
                else:
                    collected.extend(payload_results)
        except BaseException:
            # Covers start failures as well as Ctrl-C; no child may survive
            # this coordinator call.
            for process, *_ in processes:
                if process.is_alive():
                    process.terminate()
            for process, *_ in processes:
                process.join()
            raise

    expected_set = set(expected)
    by_video: dict[str, dict[str, Any]] = {}
    for result in collected:
        video = result.get("video")
        if not isinstance(video, str):
            raise RuntimeError(f"{label} worker returned a result without a video key")
        if video not in expected_set:
            raise RuntimeError(f"{label} worker returned an extra video result: {video}")
        if video in by_video:
            raise RuntimeError(f"{label} worker returned duplicate result: {video}")
        by_video[video] = result
    missing = [video for video in expected if video not in by_video]
    if missing:
        raise RuntimeError(f"{label} worker results missing videos: {missing}")
    coordinated = CoordinatorResult([by_video[video] for video in expected], tuple(infos))
    global _LAST_SCHEDULE
    _LAST_SCHEDULE = coordinated.scheduling_record()
    return coordinated


def last_schedule() -> dict[str, Any]:
    """Return the most recent coordinator schedule in this process."""
    return {"workers": [dict(item) for item in _LAST_SCHEDULE.get("workers", [])]}


# Short alias for callers and external research scripts.
coordinate_spawn = run_spawn_coordinator
