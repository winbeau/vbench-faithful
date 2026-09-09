from __future__ import annotations

from collections.abc import Callable

from .errors import DependencyError
from .schemas import VideoResult


def run_audit_stub(videos: list[str]) -> list[VideoResult]:
    return [VideoResult(video=video, status="not_implemented") for video in videos]


def run_vbench_backend(backend: Callable[[str], VideoResult], videos: list[str]) -> list[VideoResult]:
    results: list[VideoResult] = []
    for video in videos:
        try:
            results.append(backend(video))
        except Exception as exc:  # one corrupt video must not abort the batch
            results.append(VideoResult(video=video, status="failed", error=f"{type(exc).__name__}: {exc}"))
    return results
