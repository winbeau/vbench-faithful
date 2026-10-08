"""Sharing model outputs must not bypass protocol identity or cache integrity."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from vbench_audit_core.run_inference_cache import RunInferenceCache


def good(value):
    return value["status"] == "succeeded"


def test_identity_isolates_model_protocol_runtime_and_run(tmp_path):
    context = {"model": "weights", "task": "ObjectDet", "source": "pin", "runtime": "torch"}
    frame = {"sha256": "pixels", "dtype": "uint8", "shape": [2, 2, 3]}
    calls = []

    def compute():
        calls.append(1)
        return {"status": "succeeded", "detections": [1, 2]}

    first = RunInferenceCache(tmp_path / "run", context)
    assert first.get_or_compute(frame, compute, valid=good)["detections"] == [1, 2]
    second = RunInferenceCache(tmp_path / "run", context)
    second.get_or_compute(frame, compute, valid=good)
    assert len(calls) == 1 and second.hits == 1
    for key in context:
        RunInferenceCache(tmp_path / "run", {**context, key: "changed"}).get_or_compute(frame, compute, valid=good)
    for key, value in [("sha256", "other pixels"), ("dtype", "float32"), ("shape", [1, 4, 3])]:
        first.get_or_compute({**frame, key: value}, compute, valid=good)
    RunInferenceCache(tmp_path / "new-run", context).get_or_compute(frame, compute, valid=good)
    assert len(calls) == 9


def test_corruption_recomputes_and_failures_never_become_hits(tmp_path):
    cache = RunInferenceCache(tmp_path, {"model": "M"})
    calls = []

    def compute():
        calls.append(1)
        return {"status": "succeeded", "scoreless": True}

    cache.get_or_compute("pixels", compute, valid=good)
    path = next(cache.root.glob("*.json"))
    path.write_text(path.read_text().replace('"scoreless": true', '"scoreless": false'))
    assert cache.get_or_compute("pixels", compute, valid=good)["scoreless"]
    assert len(calls) == 2 and cache.misses == 2
    for _ in range(2):
        cache.get_or_compute("failed pixels", lambda: {"status": "failed"}, valid=good)
    assert cache.hits == 0 and cache.misses == 4
    with pytest.raises(RuntimeError):
        cache.get_or_compute("exception", lambda: (_ for _ in ()).throw(RuntimeError()), valid=good)


def test_concurrent_consumers_only_infer_once(tmp_path):
    start = Barrier(2)
    calls = []

    def work(_):
        cache = RunInferenceCache(tmp_path, {"runtime": "same"})
        start.wait(timeout=5)
        def compute():
            calls.append(1)
            return {"status": "succeeded", "detections": []}
        return cache.get_or_compute("frame", compute, valid=good)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(work, range(2))) == [{"status": "succeeded", "detections": []}] * 2
    assert len(calls) == 1
