"""Exercise actual persistent processes, failure isolation and cancellation."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval import Workers
from vbench_audit_core.worker_rpc import JsonWorker


SERVER = '''
import os, sys, time
from pathlib import Path
from vbench_audit_core.worker_rpc import serve
count = 0
def handle(request):
    global count
    if request.get('fail'):
        raise ValueError('declared failure')
    if request.get('crash'):
        os._exit(7)
    if request.get('wait'):
        Path(request['marker']).touch()
        time.sleep(30)
    count += 1
    return {'pid': os.getpid(), 'count': count}
serve(sys.argv[1], handle)
'''


def test_sequential_requests_share_process_and_errors_do_not_lose_later_work(tmp_path):
    workers = Workers()
    with tempfile.TemporaryDirectory(prefix="test-vbench-rpc-") as directory:
        socket = Path(directory) / "w.sock"
        worker = JsonWorker(workers, [sys.executable, "-c", SERVER, socket], os.environ.copy(), tmp_path / "worker.log", socket)
        try:
            first, second = worker.request({}), worker.request({})
            assert first["pid"] == second["pid"] and second["count"] == 2
            with pytest.raises(RuntimeError, match="declared failure"):
                worker.request({"fail": True})
            assert worker.request({})["count"] == 3
            with pytest.raises(RuntimeError, match="disconnected"):
                worker.request({"crash": True})
            worker.child.wait(timeout=5)
            assert worker.request({})["count"] == 1
        finally:
            workers.stop()
        assert not workers.processes and worker.child.poll() is not None


def test_cancellation_unblocks_request_and_reaps_owned_worker(tmp_path):
    workers, errors = Workers(), []
    marker = tmp_path / "started"
    with tempfile.TemporaryDirectory(prefix="test-vbench-rpc-") as directory:
        socket = Path(directory) / "w.sock"
        worker = JsonWorker(workers, [sys.executable, "-c", SERVER, socket], os.environ.copy(), tmp_path / "worker.log", socket)
        def run():
            try:
                worker.request({"wait": True, "marker": str(marker)})
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=run)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists()
        finally:
            workers.stop()
            thread.join(timeout=5)
        assert not thread.is_alive() and errors and not workers.processes
