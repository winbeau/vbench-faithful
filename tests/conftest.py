"""Model-free transport substitution for controller routing/coverage tests."""
import pytest


@pytest.fixture
def fake_official_transport(monkeypatch):
    def install(runner):
        class Transport:
            def __init__(self, workers, command, env, log, socket_path):
                self.workers, self.env, self.log = workers, env, log
            def request(self, payload):
                command = ["fake-worker", "--mode", "origin"]
                for key, value in payload.items():
                    command.extend(["--" + key, value])
                self.workers.run(command, {**self.env, "VBENCH_EVAL_DIMENSION": payload["dimension"]}, self.log)
                return {"ok": True, "pid": 1, "compute_seconds": 0.}
        monkeypatch.setattr(runner, "JsonWorker", Transport)
    return install
