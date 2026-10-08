"""Small JSON-over-Unix-socket transport for a controller-owned worker."""
import json
from pathlib import Path
import socket
import time


class JsonWorker:
    """One sequential caller per worker; process lifetime belongs to Workers."""

    def __init__(self, workers, command, env, log, socket_path):
        self.workers, self.command, self.env = workers, command, env
        self.log, self.socket_path = Path(log), Path(socket_path)
        self.child = None

    def request(self, payload):
        if self.child is None or self.child.poll() is not None:
            if self.child is not None:
                self.workers.finish(self.child)
            self.socket_path.unlink(missing_ok=True)
            self.child = self.workers.start(self.command, self.env, self.log)
        deadline = time.monotonic() + 60
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            while True:
                if self.child.poll() is not None:
                    raise RuntimeError(f"Persistent worker exited {self.child.returncode}; see {self.log}")
                try:
                    connection.connect(str(self.socket_path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Worker startup timed out; see {self.log}")
                    time.sleep(.05)
            connection.sendall(json.dumps(payload).encode() + b"\n")
            with connection.makefile("rb") as reader:
                line = reader.readline()
            if not line:
                raise RuntimeError(f"Worker disconnected; see {self.log}")
            reply = json.loads(line)
            if not reply.get("ok"):
                raise RuntimeError(f"{reply.get('error', 'Worker failed')}; see {self.log}")
            return reply


def serve(socket_path, handler):
    """Serve one controller's sequential requests in its private socket folder."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        listener.listen(1)
        while True:
            connection, _ = listener.accept()
            with connection:
                try:
                    with connection.makefile("rb") as reader:
                        payload = json.loads(reader.readline())
                    reply = {"ok": True, **handler(payload)}
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                connection.sendall(json.dumps(reply, allow_nan=False).encode() + b"\n")
