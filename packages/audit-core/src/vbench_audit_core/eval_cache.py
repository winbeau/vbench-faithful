"""Locked, content-addressed worker artifacts; failed computations never become hits."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import uuid


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


class ArtifactCache:
    """Artifacts stay at their original absolute path, including diagnostic references.

    A run links to an immutable generation. Receipts hash every artifact, so a
    damaged entry is a miss. Rebuilding never replaces an earlier run's evidence.
    A caller must validate coverage/status before returning True from compute.
    """

    def __init__(self, root):
        self.root = Path(root).resolve() / "artifacts-v1"

    def materialize(self, specification, destination, compute, *, reuse=True):
        key = identity(specification)
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if not reuse:
            destination.mkdir(parents=True)
            compute(destination)
            return destination, False, key
        home = self.root / key
        with exclusive_lock(self.root / "locks" / (key + ".lock")):
            receipt_path = home / "receipt.json"
            generation = self._verified(home, receipt_path)
            if generation is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(generation, target_is_directory=True)
                return generation, True, key
            generation = home / uuid.uuid4().hex
            generation.mkdir(parents=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(generation, target_is_directory=True)
            # Preserve failed logs at the linked path, but never publish a hit.
            if compute(generation):
                hashes = {str(p.relative_to(generation)): file_digest(p)
                          for p in sorted(generation.rglob("*")) if p.is_file()}
                if any(p.is_symlink() for p in generation.rglob("*")):
                    raise ValueError("Cache artifacts must be regular files")
                receipt = {"schema": "eval-artifacts/1", "key": key,
                           "generation": generation.name, "files": hashes}
                temp = home / "receipt.json.tmp"
                temp.write_text(json.dumps(receipt, sort_keys=True) + "\n")
                temp.replace(receipt_path)
            return generation, False, key

    @staticmethod
    def _verified(home, receipt_path):
        try:
            receipt = json.loads(receipt_path.read_text())
            name = receipt["generation"]
            if (receipt["schema"] != "eval-artifacts/1" or receipt["key"] != home.name
                    or len(name) != 32 or any(c not in "0123456789abcdef" for c in name)):
                return None
            generation = home / name
            paths = list(generation.rglob("*"))
            if any(p.is_symlink() for p in paths):
                return None
            actual = {str(p.relative_to(generation)): file_digest(p) for p in paths if p.is_file()}
            return generation if actual and actual == receipt["files"] else None
        except (OSError, ValueError, KeyError, TypeError):
            return None
