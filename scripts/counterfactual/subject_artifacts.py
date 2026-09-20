"""Lossless, hashed artifacts; no writes to frozen research trees."""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, sha256_file


def canonical_json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def object_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(row) for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"unsafe artifact identifier: {value!r}")
    return value


def safe_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    for tree in (ROOT, ROOT / "data", ROOT / "results", ROOT / "splits", ROOT / "runs", ROOT.parent / "VBench"):
        if path == tree or (tree != ROOT and tree in path.parents):
            raise ValueError(f"refusing to write to protected path: {path}")
    return path


def new_output(path: Path) -> Path:
    path = safe_output(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("artifact path escapes dataset root")
    return path


def write_npz(path: Path, **arrays) -> None:
    """Explicit ZIP timestamps/order; avoid wall-clock metadata in replay hashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def write_png_sequence(root: Path, relative: str, frames: np.ndarray) -> list[dict]:
    directory = artifact_path(root, relative)
    directory.mkdir(parents=True, exist_ok=False)
    files = []
    for index, frame in enumerate(frames):
        path = directory / f"{index:06d}.png"
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise OSError("PNG encoding failed")
        path.write_bytes(encoded.tobytes())
        decoded = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if not np.array_equal(decoded, frame):
            raise AssertionError("lossless PNG round trip failed")
        files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return files


def read_png_sequence(root: Path, files: list[dict]) -> np.ndarray:
    frames = []
    for entry in files:
        path = artifact_path(root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"invalid PNG: {path}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def upstream_frames(path: Path) -> np.ndarray:
    """All frames, native resolution, exactly the pinned VBench decode path."""
    from subject_consistency.backends.vbench import import_official_module
    from subject_consistency.metric import upstream_path

    module, _ = import_official_module(upstream_path())
    tensor = module.load_video(str(path))
    return tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
