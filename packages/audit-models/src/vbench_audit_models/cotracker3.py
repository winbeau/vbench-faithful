"""Local-only, hash-bound CoTracker3 offline adapter; no motion score.

This module never uses torch.hub or a download helper. Its query coordinates
and model visibility are predictions, not independent motion certificates.
No existing CoTracker2 adapter or public metric default is changed.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import re
import sys
from collections.abc import Mapping

import numpy as np

from .point_tracker import CoTracker2Model


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_identity(source_root: Path) -> dict[str, str]:
    """Inventory the complete local Python namespace without importing it."""
    root = Path(source_root).resolve()
    required = {"cotracker/predictor.py", "cotracker/models/build_cotracker.py",
                "cotracker/models/core/cotracker/cotracker3_offline.py"}
    paths = sorted((root / "cotracker").rglob("*.py"))
    files = {}
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("tracker source symlink escapes the configured checkout")
        files[str(path.relative_to(root))] = file_sha256(path)
    if not required <= files.keys():
        raise FileNotFoundError("local CoTracker3 predictor, builder and offline model sources required")
    return files


def verify_local_assets(source_root, checkpoint, *, expected_source, expected_sha256, expected_size_bytes):
    if (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or type(expected_size_bytes) is not int or expected_size_bytes <= 0
            or not isinstance(expected_source, Mapping) or not expected_source):
        raise ValueError("explicit checkpoint SHA/size and complete source manifest required")
    root, weight = Path(source_root).resolve(), Path(checkpoint).resolve()
    if not weight.is_file():
        raise FileNotFoundError(f"local checkpoint missing (no automatic download): {weight}")
    if weight.stat().st_size != expected_size_bytes or file_sha256(weight) != expected_sha256:
        raise ValueError("local checkpoint differs from the pinned size/SHA")
    actual = source_identity(root)
    if actual != dict(expected_source):
        raise ValueError("complete tracker source manifest differs")
    return {"model": "CoTracker3 offline", "source_root": str(root), "source_files": actual,
            "checkpoint": str(weight), "checkpoint_sha256": expected_sha256,
            "checkpoint_size_bytes": expected_size_bytes,
            "predictor_args": {"v2": False, "offline": True, "window_len": 60}}


def _check_imported_namespace(root, expected_source):
    for name, module in tuple(sys.modules.items()):
        if name != "cotracker" and not name.startswith("cotracker."):
            continue
        if module is None:
            raise RuntimeError("an incomplete cotracker module is already imported")
        filename = getattr(module, "__file__", None)
        if filename is not None:
            path = Path(filename).resolve()
            if not path.is_relative_to(root / "cotracker"):
                raise RuntimeError("a different cotracker checkout is already imported")
            if str(path.relative_to(root)) not in expected_source:
                raise RuntimeError("an unbound cotracker source module is already imported")
        else:
            paths = list(getattr(module, "__path__", []))
            if not paths or any(not Path(p).resolve().is_relative_to(root / "cotracker") for p in paths):
                raise RuntimeError("cotracker namespace escapes configured checkout")


class CoTracker3OfflineModel:
    def __init__(self, source_root: Path, checkpoint: Path, device: str, *,
                 expected_source: Mapping[str, str], expected_sha256: str, expected_size_bytes: int):
        # Validate bytes before importing any upstream model or allocating it.
        self.identity = verify_local_assets(source_root, checkpoint, expected_source=expected_source,
                                            expected_sha256=expected_sha256, expected_size_bytes=expected_size_bytes)
        root = Path(self.identity["source_root"])
        _check_imported_namespace(root, expected_source)
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("cotracker.predictor")
            if Path(module.__file__).resolve() != root / "cotracker/predictor.py":
                raise RuntimeError("predictor import escaped configured source file")
            _check_imported_namespace(root, expected_source)
        finally:
            sys.path.remove(str(root))

        import torch

        self.device = torch.device(device)
        # Do not invoke the upstream pickled-checkpoint loader. The constructor
        # is local and uninitialized; strict safe state loading must succeed.
        state = torch.load(self.identity["checkpoint"], map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "model" in state:
            state = state["model"]
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint must contain a tensor state dictionary")
        predictor = module.CoTrackerPredictor(checkpoint=None, **self.identity["predictor_args"])
        predictor.model.load_state_dict(state, strict=True)
        if file_sha256(Path(self.identity["checkpoint"])) != expected_sha256 or source_identity(root) != dict(expected_source):
            raise ValueError("tracker assets changed during initialization")
        self.model = predictor.eval().to(self.device)
        self.identity.update(query_frame_forced_by_upstream=True, independent_reverse_endpoint_check=False,
                             device=str(self.device), torch_version=torch.__version__)

    def track_queries(self, frames: np.ndarray, queries_txy: np.ndarray) -> dict[str, np.ndarray]:
        """Reuse the native-time query contract, not the v2 network or weights.

        Arbitrary observed query times are kept; backward_tracking fills the
        pre-query segment. It is not an independent reverse endpoint check.
        Predicted out-of-frame coordinates remain unmodified.
        """
        result = CoTracker2Model.track_queries(self, frames, queries_txy)
        if result["visible"].dtype != np.bool_:
            raise ValueError("upstream predictor must return boolean model visibility")
        return result
