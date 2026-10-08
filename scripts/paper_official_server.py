#!/usr/bin/env python3
"""Original pinned VBench compute APIs in one process per selected GPU."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

from paper_common import OFFICIAL, configure_imports, digest, load_assets, write_json
from vbench_audit_core.worker_rpc import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    assets = load_assets(args.assets)
    configure_imports(assets)
    from vbench_audit_core.upstream import verify_upstream
    verify_upstream(assets["vbench"])
    sys.path.insert(0, assets["vbench"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import cv2
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Original VBench worker requires exactly one visible CUDA GPU")
    from paper_visual import official

    def handle(request):
        if set(request) != {"dimension", "input", "output"} or request["dimension"] not in OFFICIAL:
            raise ValueError("Invalid original VBench request")
        started = time.monotonic()
        input_path, output = Path(request["input"]), Path(request["output"])
        rows = json.loads(input_path.read_text())
        for row in rows:
            if digest(row["video"]) != row["video_sha256"]:
                raise ValueError("Video changed after input validation")
        print(f"Starting original {request['dimension']} ({len(rows)} inputs)", flush=True)
        try:
            payload = official(rows, request["dimension"], assets, output, upstream_only=True)
            payload.update(dimension=request["dimension"], mode="origin", torch=torch.__version__,
                worker_sha256=digest(__file__), input_manifest_sha256=digest(input_path),
                execution_model="original compute APIs in a persistent process", worker_pid=os.getpid())
            write_json(output, payload)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        print(f"Finished original {request['dimension']} in {elapsed:.3f}s", flush=True)
        return {"pid": os.getpid(), "compute_seconds": elapsed}

    serve(args.socket, handle)


if __name__ == "__main__":
    main()
