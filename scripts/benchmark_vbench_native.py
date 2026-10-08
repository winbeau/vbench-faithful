#!/usr/bin/env python3
"""Time the pinned VBench compute APIs in one process with prepared metadata.

The computation loop matches VBench.evaluate. Metadata is supplied explicitly
because this timing cohort has reviewed queries for every dimension on the same
media. This does not use or modify VBench's filename-based dataset builder.
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import time
import sys

from paper_common import (ROOT, OFFICIAL, configure_imports, digest, load_assets,
                          load_inputs, verify_assets, write_json)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "assets", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    # Only the lightweight shared configuration module is imported before this.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = args.output.resolve()
    if output == ROOT or any(output.is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("Use a new output outside frozen research directories")
    output.mkdir(parents=True, exist_ok=False)
    assets = load_assets(args.assets)
    configure_imports(assets)
    rows = load_inputs(args.input)
    verify_assets(assets, OFFICIAL, "origin", workers=4)
    from vbench_audit_core.upstream import verify_upstream
    state = verify_upstream(assets["vbench"])
    sys.path.insert(0, assets["vbench"])
    import torch
    import cv2
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from vbench.utils import init_submodules
    full_info = [{"prompt_en": r["prompt"], "dimension": r["dimensions"],
                  "video_list": [r["video"]], "auxiliary_info": r["auxiliary_info"]} for r in rows]
    metadata = output / "full-info.json"
    write_json(metadata, full_info)
    modules = init_submodules(OFFICIAL, local=True, read_frame=False)
    report = {"input_sha256": digest(args.input), "upstream_sha": state.sha,
              "gpu": args.gpu, "torch": torch.__version__, "dimensions": {},
              "preflight_seconds": time.monotonic() - started,
              "execution": "pinned upstream compute APIs; one process; no result reuse"}
    for dimension in OFFICIAL:
        tick = time.monotonic()
        module = importlib.import_module("vbench." + dimension)
        aggregate, returned = getattr(module, "compute_" + dimension)(
            str(metadata), torch.device("cuda:0"), modules[dimension])
        report["dimensions"][dimension] = {"elapsed_seconds": time.monotonic() - tick,
            "input_count": sum(dimension in r["dimensions"] for r in rows),
            "returned_count": len(returned), "official_aggregate": float(aggregate)}
        write_json(output / (dimension + ".json"), returned)
        print(dimension, report["dimensions"][dimension], flush=True)
        report["wall_seconds"] = time.monotonic() - started
        write_json(output / "timing.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
