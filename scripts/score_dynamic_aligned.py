#!/usr/bin/env python3
"""Score a native 16-frame/8-FPS clip with the published, fixed aligned-v1 model."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="published dynamic_degree directory")
    parser.add_argument("--upstream", type=Path, required=True, help="clean pinned facebookresearch/vjepa2 checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "metrics/dynamic-degree/src"),
                   str(root / "packages/audit-models/src")]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import hashlib
    import cv2
    import numpy as np
    import torch
    from dynamic_degree.learned_probe import MotionProbe
    from vbench_audit_models.vjepa import FrozenVJEPA, sha256
    from scripts.counterfactual.official_video_jitter import native_video

    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selection = json.loads((args.model_dir / "selection.json").read_text())
    config = json.loads((args.model_dir / "configs/vjepa-probe-v1.json").read_text())
    head_path = args.model_dir / "aligned.pt"
    if (sha256(head_path) != selection["head"]["sha256"]
            or selection["head"]["sha256"] != "6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53"):
        raise ValueError("Published aligned-v1 head differs")
    frames, pts, fps = native_video(args.video, 1e-6)
    if len(frames) != 16 or abs(fps - 8) > 1e-6:
        raise ValueError("This frozen protocol requires all 16 native frames at 8 FPS; no resampling")
    encoder = FrozenVJEPA(args.upstream, args.model_dir / "backbone/vjepa2_1_vitb_dist_vitG_384.pt", config)
    head = MotionProbe().eval().requires_grad_(False)
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
    head = head.to("cuda:0")
    features = encoder.encode(frames)
    with torch.inference_mode():
        latent = float(head(torch.from_numpy(features).to("cuda:0", torch.float32)[None])[0])
    saved = io.BytesIO()
    np.save(saved, features, allow_pickle=False)
    result = {"version": "aligned-v1", "score": 1 / (1 + math.exp(-latent)), "latent": latent,
              "video_sha256": sha256(args.video), "head_sha256": sha256(head_path),
              "feature_sha256": hashlib.sha256(saved.getvalue()).hexdigest(),
              "decoded_pixels_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
              "frames": len(frames), "fps": fps, "encoder": encoder.identity,
              "torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("version", "score", "video_sha256", "feature_sha256")}))


if __name__ == "__main__":
    main()
