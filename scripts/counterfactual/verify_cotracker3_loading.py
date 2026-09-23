"""Real local checkpoint/predictor parity smoke; no metric score or download."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_feature_tracks import make_tracker, prepare_queries
from .probe_native_region_motion import resolve_media
from .static_jitter import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "tracker-root", "tracker-weight", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--video-root", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fresh parity output required")
    request = json.loads(Path(args.request).read_text())
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 40 * 1024**3:
        raise RuntimeError("40 GiB free required")
    torch.cuda.set_per_process_memory_fraction(.20)
    start = time.monotonic()
    adapter = make_tracker(request, args.tracker_root, args.tracker_weight, "cuda:0")
    from cotracker.predictor import CoTrackerPredictor
    reference = CoTrackerPredictor(checkpoint=args.tracker_weight, v2=False, offline=True, window_len=60).eval().to("cuda:0")
    a, b = adapter.model.state_dict(), reference.state_dict()
    state_equal = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    if not state_equal:
        raise ValueError("safe-loader state differs from upstream checkpoint loader")
    row = next(r for r in request["videos"] if r["family"] == "original")
    frames, times, _ = decode_video(resolve_media(row, args.video_root), TrajectoryConfig(sample_fps=8, max_side=512))
    import hashlib
    if hashlib.sha256(frames.tobytes()).hexdigest() != row["frames_sha256"] or not np.array_equal(times, row["timestamps"]):
        raise ValueError("native input changed")
    queries, _, _ = prepare_queries(frames[0], 0, "grid", 12)
    actual = adapter.track_queries(frames, queries)
    with torch.inference_mode():
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None].to("cuda:0", torch.float32)
        q = torch.tensor(queries, device="cuda:0")[None]
        tracks, visible = reference(video, queries=q, backward_tracking=True)
    expected = tracks[0].cpu().numpy()
    visibility_equal = np.array_equal(actual["visible"], visible[0].cpu().numpy())
    max_error = float(np.max(np.abs(actual["tracks"] - expected)))
    output.mkdir(parents=True)
    np.savez_compressed(output / "parity.npz", actual=actual["tracks"], expected=expected,
                        actual_visible=actual["visible"], expected_visible=visible[0].cpu().numpy(), queries=queries)
    result = {"status": "passed" if visibility_equal and max_error <= 1e-6 else "failed",
              "score": None, "scope": "all state tensors and one official original, 16 native frames, 12x12 fixed grid; not physical tracking truth or full model benchmark",
              "candidate_id": row["candidate_id"], "input_sha256": row["sha256"], "state_tensors": len(a),
              "state_equal": state_equal, "visibility_equal": visibility_equal, "track_max_abs_error_pixels": max_error,
              "assets": adapter.identity, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_seconds": time.monotonic()-start,
              "script_sha256": digest(Path(__file__)), "request_sha256": digest(Path(args.request)),
              "evidence_sha256": digest(output / "parity.npz"), "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ("status", "state_tensors", "track_max_abs_error_pixels", "visibility_equal", "elapsed_seconds")}))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
