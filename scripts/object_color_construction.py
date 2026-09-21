"""Construction-only masks and lossless visibility interventions.

These arrays never enter a metric/model scoring interface. The RGB videos are
the sole visual inputs to Official and Repair.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import cv2
import numpy as np

from vbench_audit_core.inputs import sha256_file


def occlude(frames, masks, visible_fraction, *, seed):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("expected byte RGB [T,H,W,3]")
    if masks.shape != frames.shape[:3] or not np.isin(masks, (0, 1)).all():
        raise ValueError("expected native binary construction masks")
    if visible_fraction not in {0, .25, .5, .75, 1} or len(frames) != 16:
        raise ValueError("frozen visibility ladder has 16 frames and five levels")
    order = np.random.default_rng(seed).permutation(len(frames))
    hidden = sorted(order[int(len(frames)*visible_fraction):].tolist())
    result = frames.copy()
    for index in hidden:
        mask = masks[index].astype(np.uint8)
        if not mask.any() or not (mask == 0).any():
            raise ValueError("cannot occlude an empty/full-image target mask")
        distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        alpha = np.minimum(distance / 1.5, 1.)[..., None] * mask[..., None]
        fill = np.median(frames[index][mask == 0], axis=0)
        result[index] = np.rint(frames[index].astype(np.float64)*(1-alpha) + fill*alpha).clip(0, 255).astype(np.uint8)
    if not np.array_equal(result[masks == 0], frames[masks == 0]):
        raise AssertionError("changed a pixel outside construction mask")
    return result, {"hidden_indices": hidden, "visible_fraction": visible_fraction,
                    "outside_mask_changed_pixels": 0,
                    "rgb_sha256": hashlib.sha256(result.tobytes()).hexdigest()}


def encode_lossless(frames, path, *, fps=8):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, channels = frames.shape
    if frames.dtype != np.uint8 or channels != 3 or not n:
        raise ValueError("lossless encoder needs nonempty native RGB bytes")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264rgb",
               "-crf", "0", "-preset", "medium", "-threads", "1", "-map_metadata", "-1",
               "-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(path)]
    subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), check=True, capture_output=True)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "frames": n, "fps": fps,
            "media_seconds": n/fps, "codec": "libx264rgb crf0 rgb24 single-thread"}


# Construction model adapters share the same package boundary as scoring models.
from vbench_audit_models.construction import ConstructionMasks
