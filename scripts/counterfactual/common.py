"""Shared video IO, provenance and determinism helpers for VBench-CF.

Every derived clip in the counterfactual dataset is produced by decisively
deterministic code: no random sampling, no model-generated frames, no
metric-dependent selection.  This module provides the primitives that the
per-dimension transforms build on.

Encoding is pinned (libx264, fixed preset/CRF, single thread, bitexact) so that
re-running the same transform on the same input yields byte-identical output.
`validate.py` re-checks that property on the generated dataset.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Conventional frame delay for a GIF that stores none (see `_probe_gif`).
GIF_DEFAULT_DELAY_MS = 100

# Pinned encoder settings.  `-threads 1` avoids slice-threading nondeterminism
# and `+bitexact` strips encoder/version strings from the bitstream.  The preset
# is deliberately `medium`: determinism comes from the pinned thread count and
# flags, not from the preset, and `veryslow` measured 3.5-4.5x slower for no
# benefit on these 16-frame clips.
X264_PRESET = "medium"
X264_CRF = "18"
ENCODE_FLAGS = (
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    X264_PRESET,
    "-crf",
    X264_CRF,
    "-pix_fmt",
    "yuv420p",
    "-threads",
    "1",
    "-fflags",
    "+bitexact",
    "-flags:v",
    "+bitexact",
    "-map_metadata",
    "-1",
    "-movflags",
    "+faststart",
)


class CounterfactualError(RuntimeError):
    """Raised when an input cannot be transformed without violating the contract."""


@dataclass(frozen=True)
class VideoMeta:
    """Probed properties of a single video file."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    container: str
    # Where `fps` came from: a real container rate, or the documented GIF
    # fallback for files that carry no frame-delay information at all.
    fps_source: str = "container"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), check=False, capture_output=True, **kwargs)


def probe_video(path: Path | str) -> VideoMeta:
    """Probe a video without decoding it.

    MP4 containers carry a reliable frame rate; GIF carries a per-frame delay
    instead, so its rate is derived from the summed delays.
    """
    path = Path(path)
    if not path.is_file():
        raise CounterfactualError(f"video not found: {path}")
    if path.suffix.lower() == ".gif":
        return _probe_gif(path)
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    if result.returncode != 0:
        raise CounterfactualError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise CounterfactualError(f"no video stream in {path}")
    stream = streams[0]
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if not fps:
        raise CounterfactualError(f"cannot determine frame rate for {path}")
    frame_count = int(stream.get("nb_frames") or 0)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    if frame_count <= 0 and duration > 0:
        frame_count = int(round(duration * fps))
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps
    return VideoMeta(
        path=str(path),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(fps),
        frame_count=int(frame_count),
        duration_s=float(duration),
        container=path.suffix.lower().lstrip("."),
    )


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "1/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_f = float(denominator)
            return float(numerator) / denominator_f if denominator_f else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _probe_gif(path: Path) -> VideoMeta:
    """Probe a GIF, falling back to the conventional delay when none is stored.

    The CogVideo GIFs in the VBench package carry no Graphic Control Extension
    at all, so they record no per-frame delay: 43 of 48 sampled files have zero
    delay blocks, and `ffprobe` reports a single 0.1 s frame.  The universal
    convention for a delay-less GIF is 100 ms per frame, which is also what
    `ffprobe` independently reports, so that is used -- and flagged in
    `fps_source` so the assumption travels with every derived row.
    """
    from PIL import Image

    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        delays: list[int] = []
        for index in range(frame_count):
            image.seek(index)
            delays.append(int(image.info.get("duration") or 0))
        width, height = image.size
    if frame_count <= 0:
        raise CounterfactualError(f"GIF has no frames: {path}")
    if all(delays):
        total_ms = sum(delays)
        source = "container"
    else:
        total_ms = frame_count * GIF_DEFAULT_DELAY_MS
        source = "gif_default_delay"
    duration = total_ms / 1000.0
    return VideoMeta(
        path=str(path),
        width=int(width),
        height=int(height),
        fps=frame_count / duration,
        frame_count=frame_count,
        duration_s=duration,
        container="gif",
        fps_source=source,
    )


def decode_video(path: Path | str) -> tuple[np.ndarray, VideoMeta]:
    """Decode every frame as uint8 RGB, preserving the exact frame count.

    `-vsync 0` is required: without it ffmpeg would duplicate or drop frames to
    satisfy a constant-rate output, which would silently corrupt every temporal
    transform in this dataset.
    """
    path = Path(path)
    meta = probe_video(path)
    if meta.container == "gif":
        frames = _decode_gif(path)
    else:
        frames = _decode_ffmpeg(path, meta)
    if len(frames) != meta.frame_count:
        raise CounterfactualError(
            f"decoded {len(frames)} frames but probed {meta.frame_count} for {path}"
        )
    return frames, meta


def _decode_gif(path: Path) -> np.ndarray:
    from PIL import Image, ImageSequence

    with Image.open(path) as image:
        frames = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]
    return np.stack(frames, axis=0)


def _decode_ffmpeg(path: Path, meta: VideoMeta) -> np.ndarray:
    result = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-vsync",
            "0",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    if result.returncode != 0:
        raise CounterfactualError(f"decode failed for {path}: {result.stderr.decode().strip()}")
    expected = meta.width * meta.height * 3 * meta.frame_count
    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    if buffer.size != expected:
        raise CounterfactualError(
            f"decoded {buffer.size} bytes, expected {expected} for {path}"
        )
    return buffer.reshape(meta.frame_count, meta.height, meta.width, 3)


def encode_video(frames: np.ndarray, out: Path | str, fps: float) -> VideoMeta:
    """Encode uint8 RGB frames to H.264 with pinned, deterministic settings."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise CounterfactualError(f"expected (N,H,W,3) RGB frames, got {frames.shape}")
    if len(frames) == 0:
        raise CounterfactualError("refusing to encode zero frames")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count, height, width, _ = frames.shape
    rate = _rate_string(fps)
    result = _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            rate,
            "-i",
            "-",
            *ENCODE_FLAGS,
            str(out),
        ],
        input=frames.tobytes(),
    )
    if result.returncode != 0:
        raise CounterfactualError(f"encode failed for {out}: {result.stderr.decode().strip()}")
    meta = probe_video(out)
    if meta.frame_count != count:
        raise CounterfactualError(
            f"{out} holds {meta.frame_count} frames but {count} were encoded"
        )
    return meta


def _rate_string(fps: float) -> str:
    """Render a frame rate ffmpeg round-trips exactly (8 -> '8', 7.5 -> '15/2')."""
    if float(fps).is_integer():
        return str(int(fps))
    from fractions import Fraction

    fraction = Fraction(fps).limit_denominator(1001)
    return f"{fraction.numerator}/{fraction.denominator}"


def resample_indices(frame_count: int, native_fps: float, target_fps: float) -> list[int]:
    """Nearest-source-frame indices sampling the same trajectory at `target_fps`.

    Downsampling only: the returned indices are strictly increasing and drawn
    from the original frames, so the timeline is an exact subset of the source
    trajectory.  Upsampling is deliberately rejected because it would require
    synthesised frames.
    """
    if target_fps > native_fps + 1e-9:
        raise CounterfactualError(
            f"refusing to upsample {native_fps} fps to {target_fps} fps: "
            "that would require synthesised frames"
        )
    duration = frame_count / native_fps
    indices: list[int] = []
    position = 0.0
    step = native_fps / target_fps
    while True:
        index = int(round(position))
        if index >= frame_count:
            break
        if not indices or index > indices[-1]:
            indices.append(index)
        position += step
    if not indices:
        raise CounterfactualError("resampling produced no frames")
    if duration <= 0:
        raise CounterfactualError("non-positive duration")
    return indices


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def stable_sample(items: Sequence[Any], count: int, seed: int) -> list[Any]:
    """Deterministically choose `count` items from `items`.

    Selection depends only on a hash of the item and a fixed seed, so it is
    reproducible without carrying RNG state and is independent of any metric
    score (plan section 3.3).
    """
    if count >= len(items):
        return list(items)
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]
