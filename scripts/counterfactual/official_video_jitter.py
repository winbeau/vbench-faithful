"""Controlled interventions on complete, native VBench 1.0 source videos.

No staticization, synthetic object motion, resizing, cropping, frame dropping, exposure
normalization, or FPS conversion. RGB-lossless output isolates intervention from
encoding changes. Original MP4s are scored directly. GIFs require a verified
pixel/timeline-identical MP4 adapter because the official infer rejects GIF paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from .static_jitter import ROOT, SourceRejected, decode, digest, perturb, source_path


def native_video(path: Path, tolerance: float):
    frames, pts, fps = decode(path)
    if len(frames) < 3 or not np.isfinite(fps) or fps <= 0:
        raise SourceRejected("source has insufficient frames or invalid native FPS")
    expected = np.arange(len(frames)) / fps
    if not np.allclose(pts, expected, atol=tolerance, rtol=0):
        raise SourceRejected("non-CFR source requires explicit timestamp-preserving adapter; not resampled")
    if path.suffix.lower() == ".gif":
        from PIL import Image, ImageSequence
        with Image.open(path) as image:
            durations = [frame.info.get("duration", 0) / 1000 for frame in ImageSequence.Iterator(image)]
        if not all(durations):
            raise SourceRejected("GIF has missing frame-duration metadata; no cadence invented")
        if len(durations) != len(frames) or not np.allclose(durations, 1 / fps, atol=tolerance, rtol=0):
            raise SourceRejected("GIF delays differ from native decoded cadence; not resampled")
    return frames, pts, fps


def encode_lossless(path: Path, frames: np.ndarray, fps: float, ffmpeg: str):
    if path.exists():
        raise FileExistsError(path)
    n, height, width, _ = frames.shape
    rate = str(Fraction(float(fps)).limit_denominator(100000))
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", rate, "-i", "pipe:0",
               "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "fast", "-threads", "1",
               "-pix_fmt", "rgb24", "-frames:v", str(n), str(path)]
    try:
        subprocess.run(command, input=frames.tobytes(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"lossless encoder exit {exc.returncode}: {exc.stderr.decode(errors='replace')[-2000:]}") from exc


def validate_native(decoded, intended, original, pts, fps, source_pts, source_fps, family, info, config):
    reasons = []
    if decoded.shape != original.shape or decoded.shape != intended.shape:
        return {"qualified": False, "reason": "native_shape_changed"}
    tolerance = config["timestamp_tolerance_seconds"]
    timeline = (len(pts) == len(source_pts) and np.allclose(pts, source_pts, atol=tolerance, rtol=0)
                and abs(fps - source_fps) <= tolerance)
    pixel_exact = bool(np.array_equal(decoded, intended))
    if not timeline:
        reasons.append("native_timeline_changed")
    if not pixel_exact:
        reasons.append("lossless_pixel_verification_failed")
    difference = float(np.mean(abs(decoded.astype(np.float32) - original)))
    correlations = []
    for a, b in zip(original, decoded):
        if np.array_equal(a, b):
            correlations.append(1.0)
            continue
        a = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (0, 0), 3).astype(float).ravel()
        b = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), (0, 0), 3).astype(float).ravel()
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-6 else 0.0)
    if min(correlations) < config["min_structure_correlation"]:
        reasons.append("structure_damaged")
    if info["clipped_fraction"] > config["max_clipped_fraction"]:
        reasons.append("excessive_added_clipping")
    if info.get("geometry_qualified") is False:
        reasons.append("local_displacement_geometry_failed")
    if family not in {"original", "encoding_control"} and difference < config["min_decoded_difference"]:
        reasons.append("perturbation_not_retained")
    return {"qualified": not reasons, "reason": ";".join(reasons) or None,
            "decoded_shape": list(decoded.shape), "pts": pts, "fps": fps,
            "pixel_exact_to_intended": pixel_exact, "native_timeline_preserved": bool(timeline),
            "decoded_difference_from_original": difference, "min_structure_correlation": min(correlations),
            "decoded_pixels_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            "mean_temporal_pixel_difference": float(np.mean(abs(np.diff(decoded.astype(np.float32), axis=0))))}


def interventions(config):
    yield {"family": "original", "amplitude": 0, "seed": 0}
    yield {"family": "encoding_control", "amplitude": 0, "seed": 0}
    for family in config["families"]:
        for amplitude in config["amplitudes"]:
            for seed in config.get("seeds", config.get("noise_seeds", [])):
                yield {"family": family, "amplitude": amplitude, "seed": seed}


def build(args):
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise FileNotFoundError("existing ffmpeg with libx264rgb required; no construction started")
    config_path, sources_path = Path(args.config), Path(args.sources)
    config = json.loads(config_path.read_text())
    if config.get("protocol") not in {"official-video-appearance-only-v1", "official-video-local-texture-jitter-v1"}:
        raise ValueError("wrong protocol: this builder never accepts synthetic-static configurations")
    sources = [r for r in map(json.loads, sources_path.read_text().splitlines()) if r["split"] == args.split]
    sources = sources[args.source_start:args.source_start + args.limit if args.limit else None]
    if not sources:
        raise ValueError("empty source selection")
    output = Path(args.output).resolve()
    if output == ROOT or any(output.is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("cannot write to workspace root or frozen trees")
    output.mkdir(parents=True, exist_ok=False)
    (output / "videos").mkdir()
    cv2.setNumThreads(2)
    specs = list(interventions(config))
    provenance = {"config": config, "config_sha256": digest(config_path),
                  "sources_sha256": digest(sources_path), "source_count": len(sources),
                  "source_start": args.source_start, "limit": args.limit, "split": args.split,
                  "builder_sha256": digest(Path(__file__)), "appearance_helpers_sha256": digest(Path(__file__).with_name("static_jitter.py")),
                  "local_warp_helpers_sha256": digest(Path(__file__).with_name("local_texture_jitter.py"))
                      if config["protocol"] == "official-video-local-texture-jitter-v1" else None,
                  "ffmpeg_sha256": digest(Path(ffmpeg)),
                  "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
                  "opencv": cv2.__version__, "numpy": np.__version__,
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "construction.json").write_text(json.dumps(provenance, indent=2))
    counts = Counter()
    with (output / "candidates.jsonl").open("x") as ledger:
        for offset, source in enumerate(sources, args.source_start):
            meta, failure, source_rejected = {}, None, False
            start = offset * len(specs)
            try:
                original_path = source_path(Path(args.data_root), source)
                # Retain source identity even when decoding/timing qualification rejects it.
                meta = {"source_video": str(original_path), "source_sha256": digest(original_path)}
                original, source_pts, source_fps = native_video(original_path, config["timestamp_tolerance_seconds"])
                meta = {**meta,
                        "source_decoded_pixels_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                        "source_shape": list(original.shape), "source_fps": source_fps, "source_pts": source_pts,
                        "source_duration_seconds": len(original) / source_fps,
                        "frame_map": "identity_all_source_frames", "coordinate_map": "identity",
                        "source_format_adaptation": "none" if original_path.suffix.lower() == ".mp4" else "lossless_rgb_mp4"}
                if original_path.suffix.lower() not in {".mp4", ".gif"}:
                    raise SourceRejected("unsupported official source format; not converted implicitly")
                encoded_path = output / "videos" / f"video_{start + 1:06d}.mp4"
                encode_lossless(encoded_path, original, source_fps, ffmpeg)
                encoded, encoded_pts, encoded_fps = decode(encoded_path)
                control = validate_native(encoded, original, original, encoded_pts, encoded_fps, source_pts,
                                          source_fps, "encoding_control", {"clipped_fraction": 0}, config)
                if not control["qualified"]:
                    raise RuntimeError(f"zero-edit encoding control failed: {control['reason']}")
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                source_rejected = isinstance(exc, SourceRejected)
            source_counts = Counter()
            for ordinal, spec in enumerate(specs, start):
                row = {**source, **meta, **spec, "kind": "official_native", "protocol": config["protocol"],
                       "candidate_id": f"video_{ordinal:06d}", "video": str(output / "videos" / f"video_{ordinal:06d}.mp4")}
                try:
                    if failure:
                        raise ValueError(failure)
                    if spec["family"] == "original":
                        path = original_path if original_path.suffix.lower() == ".mp4" else encoded_path
                        row.update(video=str(path), **control)
                    elif spec["family"] == "encoding_control":
                        path = encoded_path
                        row.update(video=str(path), **control)
                    else:
                        if config["protocol"] == "official-video-local-texture-jitter-v1":
                            from .local_texture_jitter import local_texture_jitter
                            if spec["family"] != "local_texture_alternating":
                                raise ValueError("unknown local displacement family")
                            intended, info, field = local_texture_jitter(original, spec["amplitude"], spec["seed"], config)
                            evidence = output / "displacement_fields" / f"{row['candidate_id']}.npz"
                            evidence.parent.mkdir(exist_ok=True)
                            np.savez_compressed(evidence, **field)
                            info.update(displacement_evidence=str(evidence), displacement_evidence_sha256=digest(evidence))
                        else:
                            intended, info = perturb(original, spec["family"], spec["amplitude"], spec["seed"])
                        row.update(info)
                        path = Path(row["video"])
                        encode_lossless(path, intended, source_fps, ffmpeg)
                        actual, pts, fps = decode(path)
                        row.update(validate_native(actual, intended, original, pts, fps, source_pts, source_fps,
                                                   spec["family"], info, config))
                    row.update(sha256=digest(path), status="qualified" if row["qualified"] else "rejected")
                except Exception as exc:
                    row.update(qualified=False, status="rejected" if source_rejected else "construction_failed",
                               reason=f"{type(exc).__name__}: {exc}")
                source_counts[row["status"]] += 1
                counts[row["status"]] += 1
                ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
                ledger.flush()
            print(json.dumps({"base_id": source["base_id"], "counts": dict(source_counts), "source_error": failure}), flush=True)
    completion = {"status": "finished", "counts": dict(counts), "manifest_sha256": digest(output / "candidates.jsonl"),
                  "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "completion.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion))
    if counts["construction_failed"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.source_start < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("source-start must be nonnegative and limit positive")
    build(args)


if __name__ == "__main__":
    main()
