#!/usr/bin/env python3
"""Construct the website's LTX interventions without touching research inputs.

Requires FFmpeg. Scoring masters preserve RGB pixels; browser encodes are
separate presentation assets and must never be used to recover the scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def pixel_hashes(path: Path, filters: str) -> list[str]:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", filters,
         "-f", "framemd5", "-"], check=True, capture_output=True, text=True,
    )
    return [line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines()
            if line and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.output / "coast-base.mp4"
    blur = args.output / "coast-background-blur.mp4"
    mirror = args.output / "coast-mirror.mp4"
    common = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(args.source)]
    encoding = ["-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "fast",
                "-threads", "4", "-pix_fmt", "rgb24", "-movflags", "+faststart"]
    run(*common, "-vf", "format=rgb24", *encoding, str(base))
    run(*common, "-vf", "format=rgb24,hflip", *encoding, str(mirror))
    # A predeclared background band sits above the car throughout the clip.
    # It is a construction region, never an input to either scoring backend.
    filters = ("[0:v]format=rgb24,split[base][region];"
               "[region]crop=iw:450:0:0,gblur=sigma=35[blur];"
               "[base][blur]overlay=0:0:format=rgb:enable='lt(n,30)',format=rgb24")
    run(*common, "-filter_complex", filters, *encoding, str(blur))
    all_rgb = "format=rgb24"
    lower = "format=rgb24,crop=iw:ih-450:0:450"
    tail = "format=rgb24,select='gte(n,30)'"
    checks = {
        "base_equals_source_rgb": pixel_hashes(base, all_rgb) == pixel_hashes(args.source, all_rgb),
        "mirror_is_exact_horizontal_flip": pixel_hashes(mirror, all_rgb) == pixel_hashes(base, all_rgb + ",hflip"),
        "lower_638_rows_unchanged": pixel_hashes(base, lower) == pixel_hashes(blur, lower),
        "frames_30_onward_unchanged": pixel_hashes(base, tail) == pixel_hashes(blur, tail),
        "intervention_changes_pixels": pixel_hashes(base, all_rgb) != pixel_hashes(blur, all_rgb),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Construction pixel checks failed: {checks}")
    records = []
    for path in (args.source, base, blur, mirror):
        records.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size})
    receipt = {
        "schema": "vbench-site-construction/1", "generator": "LTX-2.3 22B Dev",
        "source_seed": 45, "width": 1920, "height": 1088, "fps": 24, "frames": 121,
        "selection": "One existing LTX clip selected before scoring; all evaluated pairs are retained.",
        "scene": "Identical video; only the declared query changes from an ocean to a sea.",
        "subject": "Gaussian blur sigma=35, rows 0:450, frames 0:30; subject annotation car.",
        "spatial": "Horizontal reflection of every RGB frame; fixed car-left-of-bicycle query.",
        "checks": checks, "files": records,
    }
    (args.output / "construction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
