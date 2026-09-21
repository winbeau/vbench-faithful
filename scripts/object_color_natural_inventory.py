"""Read-only per-video duration, byte identity and runtime inventory."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import math

from scripts.object_color_natural import cohort
from scripts.object_color_natural_media import save
from vbench_audit_core.inputs import sha256_file
from vbench_audit_core.upstream import verify_upstream


def media_duration(probe, path):
    candidates = [probe.get("format", {}).get("duration"), probe["streams"][0].get("duration")]
    for value in candidates:
        if value not in (None, "N/A") and math.isfinite(float(value)) and float(value) > 0:
            return float(value), "ffprobe"
    if path.suffix.casefold() == ".gif":
        from PIL import Image
        with Image.open(path) as image:
            durations = []
            for i in range(image.n_frames):
                image.seek(i)
                durations.append(image.info.get("duration"))
        if all(isinstance(d, (int, float)) and d >= 0 for d in durations) and sum(durations) > 0:
            return sum(durations)/1000., "gif_embedded_frame_delays"
        return None, "GIF_has_no_positive_embedded_frame_delays; do_not_invent_fps"
    return None, "duration_not_available"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("/root/wenbiao_zhao/VBench"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    records = [r for d in ("object_class", "color") for r in cohort(root, d)]
    ledger = {r["relative_path"]: r for r in json.loads((root/"media-ledger.json").read_text())["files"]}

    def inspect(row):
        path = root/"media"/row["relative_video_path"]
        digest = sha256_file(path)
        if digest != ledger[row["relative_video_path"]]["sha256"]:
            raise ValueError("official media byte identity changed")
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration",
            "-of", "json", str(path)], text=True))
        seconds, duration_source = media_duration(probe, path)
        return {"video_uid": row["video_uid"], "dimension": row["dimension"],
            "generator": row["generator"], "relative_video_path": row["relative_video_path"],
            "sha256": digest, "bytes": path.stat().st_size, "stream": probe["streams"][0],
            "media_seconds": seconds, "duration_source": duration_source}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        media = list(pool.map(inspect, records))
    versions = {}
    for name in ("torch", "torchvision", "transformers", "detectron2", "decord", "numpy", "scipy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"media": media, "unique_media_files": len(media),
        "known_media_seconds": sum(r["media_seconds"] for r in media if r["media_seconds"] is not None),
        "unknown_duration_videos": sum(r["media_seconds"] is None for r in media),
        "by_dimension_known_seconds": {d: sum(r["media_seconds"] for r in media if r["dimension"] == d and r["media_seconds"] is not None)
                                 for d in ("object_class", "color")},
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "versions": versions, "upstream": asdict(verify_upstream(args.upstream)),
        "gpu_inventory_csv": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"], text=True),
        "scope": "read-only media/runtime inventory; actual scoring device masks and wall times are in per-backend run.json"}
    save(root/"inventory.json", report)
    print(json.dumps({"verified_media": len(media), "known_media_seconds": report["known_media_seconds"],
                      "unknown_duration_videos": report["unknown_duration_videos"]}), flush=True)


if __name__ == "__main__":
    main()
