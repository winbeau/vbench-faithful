#!/usr/bin/env python3
"""Materialize the reviewed 32-video timing queries without changing any media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from paper_common import ROOT, OFFICIAL, digest, write_json


def prepare(annotations, video_root):
    videos = annotations["videos"]
    if len(videos) != 32 or len({v["video_sha256"] for v in videos}) != 32:
        raise ValueError("This timing cohort requires 32 distinct media hashes")
    rows = []
    for video in videos:
        path = (Path(video_root) / video["video"]).resolve()
        if digest(path) != video["video_sha256"]:
            raise ValueError(f"Media changed: {path}")
        info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)
        ], text=True))["streams"][0]
        if info != video["media"]:
            raise ValueError(f"Native media protocol changed: {path}")
        for dimension in OFFICIAL:
            rows.append({"id": f"speed32:{video['index']:02d}:{dimension}",
                         "video": str(path), "video_sha256": video["video_sha256"],
                         "prompt": video["queries"].get(dimension, video["source_prompt"]),
                         "dimensions": [dimension],
                         "auxiliary_info": {dimension: video["auxiliary_info"][dimension]}
                         if dimension in video["auxiliary_info"] else {},
                         "subject_en": video["subject_en"]})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate annotation index")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=ROOT / "configs/benchmarks/same32-20261008.json")
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(ROOT / p) for p in ("data", "results", "splits", "runs")):
        raise ValueError("Use a new manifest outside frozen research directories")
    rows = prepare(json.loads(args.annotations.read_text()), args.video_root)
    write_json(output, rows)
    print(f"32 videos, 16 dimensions, {len(rows)} input records: {output}")


if __name__ == "__main__":
    main()
