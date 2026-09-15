#!/usr/bin/env python3
"""Time one four-generator VBench video group on four physical GPUs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from PIL import Image, ImageSequence
except Exception:  # pragma: no cover - only needed on the model host
    Image = None
    ImageSequence = None

try:
    import decord
except Exception:  # pragma: no cover - only needed on the model host
    decord = None

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("cogvideo", "lavie", "modelscope", "videocraft")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def media_duration(path: Path) -> float | None:
    if path.suffix.lower() == ".gif" and Image is not None:
        with Image.open(path) as image:
            total_ms = 0.0
            for frame in ImageSequence.Iterator(image):
                total_ms += float(frame.info.get("duration", 0.0))
            return total_ms / 1000.0 if total_ms else None
    if decord is not None:
        try:
            video = decord.VideoReader(str(path))
            fps = float(video.get_avg_fps())
            return len(video) / fps if fps > 0 else None
        except Exception:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/e0_scoring_manifest.csv")
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()

    selected = [
        row for row in rows(args.manifest)
        if row["dimension"] == "subject_consistency" and row["group_id"] == str(args.group_id)
    ]
    by_generator = {row["generator"]: row for row in selected}
    missing = [name for name in GENERATORS if name not in by_generator]
    if missing:
        raise SystemExit(f"group {args.group_id} missing generators: {missing}")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpus) != 4:
        raise SystemExit("exactly four GPUs are required for a group timing run")

    args.output_root.mkdir(parents=True, exist_ok=True)
    children: list[tuple[str, dict[str, str], subprocess.Popen[str], float]] = []
    common = [
        sys.executable,
        str(ROOT / "scripts/run_official_dataset_compare_dimension.py"),
        "--dimension", "subject_consistency",
        "--smoke",
        "--data-root", str(args.data_root),
        "--official-root", str(ROOT / "results/e0/raw_official_scores"),
        "--upstream", str(args.upstream),
    ]
    for generator, gpu in zip(GENERATORS, gpus):
        row = by_generator[generator]
        worker_root = args.output_root / generator
        worker_root.mkdir(parents=True, exist_ok=True)
        log = (worker_root / "console.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
        start = time.perf_counter()
        process = subprocess.Popen(common + ["--video-uid", row["video_uid"], "--output-root", str(worker_root)], env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        children.append((generator, row, process, start))

    records = []
    for generator, row, process, start in children:
        return_code = process.wait()
        elapsed = time.perf_counter() - start
        path = args.data_root / row["relative_video_path"]
        if not path.is_file():
            for suffix in (".mp4", ".gif"):
                candidate = path.with_suffix(suffix)
                if candidate.is_file():
                    path = candidate
                    break
        records.append({
            "generator": generator,
            "gpu": gpus[GENERATORS.index(generator)],
            "video_uid": row["video_uid"],
            "relative_video_path": row["relative_video_path"],
            "media_duration_s": media_duration(path),
            "elapsed_s": elapsed,
            "return_code": return_code,
        })

    total = max(item["elapsed_s"] for item in records)
    result = {
        "group_id": str(args.group_id),
        "dimension": "subject_consistency",
        "shared_video_dimensions": ["dynamic_degree", "dynamics_degree", "motion_smoothness"],
        "parallel_gpus": gpus,
        "videos": records,
        "video_count": len(records),
        "wall_clock_s": total,
        "all_success": all(item["return_code"] == 0 for item in records),
        "command_note": "end-to-end worker timing includes per-worker model load and one-video inference",
    }
    output = args.output_root / "timing.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
