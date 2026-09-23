"""Side-by-side DISPLAY copies. These are never scoring inputs."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from .static_jitter import decode, digest
from .score_static_jitter import select_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, help="local copy of complete construction directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701])
    parser.add_argument("--stress-test", action="store_true", help="display verified quality-rejected local warps with explicit labels")
    args = parser.parse_args(argv)
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        raise FileNotFoundError("existing ffmpeg required")
    build, output = Path(args.build), Path(args.output)
    rows = [json.loads(line) for line in (build / "candidates.jsonl").read_text().splitlines()]
    eligible_ids = {r["candidate_id"] for r in select_candidates(rows, stress_test=args.stress_test)}
    sources = [r for r in rows if r["family"] == "original"]
    output.mkdir(parents=True, exist_ok=False)
    records, contact = [], []
    for index, source in enumerate(sources, 1):
        members = [r for r in rows if r["base_id"] == source["base_id"]]
        control = next(r for r in members if r["family"] == "encoding_control")
        if control["decoded_pixels_sha256"] != source["source_decoded_pixels_sha256"]:
            raise ValueError("preview base must be pixel-identical to decoded official source")
        selected = [control] + [next(r for r in members if r["amplitude"] == a and r["seed"] == seed)
                                for a in args.amplitudes for seed in args.seeds]
        sequences = []
        for row in selected:
            path = build / "videos" / Path(row["video"]).name
            if digest(path) != row["sha256"] or row["candidate_id"] not in eligible_ids:
                raise ValueError("preview input changed or not eligible for declared display population")
            frames, pts, fps = decode(path)
            if frames.shape != tuple(source["source_shape"]) or not np.allclose(pts, source["source_pts"]):
                raise ValueError("preview inputs do not share native geometry/timing")
            sequences.append(frames)
        count, height, width, _ = sequences[0].shape
        panel_count = len(selected)
        composed = np.full((count, height + 60, panel_count * width, 3), 250, dtype=np.uint8)
        labels = ["Original"] + [f"{r['amplitude']:g} px / seed {r['seed']}" + (" *" if r["status"] == "rejected" else "")
                                 for r in selected[1:]]
        for column, (frames, label) in enumerate(zip(sequences, labels)):
            composed[:, 60:, column * width:(column + 1) * width] = frames
            for frame in composed:
                cv2.putText(frame, label, (column * width + 8, 48), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        title = f"{index:02d} / {source['generator']} / {source['prompt_id']}"
        if any(r["status"] == "rejected" for r in selected):
            title = f"{index:02d} / {source['generator']} / * = quality gate rejected; STRESS ONLY"
        for frame in composed:
            cv2.putText(frame, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
        path = output / f"{index:02d}_{source['generator']}_local_texture.mp4"
        # Repeat four times at native FPS for viewing; not a new scientific clip.
        display = np.tile(composed, (4, 1, 1, 1))
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{panel_count * width}x{height + 60}", "-r", str(Fraction(float(fps)).limit_denominator(100000)),
                        "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "17", "-preset", "fast",
                        "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                       input=display.tobytes(), check=True)
        still = composed[count // 2]
        cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
        contact.append(cv2.resize(still, (1152, round(still.shape[0] * 1152 / still.shape[1]))))
        records.append({"base_id": source["base_id"], "prompt": source["prompt_id"], "preview": path.name,
                        "panel_candidates": [r["candidate_id"] for r in selected], "seeds": args.seeds,
                        "amplitudes": args.amplitudes, "construction_status": [r["status"] for r in selected],
                        "stress_test": args.stress_test,
                        "fps": fps, "source_frames": count, "display_repetitions": 4,
                        "preview_sha256": digest(path), "scoring_input": False})
    cv2.imwrite(str(output / "contact_sheet.png"), cv2.cvtColor(np.concatenate(contact), cv2.COLOR_RGB2BGR))
    manifest = {"builder_sha256": digest(Path(__file__)), "records": records,
                "note": "Display only: original panel is the verified pixel-identical zero-edit control. "
                        "Scores use untouched original MP4s; previews are annotated, lossy and repeated, never scored."}
    (output / "preview_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"previews": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()
