#!/usr/bin/env python3
"""Caption one real video frame with the official VBench Tag2Text pipeline (B5).

Runs **on the scoring host**, where the pinned VBench checkout and the Tag2Text
weights live; it uses ``vbench.utils.load_video`` / ``tag2text_transform`` and
``tag2text_caption`` exactly as ``vbench.scene`` does, so the captions match the
production distribution instead of an approximation.

The prompt split is enforced here: only prompts whose frozen split matches
``--split-filter`` are captioned, so test prompts can never leak into training.

Usage::

    python scripts/caption_frames.py --manifest <scene.csv> --video-root <dir> \
        --frozen-split splits/e0_prompt_split.csv --split-filter dev_only \
        --output captions-dev.jsonl --limit 200 --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


def load_frozen_split(path: Path) -> dict[str, set[str]]:
    frozen: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frozen.setdefault(row["prompt_id"], set()).add(row["split"])
    return frozen


def split_of(frozen: dict[str, set[str]], prompt: str) -> str:
    splits = frozen.get(prompt)
    if not splits:
        return "outside"
    if splits == {"dev"}:
        return "dev_only"
    if splits == {"test"}:
        return "test_only"
    return "dev_and_test"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="human-preference scene.csv")
    parser.add_argument("--video-root", required=True, help="directory that video paths in the manifest are relative to")
    parser.add_argument("--frozen-split", required=True)
    parser.add_argument("--split-filter", default="dev_only", choices=["dev_only", "test_only", "dev_and_test", "outside", "all"])
    parser.add_argument("--vbench-root", default="/root/wenbiao_zhao/VBench")
    parser.add_argument("--weights", default="/root/.cache/vbench/caption_model/tag2text_swin_14m.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-index", default="middle", help="middle | first | last | integer")
    parser.add_argument("--frame-fractions", default=None, help="comma list of positions in [0,1); overrides --frame-index and emits one row per frame")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--caption-retries", type=int, default=3)
    parser.add_argument("--save-frames", default=None, help="directory to write the captioned frame as PNG (for vision labelling)")
    args = parser.parse_args(argv)

    frozen = load_frozen_split(Path(args.frozen_split))
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt = row.get("prompt_en", "")
            if args.split_filter != "all" and split_of(frozen, prompt) != args.split_filter:
                continue
            for key in ("video_a_path", "video_b_path"):
                relative = row.get(key)
                if not relative:
                    continue
                if (prompt, relative) in seen:
                    continue
                seen.add((prompt, relative))
                rows.append({"prompt": prompt, "relative_path": relative, "generator": relative.split("/")[1] if "/" in relative else "unknown", "instance_id": row.get("instance_id", "")})
    if args.limit is not None:
        rows = rows[: args.limit]
    print(json.dumps({"selected_videos": len(rows), "unique_prompts": len({row['prompt'] for row in rows}), "split_filter": args.split_filter}), flush=True)

    sys.path.insert(0, str(Path(args.vbench_root).resolve()))
    import torch  # noqa: E402
    from vbench.third_party.tag2Text.tag2text import tag2text_caption  # noqa: E402
    from vbench.utils import load_video, tag2text_transform  # noqa: E402

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = tag2text_caption(pretrained=args.weights, image_size=384, vit="swin_b").to(device).eval()
    transform = tag2text_transform(384)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(args.save_frames) if args.save_frames else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            video_path = Path(args.video_root) / row["relative_path"]
            if not video_path.exists():
                failed += 1
                handle.write(json.dumps({**row, "status": "missing_video", "video_path": str(video_path)}, ensure_ascii=False) + "\n")
                continue
            try:
                frames = load_video(str(video_path), num_frames=args.num_frames, return_tensor=False, width=384, height=384)
            except Exception as error:  # noqa: BLE001 - report, never crash the batch
                failed += 1
                handle.write(json.dumps({**row, "status": f"load_error:{type(error).__name__}", "video_path": str(video_path)}, ensure_ascii=False) + "\n")
                continue
            if args.frame_fractions:
                indices = [min(len(frames) - 1, max(0, int(float(frac) * len(frames)))) for frac in args.frame_fractions.split(",")]
            elif args.frame_index == "middle":
                indices = [len(frames) // 2]
            elif args.frame_index == "first":
                indices = [0]
            elif args.frame_index == "last":
                indices = [len(frames) - 1]
            else:
                indices = [min(int(args.frame_index), len(frames) - 1)]
            for position, index in enumerate(indices):
              frame = frames[index]
              caption = None
              for _ in range(max(1, args.caption_retries)):
                try:
                    tensor = transform(frame).to(device).unsqueeze(0)
                    with torch.no_grad():
                        captions, _tags = model.generate(tensor, tag_input=None, return_tag_predict=True)
                    caption = captions[0] if isinstance(captions, (list, tuple)) else str(captions)
                    break
                except Exception:  # noqa: BLE001
                    continue
              if caption is None:
                failed += 1
                handle.write(json.dumps({**row, "status": "caption_error", "video_path": str(video_path)}, ensure_ascii=False) + "\n")
                continue
              frame_path = None
              if frame_dir is not None:
                safe = f"{written:05d}_{abs(hash((row['prompt'], row['relative_path']))) % 10**10}.png"
                frame_path = frame_dir / safe
                try:
                    from PIL import Image  # noqa: PLC0415 - available on the scoring host

                    if hasattr(frame, "convert"):
                        image = frame.convert("RGB")
                    else:
                        import numpy as np  # noqa: PLC0415

                        array = np.asarray(frame)
                        image = Image.fromarray(array.astype("uint8")).convert("RGB")
                    image.save(frame_path)
                except Exception as error:  # noqa: BLE001 - frame saving is best effort but must be visible
                    print(json.dumps({"frame_save_error": f"{type(error).__name__}: {error}"}), flush=True)
                    frame_path = None
              written += 1
              handle.write(
                json.dumps(
                    {
                        **row,
                        "status": "ok",
                        "video_path": str(video_path),
                        "frame_index": index,
                        "frames_sampled": len(frames),
                        "caption": caption,
                        "frame_path": str(frame_path) if frame_path else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
              )
              if written % 25 == 0:
                print(json.dumps({"written": written, "failed": failed}), flush=True)
    print(json.dumps({"status": "done", "written": written, "failed": failed, "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
