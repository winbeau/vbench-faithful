#!/usr/bin/env python3
"""Objects occlusion ladder (Table C, sensitivity half): occlude one target and re-detect.

For each video the target object is located in the first frame with GRiT; that region is
then covered with a neutral patch over an increasing area on **every** frame and GRiT is
re-run. Frames are used directly, so no video has to be re-encoded.

Per level the official multiple-objects rule (a frame counts only if every required
object is detected) is evaluated, together with two controls:

* equal-area patch placed on background (not on the object) -- separates occlusion from
  "a patch exists in the frame";
* every other object's detection is compared with the unoccluded frame -- a patch that
  removes unrelated detections is a detector artefact, not evidence about the target.

Usage (on the scoring host)::

    python scripts/cache_occlusion_outputs.py --manifest <...>/multiple-objects.csv \
        --video-root <...> --out data/backend-cache/occlusion.jsonl --limit 30 --device cuda:2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

LEVELS = (0.0, 0.5, 0.8, 1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vbench-root", default="/root/wenbiao_zhao/VBench")
    parser.add_argument("--grit-weights", default="/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth")
    parser.add_argument("--num-frames", type=int, default=8, help="fewer frames keeps the ladder affordable")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--targets", default=None, help="batch_predict objects output; falls back to 'a X and a Y' parsing")
    return parser.parse_args(argv)


def detect(model, frames, device):
    import torch

    out = []
    with torch.no_grad():
        for frame in frames:
            ret = model.run_caption_tensor(frame)
            current = []
            if len(ret[0]) > 0:
                for info in ret[0]:
                    current.append({"label": info[0], "box": [float(v) for v in info[1][:4]]})
            out.append(current)
    return out


def occlude(frame, box, level):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half_w = max(1, int((x1 - x0) * level / 2))
    half_h = max(1, int((y1 - y0) * level / 2))
    patched = frame.copy()
    patched[max(0, cy - half_h) : cy + half_h, max(0, cx - half_w) : cx + half_w] = 128
    return patched


def background_patch(frame, box, level, size):
    """Equal-area patch far from the object, to control for 'a patch exists'."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    width, height = (x1 - x0), (y1 - y0)
    half_w = max(1, int(width * level / 2))
    half_h = max(1, int(height * level / 2))
    cx = 10 if x0 > size[1] // 2 else size[1] - 10
    cy = 10 if y0 > size[0] // 2 else size[0] - 10
    patched = frame.copy()
    patched[max(0, cy - half_h) : cy + half_h, max(0, cx - half_w) : cx + half_w] = 128
    return patched


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.vbench_root).resolve()))
    import torch  # noqa: E402
    import torchvision.transforms as tv_transforms  # noqa: E402
    from vbench.third_party.grit_model import DenseCaptioning  # noqa: E402
    from vbench.utils import load_video  # noqa: E402

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DenseCaptioning(device)
    model.initialize_model_det(model_weight=args.grit_weights)

    targets: dict[str, list[str]] = {}
    if args.targets and Path(args.targets).exists():
        for line in Path(args.targets).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                value = row.get("target") or {}
                entities = value.get("entities") if isinstance(value, dict) else None
                if entities:
                    targets[row["prompt"]] = [str(name).lower() for name in entities]

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt = (row.get("prompt_en") or "").strip()
            required = targets.get(prompt)
            if not required:
                continue
            for key in ("video_a_path", "video_b_path"):
                relative = row.get(key)
                if relative and relative not in seen:
                    seen.add(relative)
                    rows.append({"prompt": prompt, "relative_path": relative})
    rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            video_path = Path(args.video_root) / row["relative_path"]
            required = targets[row["prompt"]]
            entry: dict[str, object] = {"prompt": row["prompt"], "video_path": str(video_path), "required": required}
            if not video_path.exists():
                entry["status"] = "missing_video"
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue
            try:
                video_tensor = load_video(str(video_path), num_frames=args.num_frames)
                _, _, h, w = video_tensor.size()
                if min(h, w) > 768:
                    scale = 720.0 / min(h, w)
                    video_tensor = tv_transforms.Resize(size=(int(scale * h), int(scale * w)))(video_tensor)
                frames = video_tensor.permute(0, 2, 3, 1).numpy()
                size = [int(frames.shape[1]), int(frames.shape[2])]
                base_detections = detect(model, frames, device)
                target_label = required[0]
                base_boxes = [item["box"] for item in base_detections[0] if str(item["label"]).lower() == target_label]
                if not base_boxes:
                    entry["status"] = "target_not_detected_in_first_frame"
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue
                box = base_boxes[0]
                levels: dict[str, object] = {}
                for level in LEVELS:
                    occluded = [occlude(frame, box, level) for frame in frames]
                    detections = detect(model, occluded, device)
                    object_pass = sum(
                        1 for frame in detections if all(any(str(item["label"]).lower() == name for item in frame) for name in required)
                    )
                    background = detect(model, [background_patch(frame, box, level, size) for frame in frames], device)
                    background_pass = sum(
                        1 for frame in background if all(any(str(item["label"]).lower() == name for item in frame) for name in required)
                    )
                    other_before = sum(len(frame) for frame in base_detections)
                    other_after = sum(len(frame) for frame in detections)
                    levels[str(level)] = {
                        "object_pass_frames": object_pass,
                        "background_pass_frames": background_pass,
                        "frames": len(detections),
                        "detections_before": other_before,
                        "detections_after": other_after,
                    }
                entry.update({"status": "ok", "frame_size": size, "target_box": box, "levels": levels})
                written += 1
            except Exception as error:  # noqa: BLE001
                entry.update({"status": f"error:{type(error).__name__}", "error": str(error)[:200]})
                skipped += 1
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if written % 5 == 0:
                print(json.dumps({"written": written, "skipped": skipped}), flush=True)
    print(json.dumps({"status": "done", "written": written, "skipped": skipped, "out": str(out_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
