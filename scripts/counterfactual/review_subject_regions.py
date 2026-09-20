"""Export construction rejection evidence and an independent prompt review UI."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .common import sha256_file
from .region_discrimination import bounding_box, intersection_area, mirror_box
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_jsonl


def construction_review(construction: Path, output: Path) -> None:
    sections, report = [], []
    for row in read_jsonl(construction / "index.jsonl"):
        source = json.loads(artifact_path(construction, row["manifest"]).read_text())
        base = source["base"]
        entry = {"base_id": base["base_id"], "prompt": base["prompt_en"], "localization_status": source["status"],
                 "rejection_reasons": source["rejection_reasons"], "geometry": []}
        if source["status"] == "accepted":
            path = artifact_path(construction, source["mask_file"]["path"])
            if sha256_file(path) != source["mask_file"]["sha256"]:
                raise ValueError("mask hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                masks, others = data["masks"], data["other_instances"]
            frames = read_png_sequence(construction, source["frames"])
            images = []
            for t, mask in enumerate(masks):
                box = bounding_box(mask)
                mirror = mirror_box(box, mask.shape[1])
                x0, y0, x1, y1 = mirror
                geometry = {"frame": t, "subject_box": box, "mirror_box": mirror,
                            "intersection_pixels": intersection_area(box, mirror),
                            "other_instance_pixels": int(others[t, y0:y1, x0:x1].sum())}
                entry["geometry"].append(geometry)
                if t not in {0, len(masks) // 2, len(masks) - 1}:
                    continue
                image = frames[t].copy()
                cv2.rectangle(image, box[:2], (box[2] - 1, box[3] - 1), (0, 220, 0), 2)
                cv2.rectangle(image, mirror[:2], (mirror[2] - 1, mirror[3] - 1), (255, 0, 0), 2)
                _, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                uri = base64.b64encode(encoded).decode()
                images.append(f'<figure><img src="data:image/png;base64,{uri}" width="320"><figcaption>frame {t}; '
                              f'box intersection {geometry["intersection_pixels"]}; other-object pixels {geometry["other_instance_pixels"]}'
                              '</figcaption></figure>')
            sections.append(f'<section><h2>{html.escape(base["prompt_en"])}</h2><p>{html.escape(base["base_id"])}</p>'
                            f'<div class="frames">{"".join(images)}</div></section>')
        report.append(entry)
    document = '<!doctype html><meta charset="utf-8"><title>Subject construction rejection evidence</title>' \
               '<style>body{font:16px system-ui;margin:2rem} .frames{display:flex;flex-wrap:wrap}figure{margin:8px}section{border-top:1px solid #ccc}</style>' \
               '<h1>Construction review</h1><p>Green: SegFormer/GrabCut subject box. Red: exact horizontal mirror. ' \
               'These are construction diagnostics, never scoring prompts. Other-object absence is conditional on SegFormer labels.</p>' + ''.join(sections)
    (output / "construction_review.html").write_text(document)
    write_jsonl(output / "rejections.jsonl", report)


def localizer_review(dataset: Path, output: Path) -> None:
    """Show only unannotated clean frames; no construction mask suggestions."""
    packets = []
    for entry in read_jsonl(dataset / "index.jsonl"):
        manifest = json.loads(artifact_path(dataset, entry["manifest"]).read_text())
        if manifest["status"] != "accepted":
            continue
        files = manifest["variants"]["clean"]
        frames = read_png_sequence(dataset, files)
        path = artifact_path(dataset, files[0]["path"])
        base = manifest["base"]
        packets.append({"video_uid": base["video_uid"], "phrase": base["subject_en"], "prompt": base["prompt_en"],
                        "source_frame_sha256": files[0]["sha256"], "image_size": list(frames.shape[1:3]),
                        "image": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()})
    # An empty review is explicit; do not manufacture a box from rejected masks.
    write_json(output / "review_inputs.json", packets)
    template = Path(__file__).with_name("subject_localizer_review.html").read_text()
    (output / "localizer_review.html").write_text(template.replace("/*REVIEW_INPUTS*/[]", json.dumps(packets).replace("</", "<\\/")))


def variant_review(dataset: Path, output: Path) -> None:
    """Display actual stored clean/background frames, without regenerating edits."""
    sections, rows = [], []
    for entry in read_jsonl(dataset / "index.jsonl"):
        manifest = json.loads(artifact_path(dataset, entry["manifest"]).read_text())
        if manifest["status"] != "accepted":
            continue
        variants = manifest["variants"]
        clean = read_png_sequence(dataset, variants["clean"])
        background = read_png_sequence(dataset, variants["full/background_corrupt"])
        examples = []
        for t in sorted({0, len(clean) // 2, len(clean) - 1}):
            figures = []
            for label, key in (("Original", "clean"), ("Background blurred; subject preserved", "full/background_corrupt")):
                path = artifact_path(dataset, variants[key][t]["path"])
                uri = base64.b64encode(path.read_bytes()).decode()
                figures.append(f'<figure><img src="data:image/png;base64,{uri}"><figcaption>{label}</figcaption></figure>')
            examples.append(f'<h3>Frame {t}</h3><div class="pair">{"".join(figures)}</div>')
        base = manifest["base"]
        sections.append(f'<section><h2>{html.escape(base["prompt_en"])}</h2>'
                        f'<p>{html.escape(base["base_id"])}</p>{"".join(examples)}</section>')
        # A first-frame sheet contains every accepted clip in manifest order.
        pair = np.concatenate([cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
                               for frame in (clean[0], background[0])], axis=1)
        row = np.full((338, 600, 3), 250, dtype=np.uint8)
        row[38:] = pair
        cv2.putText(row, base["prompt_en"][:75], (8, 16), cv2.FONT_HERSHEY_SIMPLEX, .39, (20, 20, 20), 1)
        cv2.putText(row, "Original", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1)
        cv2.putText(row, "Background blurred", (308, 32), cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1)
        rows.append(row)
    document = ('<!doctype html><meta charset="utf-8"><title>Subject preserved, background blurred</title>'
                '<style>body{font:16px system-ui;margin:2rem;max-width:1200px}.pair{display:flex;gap:1rem}'
                'figure{margin:0;flex:1}img{max-width:100%}section{border-top:1px solid #ccc;margin-top:2rem}</style>'
                '<h1>Subject preserved, background blurred</h1>'
                '<p>Actual frozen PNGs: first, middle and last frames of every accepted clip. '
                'The subject mask is unchanged; the full background complement receives Gaussian blur. '
                'This preview does not certify the semantic accuracy of the construction masks.</p>' + ''.join(sections))
    (output / "background_blur_review.html").write_text(document)
    if rows:
        if not cv2.imwrite(str(output / "background_blur_preview.png"),
                           cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR)):
            raise OSError("preview PNG write failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--construction", type=Path)
    choice.add_argument("--dataset", type=Path)
    choice.add_argument("--variants", type=Path, help="Preview actual clean/background PNGs for all accepted clips.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = new_output(args.output)
    if args.construction:
        construction_review(args.construction, output)
    elif args.variants:
        variant_review(args.variants, output)
    else:
        localizer_review(args.dataset, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
