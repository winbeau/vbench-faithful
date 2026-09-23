#!/usr/bin/env python3
"""Build the two-panel VBench counterfactual case figure.

The source frames are read from source/. The script creates the intermediate
crops/box overlays first and then lays out the final PNG/PDF.
All numbers are from the frozen H100 matrix artifacts:

* Scene: LaVie ocean-0; Origin "an ocean" = 12/16 and "a sea" = 0/16.
* Spatial: the published horizontal-mirror gallery case; A=traffic light and
  B=bus, with Origin score 1.0 before and after the mirror.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
INTERMEDIATE = ROOT / "intermediate"
OUTPUT = ROOT / "output"

CANVAS_SIZE = (2400, 1280)
MARGIN = 42
PANEL_GAP = 42

COLORS = {
    "ink": "#20252b",
    "muted": "#5f6b76",
    "line": "#c9d1d8",
    "panel": "#ffffff",
    "blue": "#1f5f8b",
    "blue_light": "#eaf3f9",
    "red": "#b23a48",
    "red_light": "#fff0f1",
    "green": "#2d7a5b",
    "green_light": "#edf7f1",
    "orange": "#c66b2d",
    "orange_light": "#fff3e8",
    "purple": "#69549a",
}

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_image(image: Image.Image, width: int, height: int, crop: tuple[int, int, int, int] | None = None) -> Image.Image:
    image = image.convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    return ImageOps.fit(image, (width, height), method=RESAMPLE_LANCZOS, centering=(0.5, 0.48))


def paste_rounded(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int], radius: int = 18) -> None:
    x0, y0, x1, y1 = box
    image = fit_image(image, x1 - x0, y1 - y0)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=radius, fill=255)
    canvas.paste(image, (x0, y0), mask)
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=radius, outline=COLORS["line"], width=3)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, face: ImageFont.FreeTypeFont, fill: str) -> None:
    x, y = xy
    bounds = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (bounds[2] - bounds[0]) / 2, y - (bounds[3] - bounds[1]) / 2), text, font=face, fill=fill)


def wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, face: ImageFont.FreeTypeFont, fill: str, width: int, spacing: int = 8) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textbbox((0, 0), trial, font=face)[2] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for index, line in enumerate(lines):
        draw.text((x, y + index * (face.size + spacing)), line, font=face, fill=fill)
    return y + len(lines) * (face.size + spacing)


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str = COLORS["line"], radius: int = 18, width: int = 3) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str, width: int = 8, head: int = 18) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        points = [(x1, y1), (x1 - direction * head, y1 - head // 2), (x1 - direction * head, y1 + head // 2)]
    else:
        direction = 1 if y1 >= y0 else -1
        points = [(x1, y1), (x1 - head // 2, y1 - direction * head), (x1 + head // 2, y1 - direction * head)]
    draw.polygon(points, fill=fill)


def prepare_scene_crop() -> Path:
    source = SOURCE / "scene-ocean-frame.png"
    image = Image.open(source)
    # The source frame is a square. Trim the lower watermark-heavy edge while
    # retaining the underwater context and the reef horizon.
    crop = (0, 0, image.width, int(image.height * 0.86))
    prepared = fit_image(image, 720, 500, crop=crop)
    target = INTERMEDIATE / "scene_ocean_frame_crop.png"
    prepared.save(target, optimize=True)
    return target


def prepare_spatial_overlay(source_name: str, target_name: str, boxes: Iterable[tuple[str, tuple[float, float, float, float], str]]) -> Path:
    source = SOURCE / source_name
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x = image.width / 480.0
    scale_y = image.height / 480.0
    label_face = font(28, bold=True)
    for label, raw_box, color in boxes:
        x0, y0, x1, y1 = [int(value) for value in raw_box]
        box = (int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y))
        draw.rounded_rectangle(box, radius=8, outline=color, width=6)
        tag_box = (box[0] + 5, max(4, box[1] + 5), box[0] + 55, max(4, box[1] + 5) + 42)
        draw.rounded_rectangle(tag_box, radius=8, fill=color)
        centered(draw, ((tag_box[0] + tag_box[2]) // 2, (tag_box[1] + tag_box[3]) // 2), label, label_face, "#ffffff")
    target = INTERMEDIATE / target_name
    image.save(target, optimize=True)
    return target


def draw_prompt_card(canvas: Image.Image, box: tuple[int, int, int, int], title: str, prompt: str, accent: str) -> None:
    draw = ImageDraw.Draw(canvas)
    rounded_box(draw, box, "#fafcfd", outline=accent, width=4)
    x0, y0, x1, y1 = box
    draw.text((x0 + 24, y0 + 18), title, font=font(22, bold=True), fill=COLORS["muted"])
    centered(draw, ((x0 + x1) // 2, (y0 + y1) // 2 + 12), prompt, font(32, bold=True), accent)


def draw_scene_panel(canvas: Image.Image, panel: tuple[int, int, int, int], scene_image: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = panel
    rounded_box(draw, panel, COLORS["panel"], outline=COLORS["line"], radius=24, width=3)
    draw.text((x0 + 28, y0 + 22), "(a)", font=font(30, bold=True), fill=COLORS["ink"])
    draw.text((x0 + 95, y0 + 22), "Scene: synonym sensitivity", font=font(30, bold=True), fill=COLORS["ink"])
    draw.line((x0 + 28, y0 + 78, x1 - 28, y0 + 78), fill=COLORS["line"], width=3)

    prompt_y = y0 + 106
    card_w = 390
    draw_prompt_card(canvas, (x0 + 28, prompt_y, x0 + 28 + card_w, prompt_y + 112), "Prompt A", '"an ocean"', COLORS["blue"])
    draw_prompt_card(canvas, (x1 - 28 - card_w, prompt_y, x1 - 28, prompt_y + 112), "Prompt B", '"a sea"', COLORS["red"])
    arrow(draw, (x0 + 28 + card_w + 36, prompt_y + 56), (x1 - 28 - card_w - 36, prompt_y + 56), COLORS["purple"], width=7, head=18)
    centered(draw, ((x0 + x1) // 2, prompt_y + 22), "same video", font(18, bold=True), COLORS["purple"])
    centered(draw, ((x0 + x1) // 2, prompt_y + 91), "synonym only", font(18), COLORS["muted"])

    image_box = (x0 + 28, y0 + 260, x0 + 490, y0 + 740)
    paste_rounded(canvas, scene_image, image_box, radius=20)
    centered(draw, ((image_box[0] + image_box[2]) // 2, image_box[3] + 25), "Representative frame: LaVie / ocean-0", font(17), COLORS["muted"])

    score_box = (x0 + 520, y0 + 260, x1 - 28, y0 + 740)
    rounded_box(draw, score_box, COLORS["blue_light"], outline=COLORS["line"], radius=20, width=3)
    sx0, sy0, sx1, sy1 = score_box
    draw.text((sx0 + 28, sy0 + 25), "Official Origin scene score", font=font(24, bold=True), fill=COLORS["ink"])
    draw.text((sx0 + 28, sy0 + 88), "ocean", font=font(24, bold=True), fill=COLORS["blue"])
    draw.text((sx0 + 28, sy0 + 126), "12 / 16 frames", font=font(21), fill=COLORS["muted"])
    draw.text((sx1 - 265, sy0 + 77), "0.750", font=font(50, bold=True), fill=COLORS["blue"])
    arrow(draw, ((sx0 + sx1) // 2, sy0 + 188), ((sx0 + sx1) // 2, sy0 + 255), COLORS["red"], width=9, head=22)
    draw.text((sx0 + 28, sy0 + 287), "sea", font=font(24, bold=True), fill=COLORS["red"])
    draw.text((sx0 + 28, sy0 + 325), "0 / 16 frames", font=font(21), fill=COLORS["muted"])
    draw.text((sx1 - 265, sy0 + 276), "0.000", font=font(50, bold=True), fill=COLORS["red"])
    rounded_box(draw, (sx0 + 24, sy1 - 70, sx1 - 24, sy1 - 22), COLORS["red_light"], outline=COLORS["red"], radius=12, width=2)
    centered(draw, ((sx0 + sx1) // 2, sy1 - 46), "Δ = -0.750  (representative video)", font(22, bold=True), COLORS["red"])
    draw.text((sx0 + 28, sy1 + 30), "20-video dev mean: 0.56250  ->  0.03125", font=font(19), fill=COLORS["muted"])

    callout = (x0 + 28, y0 + 820, x1 - 28, y1 - 28)
    rounded_box(draw, callout, "#fbfcfd", outline=COLORS["line"], radius=18, width=2)
    centered(draw, ((callout[0] + callout[2]) // 2, callout[1] + 38), "Same video and same captions; only the prompt word changes.", font(21), COLORS["ink"])
    centered(draw, ((callout[0] + callout[2]) // 2, callout[1] + 92), "Semantic meaning unchanged, but the score drops.", font(24, bold=True), COLORS["red"])


def draw_spatial_panel(canvas: Image.Image, panel: tuple[int, int, int, int], before: Image.Image, after: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = panel
    rounded_box(draw, panel, COLORS["panel"], outline=COLORS["line"], radius=24, width=3)
    draw.text((x0 + 28, y0 + 22), "(b)", font=font(30, bold=True), fill=COLORS["ink"])
    draw.text((x0 + 95, y0 + 22), "Spatial: direction blindness", font=font(30, bold=True), fill=COLORS["ink"])
    draw.line((x0 + 28, y0 + 78, x1 - 28, y0 + 78), fill=COLORS["line"], width=3)

    image_y = y0 + 112
    image_w = 460
    image_h = 438
    left_box = (x0 + 28, image_y, x0 + 28 + image_w, image_y + image_h)
    right_box = (x1 - 28 - image_w, image_y, x1 - 28, image_y + image_h)
    paste_rounded(canvas, before, left_box, radius=18)
    paste_rounded(canvas, after, right_box, radius=18)
    centered(draw, ((left_box[0] + left_box[2]) // 2, left_box[1] - 23), "Before mirror", font(21, bold=True), COLORS["ink"])
    centered(draw, ((right_box[0] + right_box[2]) // 2, right_box[1] - 23), "After horizontal mirror", font(21, bold=True), COLORS["ink"])
    arrow(draw, (left_box[2] + 24, image_y + image_h // 2), (right_box[0] - 24, image_y + image_h // 2), COLORS["purple"], width=8, head=20)
    centered(draw, ((left_box[2] + right_box[0]) // 2, image_y + image_h // 2 - 28), "hflip", font(18, bold=True), COLORS["purple"])

    relation_y = y0 + 600
    relation_w = 405
    draw_prompt_card(canvas, (x0 + 28, relation_y, x0 + 28 + relation_w, relation_y + 115), "Original", "A left of B", COLORS["blue"])
    draw_prompt_card(canvas, (x1 - 28 - relation_w, relation_y, x1 - 28, relation_y + 115), "After mirror", "A right of B", COLORS["orange"])
    arrow(draw, (x0 + 28 + relation_w + 40, relation_y + 58), (x1 - 28 - relation_w - 40, relation_y + 58), COLORS["purple"], width=7, head=18)
    centered(draw, ((x0 + x1) // 2, relation_y + 20), "semantic reversal", font(18, bold=True), COLORS["purple"])
    centered(draw, ((x0 + x1) // 2, relation_y + 93), "A = traffic light; B = bus", font(17), COLORS["muted"])

    score_box = (x0 + 28, y0 + 762, x1 - 28, y0 + 958)
    rounded_box(draw, score_box, COLORS["orange_light"], outline=COLORS["orange"], radius=18, width=3)
    sx0, sy0, sx1, sy1 = score_box
    centered(draw, ((sx0 + sx1) // 2, sy0 + 48), "Official Origin score", font(23, bold=True), COLORS["ink"])
    centered(draw, ((sx0 + sx1) // 2, sy0 + 107), "1.000  ->  1.000", font(47, bold=True), COLORS["orange"])
    centered(draw, ((sx0 + sx1) // 2, sy0 + 162), "Δ = 0.000", font(24, bold=True), COLORS["red"])

    callout = (x0 + 28, y0 + 1000, x1 - 28, y1 - 28)
    rounded_box(draw, callout, "#fbfcfd", outline=COLORS["line"], radius=18, width=2)
    centered(draw, ((callout[0] + callout[2]) // 2, callout[1] + 38), "Target relation changes, but the score stays unchanged.", font(22, bold=True), COLORS["red"])
    centered(draw, ((callout[0] + callout[2]) // 2, callout[1] + 88), "The locked geometry uses absolute offsets, not direction signs.", font(19), COLORS["muted"])


def main() -> None:
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    scene_crop_path = prepare_scene_crop()
    spatial_before_path = prepare_spatial_overlay(
        "spatial-before.png",
        "spatial_before_annotated.png",
        [
            ("A", (119, 0, 321, 305), COLORS["blue"]),
            ("B", (425, 194, 479, 318), COLORS["orange"]),
        ],
    )
    spatial_after_path = prepare_spatial_overlay(
        "spatial-after.png",
        "spatial_after_annotated.png",
        [
            ("A", (159, 0, 361, 305), COLORS["blue"]),
            ("B", (1, 194, 55, 318), COLORS["orange"]),
        ],
    )

    canvas = Image.new("RGB", CANVAS_SIZE, "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    panel_top = MARGIN
    panel_bottom = CANVAS_SIZE[1] - MARGIN
    panel_width = (CANVAS_SIZE[0] - 2 * MARGIN - PANEL_GAP) // 2
    left_panel = (MARGIN, panel_top, MARGIN + panel_width, panel_bottom)
    right_panel = (MARGIN + panel_width + PANEL_GAP, panel_top, CANVAS_SIZE[0] - MARGIN, panel_bottom)
    draw_scene_panel(canvas, left_panel, Image.open(scene_crop_path))
    draw_spatial_panel(canvas, right_panel, Image.open(spatial_before_path), Image.open(spatial_after_path))

    final_png = OUTPUT / "vbench_counterfactual_cases_2panel.png"
    final_pdf = OUTPUT / "vbench_counterfactual_cases_2panel.pdf"
    canvas.save(final_png, dpi=(300, 300), optimize=True)
    canvas.save(final_pdf, "PDF", resolution=300.0)

    provenance = {
        "figure": "VBench two-panel counterfactual cases",
        "source_project": "/root/wenbiao_zhao/vbench-prompts-compile-git",
        "source_matrix_commit": "ea9d500",
        "scoring_code_commit": "4d2a78d",
        "scene": {
            "video": "videos/lavie/scene/ocean-0.mp4",
            "prompt_base": "an ocean",
            "prompt_counterfactual": "a sea",
            "frame_caption_source": "data/backend-cache/ocean-dev-v1.jsonl",
            "selected_video_score": {"ocean": 0.75, "sea": 0.0, "ocean_hits": 12, "sea_hits": 0, "frames": 16},
            "dev_mean": {"ocean": 0.5625, "sea": 0.03125, "videos": 20},
            "source_sha256": sha256(SOURCE / "scene-ocean-frame.png"),
        },
        "spatial": {
            "case": "horizontal mirror",
            "prompt": "a bus on the right of a traffic light, front view",
            "semantic_relabeling": "A=traffic light, B=bus: A left of B -> A right of B",
            "origin_score_before": 1.0,
            "origin_score_after": 1.0,
            "source_sha256_before": sha256(SOURCE / "spatial-before.png"),
            "source_sha256_after": sha256(SOURCE / "spatial-after.png"),
        },
        "outputs": {
            "png": str(final_png.relative_to(ROOT)),
            "pdf": str(final_pdf.relative_to(ROOT)),
            "intermediate_scene": str(scene_crop_path.relative_to(ROOT)),
            "intermediate_spatial_before": str(spatial_before_path.relative_to(ROOT)),
            "intermediate_spatial_after": str(spatial_after_path.relative_to(ROOT)),
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"png": str(final_png), "pdf": str(final_pdf), "provenance": str(OUTPUT / "provenance.json")}, indent=2))


if __name__ == "__main__":
    main()
