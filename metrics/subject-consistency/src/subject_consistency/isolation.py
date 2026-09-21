"""Isolate subject pixels before a globally mixing image encoder sees them.

Only independently predicted scoring masks may enter an experiment's scorer.
For a fixed binary mask, edits outside its support cannot affect these inputs.
This conditional guarantee does not assert that a predicted mask is correct or
invariant to corruption. Both localization drift and boundary errors remain
measurable end-to-end failure modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IsolatedInputs:
    frames: Any
    masks: Any
    present: Any
    source_coverage: tuple[float, ...]
    boxes: tuple[tuple[int, int, int, int] | None, ...]


def isolate_subject_inputs(frames: Any, masks: Any, present: Any, *, view: str = "crop",
                           fill: int = 128, output_size: int = 224, margin: float = .10) -> IsolatedInputs:
    import math
    import torch
    from torch.nn import functional as F
    from .models import native_rgb_uint8

    if view not in ("full", "crop"):
        raise ValueError("subject view must be full or crop")
    if type(fill) is not int or not 0 <= fill <= 255 or output_size < 1 or not 0 <= margin <= 1:
        raise ValueError("invalid isolation parameters")
    frames = native_rgb_uint8(frames)
    if masks.ndim != 4 or masks.shape[0] != frames.shape[0] or masks.shape[2:] != frames.shape[2:]:
        raise ValueError("isolation masks must be [T,I,H,W] at native decoded resolution")
    if masks.shape[1] < 1 or tuple(present.shape) != tuple(masks.shape[:2]):
        raise ValueError("present must have shape [T,I]")
    if not bool(torch.isfinite(masks).all()) or bool(((masks < 0) | (masks > 1)).any()):
        raise ValueError("mask values must be finite and in [0,1]")
    masks = (masks.to(frames.device) > .5) & present.to(device=frames.device, dtype=torch.bool)[:, :, None, None]
    present = masks.flatten(2).any(-1)
    union = masks.any(1)
    coverage = tuple(float(x) for x in union.float().mean((1, 2)).cpu().tolist())
    isolated = torch.where(union[:, None], frames, torch.full_like(frames, fill))
    if view == "full":
        return IsolatedInputs(isolated, masks.float(), present, coverage, (None,) * len(frames))

    images, aligned, boxes = [], [], []
    for image, instances, support in zip(isolated, masks, union):
        coords = support.nonzero()
        if not len(coords):
            images.append(torch.full((3, output_size, output_size), fill, dtype=frames.dtype, device=frames.device))
            aligned.append(torch.zeros((masks.shape[1], output_size, output_size), device=frames.device))
            boxes.append(None)
            continue
        y0, x0 = coords.amin(0).tolist()
        y1, x1 = (coords.amax(0) + 1).tolist()
        side = math.ceil(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
        left, top = math.floor((x0 + x1 - side) / 2), math.floor((y0 + y1 - side) / 2)
        right, bottom = left + side, top + side
        # Centering and padding are based solely on the scoring mask, never on
        # construction masks, DINO features, a clean reference, or score changes.
        canvas = torch.full((3, side, side), fill, dtype=frames.dtype, device=frames.device)
        weights = torch.zeros((masks.shape[1], side, side), dtype=torch.float32, device=frames.device)
        sx0, sy0 = max(0, left), max(0, top)
        sx1, sy1 = min(frames.shape[3], right), min(frames.shape[2], bottom)
        canvas[:, sy0-top:sy1-top, sx0-left:sx1-left] = image[:, sy0:sy1, sx0:sx1]
        weights[:, sy0-top:sy1-top, sx0-left:sx1-left] = instances[:, sy0:sy1, sx0:sx1].float()
        images.append(F.interpolate(canvas[None].float(), (output_size, output_size), mode="bilinear",
                                    align_corners=False, antialias=True)[0].round().clamp(0, 255).to(torch.uint8))
        aligned.append(F.interpolate(weights[None], (output_size, output_size), mode="area")[0])
        boxes.append((left, top, right, bottom))
    return IsolatedInputs(torch.stack(images), torch.stack(aligned), present, coverage, tuple(boxes))
