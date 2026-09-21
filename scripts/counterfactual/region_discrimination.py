"""Pixel contracts for the subject/background discrimination family.

Isolation is part of the experiment, not an implementation convenience:
construction = SegFormer + GrabCut; scoring localization = independent
MobileSAM prompts; representation = DINO. DINO never chooses pixels, and
construction masks must never be supplied to the scoring mask provider.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


POSITIONS = ("full", "start", "middle", "end")
LEVELS = ("clean", "background_corrupt", "subject_corrupt")


class RejectedBase(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def binary_masks(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 masks {shape}, got {value.dtype} {value.shape}")
    if not np.isin(value, [0, 1]).all():
        raise ValueError("masks must contain only 0 and 1")
    return value


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise RejectedBase("empty_mask")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def mirror_box(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    """Half-open integer box reflected across x = width / 2; no clipping."""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1):
        raise ValueError("invalid box")
    return width - x1, y0, width - x0, y1


def box_area(box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)


def intersection_area(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def window_indices(total: int, position: str) -> tuple[int, ...]:
    if position == "full":
        if total < 2:
            raise RejectedBase("insufficient_frames")
        return tuple(range(total))
    # Python round is ties-to-even. Unlike the legacy family, do not silently
    # turn round(.25 * T) == 0 into an invented one-frame window.
    width = round(0.25 * total)
    if width < 1 or 3 * width > total or total < 3:
        raise RejectedBase("insufficient_frames")
    starts = {"start": 0, "middle": (total - width) // 2, "end": total - width}
    if position not in starts:
        raise ValueError(f"unknown position: {position}")
    return tuple(range(starts[position], starts[position] + width))


def clean_components(mask: np.ndarray, *, person: bool) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 20]
    if person and kept:
        kept = [max(kept, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))]
    result = np.isin(labels, kept).astype(np.uint8)
    if person and result.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - result, connectivity=8)
        border = set(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])).tolist())
        for i in range(1, count):
            if i not in border and stats[i, cv2.CC_STAT_AREA] < 200:
                result[labels == i] = 1
    return result


def refine_grabcut(image: np.ndarray, semantic_mask: np.ndarray, *, person: bool) -> np.ndarray:
    binary_masks(semantic_mask, image.shape[:2])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    core = cv2.erode(semantic_mask, kernel)
    extent = cv2.dilate(semantic_mask, kernel)
    if not core.any() or extent.all():
        raise RejectedBase("grabcut_unusable_seeds")
    seeds = np.full(semantic_mask.shape, cv2.GC_BGD, dtype=np.uint8)
    seeds[extent > 0] = cv2.GC_PR_BGD
    seeds[semantic_mask > 0] = cv2.GC_PR_FGD
    seeds[core > 0] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(image, seeds, None, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), 4, cv2.GC_INIT_WITH_MASK)
    return clean_components(np.isin(seeds, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8), person=person)


def recover_grabcut_extent(image: np.ndarray, mask: np.ndarray, *, person: bool,
                           padding_fraction: float) -> np.ndarray:
    """A separately configured second pass can recover coarse semantic omissions.

    Existing foreground cores stay fixed. Unknown pixels extend to padded
    component boxes rather than only a five-pixel dilation. The complement
    remains certain background; no scoring evidence or other clip is used.
    """
    binary_masks(mask, image.shape[:2])
    if not np.isfinite(padding_fraction) or not 0 < padding_fraction <= 1:
        raise ValueError('invalid component-box padding')
    height, width = mask.shape
    extent = np.zeros_like(mask)
    _, _, components, _ = cv2.connectedComponentsWithStats(mask, 8)
    for x, y, w, h, area in components[1:]:
        if area < 20:
            continue
        pad = int(np.ceil(padding_fraction * max(w, h)))
        extent[max(0,y-pad):min(height,y+h+pad),max(0,x-pad):min(width,x+w+pad)] = 1
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    if not core.any() or extent.all():
        raise RejectedBase('recovery_grabcut_unusable_seeds')
    seeds = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    seeds[extent > 0] = cv2.GC_PR_BGD
    seeds[mask > 0] = cv2.GC_PR_FGD
    seeds[core > 0] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(image, seeds, None, np.zeros((1,65),np.float64),
               np.zeros((1,65),np.float64), 4, cv2.GC_INIT_WITH_MASK)
    return clean_components(np.isin(seeds,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8),person=person)


def inward_alpha(mask: np.ndarray) -> np.ndarray:
    # Padding defines the exterior even when the mask touches an image edge.
    distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return np.minimum(distance / 1.5, 1.0) * mask


def recover_with_fallback(image: np.ndarray, mask: np.ndarray, *, person: bool,
                          padding_fraction: float, keep_valid_initial: bool = False):
    """One recovery rule for fresh and cached construction, without scoring input."""
    valid_initial = .01 <= float(mask.mean()) <= .50
    try:
        current = recover_grabcut_extent(image, mask, person=person, padding_fraction=padding_fraction)
    except RejectedBase as exc:
        if keep_valid_initial and valid_initial:
            return mask.copy(), exc.reason
        raise
    if keep_valid_initial and valid_initial and not .01 <= float(current.mean()) <= .50:
        return mask.copy(), 'recovered_area_outside_frozen_gates'
    return current, None


def corrupt_image(image: np.ndarray, mask: np.ndarray, *, operator: str, scale: float) -> np.ndarray:
    """One float64 blend from the original, inward-only feather, one rounding."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 RGB [H,W,3]")
    binary_masks(mask, image.shape[:2])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("operator scale must be positive and finite")
    if operator == "gaussian":
        edited = cv2.GaussianBlur(image.astype(np.float64), (0, 0), scale, scale,
                                 borderType=cv2.BORDER_REFLECT_101)
    elif operator == "mosaic":
        if int(scale) != scale:
            raise ValueError("mosaic block size must be an integer")
        height, width = image.shape[:2]
        small = cv2.resize(image.astype(np.float64), (max(1, width // int(scale)), max(1, height // int(scale))),
                           interpolation=cv2.INTER_AREA)
        edited = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"unknown operator: {operator}")
    alpha = inward_alpha(mask)[..., None].astype(np.float64)
    blended = np.rint(image.astype(np.float64) * (1 - alpha) + edited * alpha)
    result = np.clip(blended, 0, 255).astype(np.uint8)
    # Explicit checks remain active under python -O.
    if not np.array_equal(result[mask == 0], image[mask == 0]):
        raise AssertionError("corruption changed pixels outside its mask")
    return result


@dataclass
class RegionFamily:
    frames: dict[str, np.ndarray]
    parameters: dict
    proofs: list[dict]


def region_discrimination(frames: np.ndarray, masks: np.ndarray, other_instances: np.ndarray,
                          *, subject: str, position: str = "full", operator: str = "gaussian",
                          background_mode: str = "complement",
                          gaussian_reference_short_side: int | None = None) -> RegionFamily:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be uint8 [T,H,W,3]")
    shape = frames.shape[:3]
    binary_masks(masks, shape)
    binary_masks(other_instances, shape)
    if background_mode not in ("complement", "mirror"):
        raise ValueError("unknown background mode")
    indexes = window_indices(len(frames), position)
    areas = masks.mean(axis=(1, 2))
    if np.any(areas < 0.01):
        raise RejectedBase("mask_area_below_1_percent")
    if np.any(areas > 0.50):
        raise RejectedBase("mask_area_above_50_percent")
    # One preregistered category rule; never tune corruption against scores.
    scale = (18 if subject == "person" else 12) if operator == "gaussian" else (22 if subject == "person" else 16)
    if gaussian_reference_short_side is not None:
        if operator != 'gaussian' or gaussian_reference_short_side <= 0:
            raise ValueError('Gaussian reference short side must be positive and Gaussian-only')
        scale = scale * min(frames.shape[1:3]) / gaussian_reference_short_side
    variants = {level: frames.copy() for level in LEVELS}
    proofs = []
    for t in indexes:
        box = bounding_box(masks[t])
        geometry = {}
        if background_mode == "mirror":
            # Archived v1 replay only. Current v2 uses the entire complement.
            mirrored = mirror_box(box, frames.shape[2])
            if intersection_area(box, mirrored):
                raise RejectedBase("mirror_overlaps_subject_box")
            x0, y0, x1, y1 = mirrored
            if other_instances[t, y0:y1, x0:x1].any():
                raise RejectedBase("mirror_contains_other_instance")
            background = np.ascontiguousarray(masks[t, :, ::-1])
            if box_area(box) != box_area(mirrored) or int(background.sum()) != int(masks[t].sum()):
                raise AssertionError("unequal intervention area")
            geometry = {"mirror_box": list(mirrored), "mirror_box_area": box_area(mirrored),
                        "box_intersection_area": 0, "other_instance_pixels_in_mirror_box": 0}
        else:
            # User clarification: extract the subject, blur EVERYTHING ELSE.
            # Other objects are background relative to that designated subject.
            background = 1 - masks[t]
            if np.any(background & masks[t]) or not np.all(background | masks[t]):
                raise AssertionError("subject and background must partition every pixel")
        changes = {}
        for level, edit_mask in (("subject_corrupt", masks[t]), ("background_corrupt", background)):
            result = corrupt_image(frames[t], edit_mask, operator=operator, scale=scale)
            variants[level][t] = result
            changed = np.any(result != frames[t], axis=-1)
            outside = int(np.count_nonzero(changed & (edit_mask == 0)))
            if outside:
                raise AssertionError("pixels outside intervention changed")
            changes[level] = {"changed_pixels": int(changed.sum()), "outside_mask_changed_pixels": outside}
        subject_changed = int(np.count_nonzero(np.any(variants["background_corrupt"][t] != frames[t], axis=-1) & (masks[t] > 0)))
        if subject_changed:
            raise AssertionError("background blur changed subject pixels")
        proofs.append({"frame": t, "subject_box": list(box),
                       "box_area": box_area(box),
                       "subject_mask_area": int(masks[t].sum()), "background_mask_area": int(background.sum()),
                       "subject_pixels_changed_by_background": subject_changed,
                       **geometry, **changes})
    parameters = {"operator": operator, "scale": scale, "feather_px": 1.5,
                                   "position": position, "window_indices": list(indexes),
                                   "window_frames": len(indexes), "window_fraction": 1.0 if position == "full" else 0.25,
                                   "background_support": "subject_mask_complement" if background_mode == "complement" else "horizontally_reflected_subject_mask",
                                   "expected": "clean ~= background_corrupt > subject_corrupt"}
    if gaussian_reference_short_side is not None:
        parameters['gaussian_reference_short_side'] = gaussian_reference_short_side
    return RegionFamily(variants, parameters, proofs)
