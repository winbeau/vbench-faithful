"""Deterministic counterfactual transforms for the seven audited dimensions.

Every family is a pure function of the decoded base frames (plus, where the
contract demands it, a detector box).  Nothing here consults a metric score:
base selection lives in `select_bases.py` and is metadata-only, so the derived
set cannot be biased towards a desired outcome (plan section 3.3).

Expected relations are expressed as integer *ranks* rather than pairwise
predictions.  Variants sharing a rank are expected to tie; a higher rank is
expected to score higher.  `pairs.py`-style expansion happens in `build.py`,
which turns ranks into `expected_relation in {-1, 0, +1}` exactly as plan
section 5.2 requires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .common import CounterfactualError, VideoMeta, resample_indices

# --------------------------------------------------------------------------
# Dynamic Degree: FPS invariance
# --------------------------------------------------------------------------

# Downsampling-only ladder.  Measured source rates are 8 fps (H.264) and 10 fps
# (CogVideo GIF), so rungs above the native rate are skipped rather than
# synthesised: duplicating frames would inject temporal jerk and confound the
# Motion Smoothness dimension, and interpolating them would need a model.
FPS_LADDER = (8.0, 6.0, 4.0, 2.0)


def fps_resample(frames: np.ndarray, meta: VideoMeta, ladder: Sequence[float] = FPS_LADDER) -> list["Variant"]:
    """Sample the same trajectory at several frame rates, preserving duration."""
    variants: list[Variant] = []
    for target in ladder:
        if target > meta.fps + 1e-9:
            continue
        indices = resample_indices(len(frames), meta.fps, float(target))
        variants.append(
            Variant(
                name=f"fps{_rate_label(target)}",
                frames=frames[indices],
                fps=float(target),
                expected_rank=1,
                parameters={
                    "target_fps": float(target),
                    "native_fps": float(meta.fps),
                    "kept_frame_count": len(indices),
                    "kept_frame_indices": indices,
                },
                note="trajectory preserved; only the temporal sampling rate changes",
            )
        )
    if not variants:
        raise CounterfactualError(f"no FPS rung is at or below the native rate {meta.fps}")
    return variants


def _rate_label(fps: float) -> str:
    return str(int(fps)) if float(fps).is_integer() else str(fps).replace(".", "p")


# --------------------------------------------------------------------------
# Subject Consistency: temporal relocation of a fixed corruption
# --------------------------------------------------------------------------


def temporal_relocation(
    frames: np.ndarray,
    meta: VideoMeta,
    boxes: Sequence[Sequence[int]] | Sequence[int],
    duration_fraction: float = 0.25,
) -> list["Variant"]:
    """Place one fixed subject-region corruption at the start, middle and end.

    The corruption (localised blur plus a fixed hue rotation) has identical
    intensity and duration in all three positions; only its temporal location
    changes, so any score gap between them is a position artefact rather than a
    difference in corruption severity.

    `boxes` is one tracked box per frame (plan section 7.5 keeps a fixed ROI for
    the ablation only), but a single box is accepted and broadcast.
    """
    count = len(frames)
    tracked = per_frame_boxes(boxes, count)
    window = max(1, int(round(duration_fraction * count)))
    if window * 3 > count:
        raise CounterfactualError(
            f"clip too short for three non-overlapping {window}-frame windows: {count} frames"
        )
    starts = {
        "start": 0,
        "middle": (count - window) // 2,
        "end": count - window,
    }
    variants = [
        Variant(
            name="clean",
            frames=frames.copy(),
            fps=meta.fps,
            expected_rank=1,
            parameters={"window_frames": window, "boxes": tracked},
            note="unmodified control; must outrank every corrupted variant",
        )
    ]
    for position, start in starts.items():
        corrupted = frames.copy()
        indexes = list(range(start, start + window))
        corrupted[indexes] = _corrupt_region(corrupted[indexes], [tracked[i] for i in indexes])
        variants.append(
            Variant(
                name=f"corrupt_{position}",
                frames=corrupted,
                fps=meta.fps,
                expected_rank=0,
                parameters={
                    "window_frames": window,
                    "window_start": start,
                    "window_indices": indexes,
                    "boxes": tracked,
                    "window_boxes": [tracked[i] for i in indexes],
                    "intensity": CORRUPTION_INTENSITY,
                },
                note="same corruption, different temporal position",
            )
        )
    return variants


def per_frame_boxes(
    boxes: Sequence[Sequence[int]] | Sequence[int], count: int
) -> list[tuple[int, int, int, int]]:
    """Normalise a single box or a tracked sequence into one box per frame."""
    values = list(boxes)
    if not values:
        raise CounterfactualError("no box supplied")
    if any(value is None for value in values):
        raise CounterfactualError("box sequence contains a missing detection")
    first = values[0]
    if isinstance(first, (int, float, np.integer, np.floating)):
        single = tuple(int(round(float(value))) for value in values)
        if len(single) != 4:
            raise CounterfactualError(f"a box must have four coordinates, got {len(single)}")
        return [single] * count  # type: ignore[list-item]
    if len(values) != count:
        raise CounterfactualError(
            f"expected one box per frame ({count}), got {len(values)}"
        )
    return [tuple(int(round(float(value))) for value in box) for box in values]  # type: ignore[misc]


CORRUPTION_INTENSITY = {"blur_radius": 6, "hue_shift_degrees": 120}


def _corrupt_region(region: np.ndarray, boxes: Sequence[Sequence[int]]) -> np.ndarray:
    """Blur and hue-rotate a per-frame detection box, deterministically."""
    from PIL import Image, ImageFilter

    shifted = np.empty_like(region)
    for index, frame in enumerate(region):
        x0, y0, x1, y1 = _clamp_box(boxes[index], region.shape[2], region.shape[1])
        if x1 <= x0 or y1 <= y0:
            raise CounterfactualError(f"degenerate corruption box: {list(boxes[index])}")
        patch = Image.fromarray(frame[y0:y1, x0:x1])
        patch = patch.filter(ImageFilter.GaussianBlur(CORRUPTION_INTENSITY["blur_radius"]))
        blurred = np.asarray(patch)
        shifted[index] = frame.copy()
        shifted[index][y0:y1, x0:x1] = _rotate_hue(blurred, CORRUPTION_INTENSITY["hue_shift_degrees"])
    return shifted


def _rotate_hue(rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate hue in HSV space and return uint8 RGB."""
    import colorsys

    flat = rgb.reshape(-1, 3).astype(np.float64) / 255.0
    out = np.empty_like(flat)
    for index, (red, green, blue) in enumerate(flat):
        hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
        hue = (hue + degrees / 360.0) % 1.0
        out[index] = colorsys.hsv_to_rgb(hue, saturation, value)
    return np.clip(out.reshape(rgb.shape) * 255.0, 0, 255).astype(np.uint8)


def _clamp_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


# --------------------------------------------------------------------------
# Human Action: filename invariance (no pixel change at all)
# --------------------------------------------------------------------------


def filename_invariance(correct_action: str, wrong_action: str) -> list["Variant"]:
    """Byte-identical copies under a correct, wrong and neutral filename.

    This family never touches pixels: the builder copies the source bytes, so
    any score change is attributable to the filename alone.
    """
    return [
        Variant(
            name="filename_correct",
            frames=None,
            fps=0.0,
            expected_rank=1,
            parameters={"filename_action": correct_action, "copy_bytes": True},
            note="canonical Kinetics-400 action name in the filename",
        ),
        Variant(
            name="filename_wrong",
            frames=None,
            fps=0.0,
            expected_rank=1,
            parameters={"filename_action": wrong_action, "copy_bytes": True},
            note="semantically related but incorrect Kinetics-400 action name",
        ),
        Variant(
            name="filename_neutral",
            frames=None,
            fps=0.0,
            expected_rank=1,
            parameters={"filename_action": None, "copy_bytes": True},
            note="neutral filename carrying no action cue",
        ),
    ]


# --------------------------------------------------------------------------
# Spatial Relationship: directional flip
# --------------------------------------------------------------------------


def directional_flip(frames: np.ndarray, meta: VideoMeta, relation: str) -> list["Variant"]:
    """Flip the axis named by the relation so the ordered query becomes false."""
    axis = flip_axis(relation)
    if axis == "horizontal":
        flipped = frames[:, :, ::-1, :]
    else:
        flipped = frames[:, ::-1, :, :]
    return [
        Variant(
            name="original",
            frames=frames.copy(),
            fps=meta.fps,
            expected_rank=1,
            parameters={"flip_axis": axis, "relation": relation},
            note="re-encoded control sharing the encoder settings of the flip",
        ),
        Variant(
            name=f"{axis}_flip",
            frames=np.ascontiguousarray(flipped),
            fps=meta.fps,
            expected_rank=0,
            parameters={"flip_axis": axis, "relation": relation},
            note="same prompt, mirrored geometry; the ordered relation no longer holds",
        ),
    ]


HORIZONTAL_RELATIONS = ("left", "right")
VERTICAL_RELATIONS = ("top", "bottom", "above", "below", "under")


def flip_axis(relation: str) -> str:
    relation = relation.strip().lower()
    if relation in HORIZONTAL_RELATIONS:
        return "horizontal"
    if relation in VERTICAL_RELATIONS:
        return "vertical"
    raise CounterfactualError(f"relation {relation!r} has no defined flip axis")


def reciprocal_relation(relation: str) -> str:
    """Swap the roles of an ordered relation so a flip restores the query."""
    relation = relation.strip().lower()
    pairs = {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
        "above": "below",
        "below": "above",
        "under": "on",
        "on": "under",
    }
    if relation not in pairs:
        raise CounterfactualError(f"relation {relation!r} has no reciprocal")
    return pairs[relation]


# --------------------------------------------------------------------------
# Scene: 2x2 environment coverage
# --------------------------------------------------------------------------


QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _quadrant_slices(height: int, width: int) -> dict[str, tuple[slice, slice]]:
    half_h = height // 2
    half_w = width // 2
    return {
        "top_left": (slice(0, half_h), slice(0, half_w)),
        "top_right": (slice(0, half_h), slice(half_w, width)),
        "bottom_left": (slice(half_h, height), slice(0, half_w)),
        "bottom_right": (slice(half_h, height), slice(half_w, width)),
    }


def environment_coverage(
    target_frames: np.ndarray,
    donor_frames: np.ndarray,
    meta: VideoMeta,
    order: Sequence[str] = QUADRANTS,
) -> list["Variant"]:
    """Mix wrong-scene donor quadrants with target-scene quadrants at 0-100%.

    Level k carries exactly k target quadrants, so target environmental support
    rises in fixed 25% steps.  The donor is resampled onto the target clip's
    timeline by normalised time, and both clips must share a resolution (the
    builder only pairs clips from the same generator).
    """
    if len(order) != 4 or set(order) != set(QUADRANTS):
        raise CounterfactualError(f"quadrant order must be a permutation of {QUADRANTS}")
    if target_frames.shape[1:] != donor_frames.shape[1:]:
        raise CounterfactualError(
            f"resolution mismatch: {target_frames.shape[1:]} vs {donor_frames.shape[1:]}"
        )
    aligned = _align_frames(donor_frames, len(target_frames))
    height, width = target_frames.shape[1], target_frames.shape[2]
    slices = _quadrant_slices(height, width)
    variants: list[Variant] = []
    for level in range(5):
        mixed = aligned.copy()
        filled = list(order[:level])
        for name in filled:
            rows, columns = slices[name]
            mixed[:, rows, columns, :] = target_frames[:, rows, columns, :]
        variants.append(
            Variant(
                name=f"coverage_{level * 25:03d}",
                frames=mixed,
                fps=meta.fps,
                expected_rank=level,
                parameters={
                    "target_quadrants": filled,
                    "target_fraction": level / 4.0,
                    "quadrant_order": list(order),
                },
                note="target environmental support rises in fixed 25% steps",
            )
        )
    return variants


def _align_frames(frames: np.ndarray, count: int) -> np.ndarray:
    """Nearest-frame resampling onto a different frame count by normalised time."""
    source = len(frames)
    if source == count:
        return frames.copy()
    if source == 1:
        return np.repeat(frames, count, axis=0)
    positions = np.round(np.linspace(0, source - 1, count)).astype(int)
    return frames[positions]


# --------------------------------------------------------------------------
# Multiple Objects: weakest-object visibility
# --------------------------------------------------------------------------


VISIBILITY_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def weakest_object_visibility(
    frames: np.ndarray,
    meta: VideoMeta,
    boxes: Sequence[Sequence[int]] | Sequence[int],
    levels: Sequence[float] = VISIBILITY_LEVELS,
) -> list["Variant"]:
    """Weaken target B by alpha-blending its tracked box towards the local mean.

    Target A is untouched.  At severity 1.0 the region collapses to a flat patch,
    so B is no longer visible; severity 0.0 is the unmodified control.
    """
    count = len(frames)
    tracked = per_frame_boxes(boxes, count)
    variants: list[Variant] = []
    for severity in levels:
        blended = frames.copy()
        for index, box in enumerate(tracked):
            x0, y0, x1, y1 = _clamp_box(box, frames.shape[2], frames.shape[1])
            region = blended[index, y0:y1, x0:x1, :]
            patch_mean = region.reshape(-1, 3).mean(axis=0)
            blended[index, y0:y1, x0:x1, :] = np.clip(
                region.astype(np.float64) * (1.0 - severity) + patch_mean * severity, 0, 255
            ).astype(np.uint8)
        variants.append(
            Variant(
                name=f"occlusion_{int(round(severity * 100)):03d}",
                frames=blended,
                fps=meta.fps,
                expected_rank=int(round((1.0 - severity) * 4)),
                parameters={
                    "severity": float(severity),
                    "boxes": tracked,
                    "operator": "alpha_blend_to_region_mean",
                },
                note="target A untouched; target B progressively suppressed",
            )
        )
    return variants


def temporal_conjunction_control(
    frames: np.ndarray,
    meta: VideoMeta,
    boxes_a: Sequence[Sequence[int]] | Sequence[int],
    boxes_b: Sequence[Sequence[int]] | Sequence[int],
) -> list["Variant"]:
    """Only A in the first half and only B in the second: no frame holds both.

    Both backends should score this incomplete; it guards against a repair that
    silently converts same-frame conjunction into temporal union.
    """
    count = len(frames)
    half = count // 2
    if half < 1 or count - half < 1:
        raise CounterfactualError(f"clip too short for a conjunction control: {count} frames")
    tracked_a = per_frame_boxes(boxes_a, count)
    tracked_b = per_frame_boxes(boxes_b, count)
    both = frames.copy()
    both[:half] = _suppress(frames[:half], tracked_a[:half])
    both[half:] = _suppress(frames[half:], tracked_b[half:])
    return [
        Variant(
            name="conjunction_control",
            frames=both,
            fps=meta.fps,
            expected_rank=0,
            parameters={
                "boxes_a": tracked_a,
                "boxes_b": tracked_b,
                "split_frame": half,
            },
            note="A only in the first half, B only in the second; never co-present",
        )
    ]


def _suppress(region: np.ndarray, boxes: Sequence[Sequence[int]]) -> np.ndarray:
    out = region.copy()
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = _clamp_box(box, out.shape[2], out.shape[1])
        patch = out[index, y0:y1, x0:x1, :]
        out[index, y0:y1, x0:x1, :] = patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# Motion Smoothness: temporal jerk severity
# --------------------------------------------------------------------------


def temporal_jerk(frames: np.ndarray, meta: VideoMeta) -> list["Variant"]:
    """Perturb temporal order while keeping frame count, rate and duration fixed.

    Levels follow plan section 13.3.  All edits reorder or repeat existing
    frames, so no new visual content is introduced and any encoding artefact is
    shared with the level-0 control.
    """
    count = len(frames)
    if count < 8:
        raise ConversionTooShort(count)
    centre = count // 2
    quarter = max(1, count // 4)
    variants = [
        Variant(
            name="jerk_0_original",
            frames=frames.copy(),
            fps=meta.fps,
            expected_rank=4,
            parameters={"level": 0, "operation": "none"},
            note="re-encoded control sharing the encoder settings of every level",
        )
    ]
    level1 = frames.copy()
    level1[centre] = frames[centre - 1]
    variants.append(
        Variant(
            name="jerk_1_duplicate",
            frames=level1,
            fps=meta.fps,
            expected_rank=3,
            parameters={"level": 1, "operation": "duplicate_one", "index": centre},
            note="one frame repeated locally",
        )
    )
    level2 = frames.copy()
    level2[centre] = frames[centre - 1]
    level2[centre + 1] = frames[centre - 1]
    level2[centre + 2] = frames[centre + 3]
    level2[centre + 3] = frames[centre + 3]
    variants.append(
        Variant(
            name="jerk_2_duplicate_skip",
            frames=level2,
            fps=meta.fps,
            expected_rank=2,
            parameters={
                "level": 2,
                "operation": "duplicate_and_skip_two",
                "window": [centre - 1, centre + 3],
            },
            note="two duplicated frames and two skipped frames",
        )
    )
    level3 = frames.copy()
    start = max(0, centre - quarter // 2)
    stop = min(count, start + max(4, quarter))
    level3[start:stop] = frames[start:stop][::-1]
    variants.append(
        Variant(
            name="jerk_3_local_reverse",
            frames=level3,
            fps=meta.fps,
            expected_rank=1,
            parameters={"level": 3, "operation": "reverse_segment", "segment": [start, stop]},
            note="one short segment played backwards",
        )
    )
    level4 = frames.copy()
    segments = []
    for anchor in (count // 5, count // 2, (4 * count) // 5):
        seg_start = max(0, min(count - 4, anchor - 2))
        seg_stop = seg_start + 4
        level4[seg_start:seg_stop] = frames[seg_start:seg_stop][::-1]
        segments.append([seg_start, seg_stop])
    variants.append(
        Variant(
            name="jerk_4_multiple",
            frames=level4,
            fps=meta.fps,
            expected_rank=0,
            parameters={"level": 4, "operation": "multiple_reversals", "segments": segments},
            note="several local reversals spread across the clip",
        )
    )
    return variants


class ConversionTooShort(CounterfactualError):
    """Raised when a clip has too few frames for meaningful temporal surgery."""


# --------------------------------------------------------------------------
# Variant container
# --------------------------------------------------------------------------


@dataclass
class Variant:
    """One candidate derivation of a base clip."""

    name: str
    frames: np.ndarray | None
    fps: float
    expected_rank: int
    parameters: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    prompt_override: str | None = None
    relation_override: str | None = None

    @property
    def copies_bytes(self) -> bool:
        return self.frames is None
