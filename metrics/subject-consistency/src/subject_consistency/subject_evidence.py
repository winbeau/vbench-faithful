"""Subject-localised evidence for the subject-consistency repair.

This module is pure algorithm: it turns per-frame patch features plus per-frame
subject masks into one subject vector per frame and reduces those vectors with
the position-invariant all-pairs rule. The shipped aggregation repair also
mixes in adjacent pairs; this representation variant uses all-pairs alone.
Nothing here loads a model, reads a video, or knows which
localizer produced the masks, so the score can be unit-tested without weights.

Design decisions the contract experiments depend on:

- ``missing_policy="zero"`` is the conservative default: a frame without subject
  evidence contributes a zero similarity to every pair it belongs to, and the
  denominator stays ``C(T, 2)``. Excluding such frames instead would let subject
  corruption that destroys detectability *raise* the score, which is the failure
  mode this repair must not have. ``exclude`` stays available as the
  detection-conditioned bound, and both numbers must be reported.
- ``instance_mode`` fixes what "the subject" means when the phrase matches more
  than one instance: ``union`` pools the union of the matched regions into one
  vector per frame, ``mean`` pools each instance and averages the vectors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

MISSING_POLICIES = ("zero", "exclude", "carry")
INSTANCE_MODES = ("union", "mean")


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class SubjectEvidenceDiagnostics:
    num_frames: int
    num_present_frames: int
    num_missing_frames: int
    missing_fraction: float
    instance_mode: str
    missing_policy: str
    pair_count: int
    pair_denominator: int
    score: float
    adjacent_score: float
    pair_min: float
    pair_max: float
    coverage: tuple[float, ...]

    def to_dict(self, *, include_coverage: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_coverage:
            payload.pop("coverage")
        return payload


def sample_frame_indices(total: int, max_frames: int | None = None) -> tuple[int, ...]:
    """Every frame by default; a capped clip uses VBench's interval midpoints.

    VBench itself consumes every decoded frame, so the cap is a declared
    deviation rather than a silent one; it is recorded in diagnostics.
    """
    total = int(total)
    if total <= 0:
        raise ValueError("total must be positive")
    if max_frames is None:
        return tuple(range(total))
    max_frames = int(max_frames)
    if max_frames < 2:
        raise ValueError("max_frames must be at least two")
    if max_frames >= total:
        return tuple(range(total))
    intervals = [int(index * total / max_frames) for index in range(max_frames + 1)]
    return tuple((intervals[index] + intervals[index + 1] - 1) // 2 for index in range(max_frames))


def resample_masks_to_grid(masks: Any, grid_height: int, grid_width: int, *,
                           image_size: tuple[int, int] | None = None, patch_size: int = 16) -> Any:
    """[T, h, w] coverage in [0, 1] -> [T, gh*gw] box-averaged coverage."""
    torch = _torch()
    grid_height, grid_width = int(grid_height), int(grid_width)
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("grid dimensions must be positive")
    if getattr(masks, "ndim", None) != 3:
        raise ValueError("masks must have shape [T, h, w]")
    values = masks.to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("mask coverage must lie in [0, 1]")
    if image_size is None:
        pooled = torch.nn.functional.adaptive_avg_pool2d(values.unsqueeze(1), (grid_height, grid_width))
    else:
        # DINO's stride-16 convolution drops a non-divisible right/bottom tail.
        # Pooling the entire original image directly into gh*gw would stretch
        # that tail onto represented patches and shift the subject boundary.
        height, width = image_size
        if (height // patch_size, width // patch_size) != (grid_height, grid_width):
            raise ValueError("resized image and patch grid disagree")
        resized = torch.nn.functional.interpolate(values.unsqueeze(1), size=image_size, mode="area")
        pooled = torch.nn.functional.avg_pool2d(resized, patch_size, stride=patch_size)
    return pooled.flatten(1)


def _pool(features: Any, weights: Any) -> tuple[Any, Any]:
    """Weighted mean of per-frame patch features (L2-normalised) and its mass."""
    torch = _torch()
    mass = weights.sum(dim=-1)
    safe = mass.clamp_min(torch.finfo(weights.dtype).eps)
    pooled = torch.einsum("tn,tnd->td", weights, features) / safe.unsqueeze(-1)
    normalized = torch.nn.functional.normalize(pooled, dim=-1, p=2)
    empty = mass <= 0
    normalized = torch.where(empty.unsqueeze(-1), torch.zeros_like(normalized), normalized)
    return normalized, mass / weights.shape[-1]


def subject_vectors(
    patch_features: Any,
    instance_masks: Any,
    instance_present: Any | None = None,
    *,
    mode: str = "union",
) -> tuple[Any, Any]:
    """[T, N, D] features + [T, I, N] masks -> [T, D] vectors and [T] coverage.

    ``instance_present`` is [T, I] and defaults to every instance in every
    frame. A frame whose usable mask mass is zero yields a zero vector and
    coverage zero; it is not silently dropped here.
    """
    torch = _torch()
    if mode not in INSTANCE_MODES:
        raise ValueError(f"unknown instance mode: {mode}")
    if getattr(patch_features, "ndim", None) != 3:
        raise ValueError("patch_features must have shape [T, N, D]")
    if getattr(instance_masks, "ndim", None) != 3:
        raise ValueError("instance_masks must have shape [T, I, N]")
    frames, tokens = int(patch_features.shape[0]), int(patch_features.shape[1])
    if int(instance_masks.shape[0]) != frames or int(instance_masks.shape[2]) != tokens:
        raise ValueError("instance_masks must share frame and token counts with patch_features")
    instances = int(instance_masks.shape[1])
    if instances < 1:
        raise ValueError("at least one instance slot is required")
    weights = instance_masks.to(device=patch_features.device, dtype=patch_features.dtype)
    if not bool(torch.isfinite(patch_features).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("features and masks must be finite")
    if bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("mask coverage must lie in [0, 1]")
    if instance_present is None:
        present = torch.ones((frames, instances), dtype=torch.bool, device=weights.device)
    else:
        present = instance_present.to(device=weights.device, dtype=torch.bool)
        if tuple(present.shape) != (frames, instances):
            raise ValueError("instance_present must have shape [T, I]")

    if mode == "union":
        masked = torch.where(present.unsqueeze(-1), weights, torch.zeros_like(weights))
        return _pool(patch_features, masked.amax(dim=1))

    per_vector = []
    per_mass = []
    for index in range(instances):
        vector, mass = _pool(patch_features, weights[:, index, :])
        per_vector.append(vector)
        per_mass.append(mass)
    stacked = torch.stack(per_vector, dim=1)
    masses = torch.stack(per_mass, dim=1)
    present_f = present.unsqueeze(-1)
    counts = present_f.sum(dim=1).clamp_min(1)
    mean_vector = torch.where(present_f, stacked, torch.zeros_like(stacked)).sum(dim=1) / counts
    normalized = torch.nn.functional.normalize(mean_vector, dim=-1, p=2)
    any_present = present.any(dim=1)
    normalized = torch.where(any_present.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    coverage = torch.where(present, masses, torch.zeros_like(masses)).sum(dim=1) / counts.squeeze(-1)
    return normalized, coverage


def _fill_missing(vectors: Any, present: Any) -> tuple[Any, Any]:
    """Forward then backward fill; returns the filled vectors and a known mask."""
    filled = vectors.clone()
    known = present.clone()
    last = None
    for index in range(int(known.shape[0])):
        if bool(known[index]):
            last = filled[index]
        elif last is not None:
            filled[index] = last
            known[index] = True
    following = None
    for index in range(int(known.shape[0]) - 1, -1, -1):
        if bool(known[index]):
            following = filled[index]
        elif following is not None:
            filled[index] = following
            known[index] = True
    return filled, known


def subject_consistency_from_vectors(
    vectors: Any,
    present: Any | None = None,
    *,
    coverage: Sequence[float] | None = None,
    missing_policy: str = "zero",
    clamp_negative: bool = True,
    instance_mode: str = "union",
) -> SubjectEvidenceDiagnostics:
    """All-pairs cosine reduction over per-frame subject vectors."""
    torch = _torch()
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"unknown missing policy: {missing_policy}")
    if instance_mode not in INSTANCE_MODES:
        raise ValueError(f"unknown instance mode: {instance_mode}")
    if getattr(vectors, "ndim", None) != 2:
        raise ValueError("vectors must have shape [T, D]")
    frames = int(vectors.shape[0])
    if frames < 2:
        raise ValueError("subject evidence requires at least two frames")
    if present is None:
        present = torch.ones(frames, dtype=torch.bool, device=vectors.device)
    present = present.to(device=vectors.device, dtype=torch.bool)
    if tuple(present.shape) != (frames,):
        raise ValueError("present must have one entry per frame")

    if int(present.sum()) < 2 and missing_policy == "exclude":
        raise ValueError("fewer than two frames carry subject evidence")
    if missing_policy == "carry":
        effective, known = _fill_missing(vectors, present)
        if not bool(known.all()):
            raise ValueError("no frame carries subject evidence")
    else:
        effective = vectors

    similarity = effective @ effective.transpose(0, 1)
    if clamp_negative:
        similarity = similarity.clamp_min(0.0)

    upper = torch.triu_indices(frames, frames, offset=1, device=similarity.device)
    adjacent_mask = (upper[1] - upper[0]) == 1
    if missing_policy == "exclude":
        keep = present[upper[0]] & present[upper[1]]
        if int(keep.sum()) < 1:
            raise ValueError("fewer than two frames carry subject evidence")
        scored = similarity[upper[0][keep], upper[1][keep]]
        adjacent_keep = keep & adjacent_mask
        adjacent_scored = similarity[upper[0][adjacent_keep], upper[1][adjacent_keep]]
    else:
        values = similarity[upper[0], upper[1]]
        if missing_policy == "zero":
            keep = present[upper[0]] & present[upper[1]]
            values = torch.where(keep, values, torch.zeros_like(values))
        scored = values
        adjacent_scored = values[adjacent_mask]

    if coverage is None:
        coverage_values = tuple(1.0 if bool(flag) else 0.0 for flag in present.detach().cpu().tolist())
    else:
        coverage_values = tuple(float(value) for value in coverage)
        if len(coverage_values) != frames:
            raise ValueError("coverage must have one entry per frame")
    missing = frames - int(present.sum())
    return SubjectEvidenceDiagnostics(
        num_frames=frames,
        num_present_frames=int(present.sum()),
        num_missing_frames=missing,
        missing_fraction=missing / frames,
        instance_mode=instance_mode,
        missing_policy=missing_policy,
        pair_count=int(scored.numel()),
        pair_denominator=frames * (frames - 1) // 2,
        score=float(scored.mean().item()),
        adjacent_score=float(adjacent_scored.mean().item()) if int(adjacent_scored.numel()) else 0.0,
        pair_min=float(scored.min().item()),
        pair_max=float(scored.max().item()),
        coverage=coverage_values,
    )


def masked_subject_consistency(
    patch_features: Any,
    instance_masks: Any,
    instance_present: Any | None = None,
    *,
    mode: str = "union",
    missing_policy: str = "zero",
    clamp_negative: bool = True,
) -> SubjectEvidenceDiagnostics:
    """Convenience wrapper: patch features + masks -> diagnostics."""
    vectors, coverage = subject_vectors(patch_features, instance_masks, instance_present, mode=mode)
    present = coverage > 0
    return subject_consistency_from_vectors(
        vectors,
        present,
        coverage=tuple(float(value) for value in coverage.detach().cpu().tolist()),
        missing_policy=missing_policy,
        clamp_negative=clamp_negative,
        instance_mode=mode,
    )
