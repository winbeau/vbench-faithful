from __future__ import annotations

import math
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .schemas import MotionField, MotionSmoothnessConfig


def _motion_failed(backend: str, videos: list[Path], exc: Exception) -> list[dict[str, object]]:
    error = f"{type(exc).__name__}: {exc}"
    return [{"video": str(video), "backend": backend, "score": None, "status": "failed", "failure_reason": error, "error": error, "diagnostics": None} for video in videos]


def _motion_worker(result_path: str, gpu_id: int, videos: list[str], backend: str, weight: str, config: dict[str, object], seed: int = 42) -> None:
    paths = [Path(video) for video in videos]
    try:
        import random
        random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        if backend == "vbench":
            from .backends.vbench import evaluate_official_batch
            results = evaluate_official_batch(paths, device, config.get("config"), Path(weight))
        else:
            from .backends.audit import evaluate_timed_frames
            from .models import RaftFlowEstimator, decode_timed_frames
            from vbench_audit_core.upstream import resolve_upstream_path
            estimator = RaftFlowEstimator(device, Path(weight), Path(str(config.get("upstream") or resolve_upstream_path())))
            audit_config = MotionSmoothnessConfig(
                tail_quantile=float(config.get("tail_quantile", 0.9)),
                tail_weight=float(config.get("tail_weight", 0.25)),
                magnitude_weight=float(config.get("magnitude_weight", 0.5)),
                direction_weight=float(config.get("direction_weight", 0.5)),
                direction_alignment=bool(config.get("direction_alignment", False)),
                temporal_aggregation=str(config.get("temporal_aggregation", "topk")),
                top_k=int(config.get("top_k", 3)),
            )
            results = []
            for video in paths:
                try:
                    frames, _ = decode_timed_frames(video)
                    item = evaluate_timed_frames(video, frames, estimator, audit_config)
                    results.append({"video": str(video), "backend": "audit", "score": item.score, "status": "succeeded", "diagnostics": item.diagnostics})
                except Exception as exc:
                    results.extend(_motion_failed("audit", [video], exc))
    except Exception as exc:
        results = _motion_failed(backend, paths, exc)
    Path(result_path).write_text(json.dumps({"gpu_id": gpu_id, "results": results}, ensure_ascii=False), encoding="utf-8")


def evaluate_backend_sharded(backend: str, videos: list[Path], gpu_ids: list[int], weight: Path, config: MotionSmoothnessConfig | None = None, seed: int = 42) -> list[dict[str, object]]:
    from vbench_audit_core.coordinator import run_spawn_coordinator
    cfg = config or MotionSmoothnessConfig()
    return run_spawn_coordinator(
        _motion_worker, [str(video) for video in videos], gpu_ids,
        worker_args=(backend, str(weight), {
            "tail_quantile": cfg.tail_quantile,
            "tail_weight": cfg.tail_weight,
            "magnitude_weight": cfg.magnitude_weight,
            "direction_weight": cfg.direction_weight,
            "direction_alignment": cfg.direction_alignment,
            "temporal_aggregation": cfg.temporal_aggregation,
            "top_k": cfg.top_k,
        }, seed),
        backend=backend, label="motion-smoothness",
    ).results


def _array(flow: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError("flow must have shape (H, W, 2)")
    if not np.isfinite(value).all():
        raise ValueError("flow contains non-finite values")
    return value


def _valid_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    value = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if value.shape != shape:
        raise ValueError("valid_mask shape must match field")
    return value


def compute_velocity(flow: np.ndarray, dt: float) -> np.ndarray:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    return _array(flow) / float(dt)


def _bilinear_sample(
    field: np.ndarray,
    displacement: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``field(x + displacement(x))`` on the displacement grid.

    Flow components use pixel coordinates ``(dx, dy)`` with x increasing to
    the right and y downward. Coordinates are not clamped into validity:
    out-of-bounds samples are invalid, and a bilinear sample is valid only if
    every source pixel with non-zero interpolation weight is valid.
    """
    values = np.asarray(field, dtype=np.float64)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] not in (1, 2)):
        raise ValueError("field must have shape (H, W), (H, W, 1), or (H, W, 2)")
    if not np.isfinite(values).all():
        raise ValueError("field contains non-finite values")
    delta = _array(displacement)
    height, width = delta.shape[:2]
    if values.shape[:2] != (height, width):
        raise ValueError("field and displacement shapes must match")
    source_valid = _valid_mask(valid_mask, (height, width))

    yy, xx = np.indices((height, width), dtype=np.float64)
    sample_x = xx + delta[..., 0]
    sample_y = yy + delta[..., 1]
    inside = (
        (sample_x >= 0.0)
        & (sample_x <= width - 1)
        & (sample_y >= 0.0)
        & (sample_y <= height - 1)
    )

    x0 = np.floor(sample_x).astype(np.int64)
    y0 = np.floor(sample_y).astype(np.int64)
    x0_safe = np.clip(x0, 0, width - 1)
    y0_safe = np.clip(y0, 0, height - 1)
    x1_safe = np.clip(x0 + 1, 0, width - 1)
    y1_safe = np.clip(y0 + 1, 0, height - 1)
    wx = sample_x - x0
    wy = sample_y - y0
    weights = (
        (1.0 - wx) * (1.0 - wy),
        wx * (1.0 - wy),
        (1.0 - wx) * wy,
        wx * wy,
    )
    coordinates = (
        (y0_safe, x0_safe),
        (y0_safe, x1_safe),
        (y1_safe, x0_safe),
        (y1_safe, x1_safe),
    )

    sampled = np.zeros_like(values, dtype=np.float64)
    valid = inside.copy()
    for weight, (sample_row, sample_col) in zip(weights, coordinates):
        corner = values[sample_row, sample_col]
        sampled += weight[..., None] * corner if values.ndim == 3 else weight * corner
        valid &= (weight <= np.finfo(np.float64).eps) | source_valid[sample_row, sample_col]
    sampled[~valid] = 0.0
    return sampled, valid


def align_motion_field(
    field: np.ndarray,
    displacement: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly evaluate a current field at points advected by prior flow."""
    aligned, valid = _bilinear_sample(_array(field), displacement, valid_mask)
    return _array(aligned), valid


def _aligned_difference(
    previous: np.ndarray,
    current: np.ndarray,
    displacement: np.ndarray | None = None,
    previous_mask: np.ndarray | None = None,
    current_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    previous_value = _array(previous)
    previous_valid = _valid_mask(previous_mask, previous_value.shape[:2])
    if displacement is None:
        current_aligned = _array(current)
        current_valid = _valid_mask(current_mask, previous_value.shape[:2])
    else:
        current_aligned, current_valid = align_motion_field(current, displacement, current_mask)
    return current_aligned - previous_value, previous_valid & current_valid


def compute_acceleration(
    velocities: Iterable[np.ndarray],
    dts: Iterable[float],
    displacements: Iterable[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Diagnostic physical acceleration; it is not used by the final score."""
    values = [_array(value) for value in velocities]
    intervals = [float(value) for value in dts]
    if len(values) != len(intervals):
        raise ValueError("velocities and dts must have equal length")
    if len(values) < 2:
        return [], []
    disp = list(displacements) if displacements is not None else [None] * (len(values) - 1)
    if len(disp) != len(values) - 1:
        raise ValueError("displacements must have one entry per velocity transition")
    acceleration: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in range(1, len(values)):
        dt = intervals[index]
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("derivative dt must be finite and positive")
        difference, mask = _aligned_difference(values[index - 1], values[index], disp[index - 1])
        acceleration.append(difference / dt)
        masks.append(mask)
    return acceleration, masks


def compute_jerk(
    accelerations: Iterable[np.ndarray],
    dts: Iterable[float],
    displacements: Iterable[np.ndarray] | None = None,
    valid_masks: Iterable[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Diagnostic physical jerk; it is not used by the final score."""
    values = [_array(value) for value in accelerations]
    intervals = [float(value) for value in dts]
    if len(values) != len(intervals):
        raise ValueError("accelerations and dts must have equal length")
    if len(values) < 2:
        return [], []
    disp = list(displacements) if displacements is not None else [None] * (len(values) - 1)
    if len(disp) != len(values) - 1:
        raise ValueError("displacements must have one entry per acceleration transition")
    masks = list(valid_masks) if valid_masks is not None else [None] * len(values)
    if len(masks) != len(values):
        raise ValueError("valid_masks must align with accelerations")
    jerk: list[np.ndarray] = []
    jerk_masks: list[np.ndarray] = []
    for index in range(1, len(values)):
        dt = intervals[index]
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("derivative dt must be finite and positive")
        difference, mask = _aligned_difference(
            values[index - 1], values[index], disp[index - 1], masks[index - 1], masks[index]
        )
        jerk.append(difference / dt)
        jerk_masks.append(mask)
    return jerk, jerk_masks


def _spatial_stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    boolean_mask = np.asarray(mask, dtype=bool)
    selected = np.asarray(values, dtype=np.float64)[boolean_mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(selected)),
        float(np.median(selected)),
        float(np.quantile(selected, 0.90)),
        float(selected.size / boolean_mask.size),
    )


def aggregate_temporal_discontinuity(
    values: Iterable[float], config: MotionSmoothnessConfig
) -> tuple[float, float, float]:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    if not np.isfinite(data).all() or np.any(data < 0.0):
        raise ValueError("temporal discontinuities must be finite and non-negative")
    mean = float(np.mean(data))
    if config.temporal_aggregation == "topk":
        # A temporal edit touches only a few transitions, so the global mean
        # dilutes it.  The mean of the k largest per-time discontinuities is the
        # localized-failure statistic plan 13.4 asks for; the number of injected
        # events is bounded, so k does not scale with clip length.
        count = min(config.top_k, data.size)
        peak = float(np.mean(np.sort(data)[-count:]))
        return mean, peak, peak
    tail_count = max(1, math.ceil((1.0 - config.tail_quantile) * data.size))
    tail = float(np.mean(np.sort(data)[-tail_count:]))
    combined = (1.0 - config.tail_weight) * mean + config.tail_weight * tail
    return mean, tail, float(combined)


def discontinuity_to_score(discontinuity: float) -> float:
    value = float(discontinuity)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("D_video must be finite and non-negative")
    score = math.exp(-value)
    if not math.isfinite(score):
        raise ValueError("motion smoothness score must be finite")
    return score


def analyze_motion_fields(
    fields: Iterable[MotionField], config: MotionSmoothnessConfig | None = None
) -> dict[str, object]:
    cfg = config or MotionSmoothnessConfig()
    motion_fields = list(fields)
    if not motion_fields:
        return {"score": 1.0, "D_video": 0.0, "diagnostics": {"status": "no_motion_fields", "D_t": []}}

    flows = [_array(item.flow) for item in motion_fields]
    if any(flow.shape != flows[0].shape for flow in flows[1:]):
        raise ValueError("all motion fields must have the same shape")
    dts = [float(item.dt) for item in motion_fields]
    velocities = [compute_velocity(flow, dt) for flow, dt in zip(flows, dts)]
    field_masks = [_valid_mask(item.valid_mask, flow.shape[:2]) for item, flow in zip(motion_fields, flows)]
    magnitudes = [np.linalg.norm(value, axis=-1) for value in velocities]
    mean_motion = float(
        np.mean(
            [np.median(value[mask]) if mask.any() else 0.0 for value, mask in zip(magnitudes, field_masks)]
        )
    )
    if len(velocities) < 3:
        return {
            "score": 1.0,
            "D_video": 0.0,
            "diagnostics": {
                "status": "insufficient_motion_states",
                "D_t": [],
                "mean_motion_magnitude": mean_motion,
            },
        }

    magnitude_t: list[float] = []
    direction_change_t: list[float] = []
    aligned_direction_t: list[float] = []
    valid_ratios: list[float] = []
    spatial_medians: list[float] = []
    spatial_p90s: list[float] = []
    for index in range(1, len(velocities)):
        current_aligned, aligned_mask = align_motion_field(
            velocities[index], flows[index - 1], field_masks[index]
        )
        mask = field_masks[index - 1] & aligned_mask
        previous = velocities[index - 1]
        previous_magnitude = np.linalg.norm(previous, axis=-1)
        current_magnitude = np.linalg.norm(current_aligned, axis=-1)
        relative = np.linalg.norm(current_aligned - previous, axis=-1) / (
            current_magnitude + previous_magnitude + cfg.eps
        )
        relative[~mask] = 0.0
        mean, median, p90, valid_ratio = _spatial_stats(relative, mask)
        magnitude_t.append(mean)
        spatial_medians.append(median)
        spatial_p90s.append(p90)
        valid_ratios.append(valid_ratio)

        direction_mask = mask & (previous_magnitude >= cfg.min_motion_magnitude) & (
            current_magnitude >= cfg.min_motion_magnitude
        )
        denominator = previous_magnitude * current_magnitude
        aligned_cosine = np.divide(
            np.sum(previous * current_aligned, axis=-1),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > 0,
        )
        aligned_direction = _spatial_stats(
            np.clip(1.0 - aligned_cosine, 0.0, 2.0), direction_mask
        )[0]

        # Raw (pixel-wise) direction: the alignment warp follows a moving
        # subject, but at a frame reversal it samples the reversed field at the
        # pre-reversal location and can cancel the flip.  The unaligned angle is
        # the direct measure of a temporal-order violation and is the default;
        # the aligned value stays available as a diagnostic and an ablation.
        raw = velocities[index]
        raw_magnitude = np.linalg.norm(raw, axis=-1)
        raw_mask = mask & (previous_magnitude >= cfg.min_motion_magnitude) & (
            raw_magnitude >= cfg.min_motion_magnitude
        )
        raw_denominator = previous_magnitude * raw_magnitude
        raw_cosine = np.divide(
            np.sum(previous * raw, axis=-1),
            raw_denominator,
            out=np.ones_like(raw_denominator),
            where=raw_denominator > 0,
        )
        raw_direction = _spatial_stats(np.clip(1.0 - raw_cosine, 0.0, 2.0), raw_mask)[0]
        aligned_direction_t.append(aligned_direction)
        direction_change_t.append(aligned_direction if cfg.direction_alignment else raw_direction)

    # Full variant (plan 13.5): per-time discontinuity is the weighted sum of the
    # normalised magnitude change (`relative`, i.e. flow acceleration) and the
    # direction change.  The estimator before this one aggregated a temporal
    # *difference* of the already-saturated magnitude change and never read
    # `direction_change_t`, so a local reversal and a hold-and-jump of similar
    # length produced the same D_video and the declared level-2 > level-3
    # ordering was not realisable.  The direction term separates an actual
    # reversal from a stall, and the top-k aggregation keeps a localized edit
    # from being diluted by the ~15-30 transitions it sits among.
    discontinuities = [
        cfg.magnitude_weight * magnitude + cfg.direction_weight * direction
        for magnitude, direction in zip(magnitude_t, direction_change_t)
    ]
    d_mean, d_tail, d_video = aggregate_temporal_discontinuity(discontinuities, cfg)
    score = discontinuity_to_score(d_video)
    diagnostics = {
        "status": "succeeded",
        "num_motion_fields": len(motion_fields),
        "median_dt": float(np.median(dts)),
        "min_dt": float(np.min(dts)),
        "max_dt": float(np.max(dts)),
        "mean_motion_magnitude": mean_motion,
        "valid_pixel_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "mean_discontinuity": d_mean,
        "tail_discontinuity": d_tail,
        "magnitude_discontinuity": float(np.mean(magnitude_t)) if magnitude_t else 0.0,
        "direction_discontinuity": float(np.mean(direction_change_t)) if direction_change_t else 0.0,
        "aligned_direction_discontinuity": (
            float(np.mean(aligned_direction_t)) if aligned_direction_t else 0.0
        ),
        "D_video": d_video,
        "score": score,
        "D_t": discontinuities,
        "state_change_t": magnitude_t,
        "direction_change_t": direction_change_t,
        "aligned_direction_t": aligned_direction_t,
        "spatial_median_t": spatial_medians,
        "spatial_p90_t": spatial_p90s,
        "config": asdict(cfg),
    }
    return {"score": score, "D_video": d_video, "diagnostics": diagnostics}
