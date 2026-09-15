from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Sequence

import cv2
import numpy as np

from .schemas import AffineFitDiagnostics, MotionDecomposition, MotionStatistics


@dataclass(frozen=True)
class AffineEstimatorConfig:
    max_correspondences: int = 20000
    min_background_candidates: int = 12
    ransac_reprojection_threshold: float = 2.0
    residual_mad_scale: float = 3.0
    border_fraction: float = 0.25


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError("flow must have shape (H, W, 2)")
    if not np.isfinite(array).all():
        raise ValueError("flow contains non-finite values")
    return np.sqrt(np.square(array[..., 0]) + np.square(array[..., 1]))


def top_fraction_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("spatial motion aggregation requires at least one pixel")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(flat.size * fraction))
    selected = np.partition(flat, flat.size - count)[-count:]
    return float(selected.mean())


def normalized_intensity(
    displacement_pixels: float,
    frame_shape: tuple[int, int],
    dt_seconds: float,
    exponent: float,
) -> tuple[float, float]:
    """Displacement in diagonals, and it divided by ``dt ** exponent``.

    ``exponent=1`` is the ballistic normalisation (the ``*_speed`` fields) and is
    exact only for constant-velocity motion.  ``exponent=0`` keeps the raw
    displacement.  Any other value removes exactly the lag dependence measured
    for this clip, so the reported intensity separates motion amplitude from the
    sampling interval that the FPS ladder varies.
    """
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    if not np.isfinite(exponent):
        raise ValueError("time-normalization exponent must be finite")
    height, width = frame_shape
    diagonal = float(np.hypot(height, width))
    if diagonal <= 0:
        raise ValueError("frame diagonal must be positive")
    displacement = float(displacement_pixels) / diagonal
    return displacement, displacement / (float(dt_seconds) ** float(exponent))


def scale_intensity(displacement_diagonals: float, dt_seconds: float, exponent: float) -> float:
    """Divide an already diagonal-normalised displacement by ``dt ** exponent``."""
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    if not np.isfinite(exponent):
        raise ValueError("time-normalization exponent must be finite")
    return float(displacement_diagonals) / (float(dt_seconds) ** float(exponent))


def resolve_lags(requested: Sequence[int], frame_count: int) -> tuple[int, ...]:
    """The frame lags to measure: the requested ones the clip supports, plus the
    longest lag it can support (so a short clip still spans more than one lag)."""
    maximum = frame_count - 1
    available: set[int] = set()
    for lag in requested:
        if int(lag) < 1:
            raise ValueError("sampling lags are counted in frames and must be >= 1")
        if int(lag) <= maximum:
            available.add(int(lag))
    cap = min(max(int(lag) for lag in requested), maximum)
    if cap >= 2:
        available.add(cap)
    return tuple(sorted(available))


def fit_power_law_exponent(
    lag_seconds: Sequence[float],
    displacements: Sequence[float],
    *,
    minimum_points: int = 2,
    weights: Sequence[float] | None = None,
) -> tuple[float, float] | None:
    """Least-squares slope of ``log(displacement)`` on ``log(lag)``.

    Returns ``(exponent, rmse_of_the_log_fit)``, or ``None`` when the clip does
    not supply at least ``minimum_points`` distinct positive lag/displacement
    pairs.  ``exponent=1`` means ballistic motion, ``0`` lag-independent motion.
    ``weights`` lets a caller down-weight a lag that rests on very few pairs.
    """
    points: list[tuple[float, float, float]] = []
    for position, (lag, displacement) in enumerate(zip(lag_seconds, displacements)):
        lag_value, displacement_value = float(lag), float(displacement)
        if not np.isfinite(lag_value) or not np.isfinite(displacement_value):
            continue
        if lag_value <= 0 or displacement_value <= 0:
            continue
        weight = 1.0 if weights is None else float(weights[position])
        if not np.isfinite(weight) or weight <= 0:
            continue
        points.append((lag_value, displacement_value, weight))
    if len(points) < minimum_points or len({round(lag, 9) for lag, _, _ in points}) < minimum_points:
        return None
    log_lag = np.log(np.array([lag for lag, _, _ in points], dtype=np.float64))
    log_displacement = np.log(np.array([value for _, value, _ in points], dtype=np.float64))
    weight_array = np.array([weight for _, _, weight in points], dtype=np.float64)
    slope, intercept = np.polyfit(log_lag, log_displacement, 1, w=weight_array)
    residual = log_displacement - (slope * log_lag + intercept)
    rmse = float(np.sqrt(np.average(np.square(residual), weights=np.square(weight_array))))
    if not np.isfinite(float(slope)) or not np.isfinite(rmse):
        return None
    return float(slope), rmse


def motion_statistics(flow: np.ndarray) -> MotionStatistics:
    magnitude = flow_magnitude(flow)
    return MotionStatistics(
        top5_mean=top_fraction_mean(magnitude),
        mean=float(magnitude.mean()),
        median=float(np.median(magnitude)),
        maximum=float(magnitude.max()),
    )


def affine_flow(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    global_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2] - xx
    global_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2] - yy
    return np.stack((global_x, global_y), axis=-1).astype(np.float64)


def _sample_correspondences(flow: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    step = max(1, int(ceil(sqrt((height * width) / maximum))))
    yy, xx = np.mgrid[0:height:step, 0:width:step]
    src = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1).astype(np.float32)
    sampled_flow = flow[yy, xx].reshape(-1, 2).astype(np.float32)
    dst = src + sampled_flow
    return src, dst, xx.reshape(-1), yy.reshape(-1)


def _fit_partial_affine(src: np.ndarray, dst: np.ndarray, config: AffineEstimatorConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(src) < 3:
        return None, None
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reprojection_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    return matrix, inliers


def decompose_motion(
    apparent_flow: np.ndarray,
    config: AffineEstimatorConfig | None = None,
    *,
    compensate_global: bool = True,
) -> MotionDecomposition:
    config = config or AffineEstimatorConfig()
    flow = np.asarray(apparent_flow, dtype=np.float64)
    flow_magnitude(flow)
    height, width = flow.shape[:2]
    zero_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    if not compensate_global:
        zero_flow = np.zeros_like(flow)
        diagnostics = AffineFitDiagnostics(
            affine_parameters=tuple(tuple(float(value) for value in row) for row in zero_matrix),
            inlier_ratio=0.0,
            candidate_count=0,
            initial_candidate_count=0,
            candidate_strategy="global_compensation_disabled",
            fallback=True,
            fallback_reason="global_compensation_disabled",
            residual_median=float(np.median(flow_magnitude(flow))),
            residual_mad=0.0,
        )
        return MotionDecomposition(
            flow, zero_flow, flow.copy(), motion_statistics(flow), motion_statistics(zero_flow), motion_statistics(flow), diagnostics
        )

    src, dst, sampled_x, sampled_y = _sample_correspondences(flow, config.max_correspondences)
    initial_matrix, initial_inliers = _fit_partial_affine(src, dst, config)
    initial_count = len(src)
    if initial_matrix is None:
        global_flow = np.zeros_like(flow)
        residual_flow = flow.copy()
        diagnostics = AffineFitDiagnostics(
            affine_parameters=tuple(tuple(float(value) for value in row) for row in zero_matrix),
            inlier_ratio=0.0,
            candidate_count=0,
            initial_candidate_count=initial_count,
            candidate_strategy="initial_fit_failed",
            fallback=True,
            fallback_reason="initial_fit_failed_no_compensation",
            residual_median=float(np.median(flow_magnitude(residual_flow))),
            residual_mad=0.0,
        )
        return MotionDecomposition(
            flow, global_flow, residual_flow, motion_statistics(flow), motion_statistics(global_flow), motion_statistics(residual_flow), diagnostics
        )

    predicted = cv2.transform(src[:, None, :], initial_matrix).reshape(-1, 2)
    residual = np.linalg.norm(dst - predicted, axis=1)
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    robust_scale = max(1e-6, 1.4826 * residual_mad)
    residual_consistent = residual <= residual_median + config.residual_mad_scale * robust_scale
    geometric_inlier = initial_inliers.reshape(-1).astype(bool) if initial_inliers is not None else np.ones(len(src), dtype=bool)
    border_x = max(1.0, width * config.border_fraction)
    border_y = max(1.0, height * config.border_fraction)
    # A deterministic spatial prior only; this is not a semantic background mask.
    # OR selects the complete image-border band. AND would select only four corners.
    border_band_prior = (
        (sampled_x < border_x)
        | (sampled_x >= width - border_x)
        | (sampled_y < border_y)
        | (sampled_y >= height - border_y)
    )
    consistent_inliers = geometric_inlier & residual_consistent
    candidates = consistent_inliers & border_band_prior

    fallback = False
    fallback_reason = None
    candidate_strategy = "geometric_inlier_residual_consistent_border_band_prior"
    if int(candidates.sum()) < config.min_background_candidates:
        candidates = consistent_inliers
        candidate_strategy = "geometric_inlier_residual_consistent_without_spatial_prior"
        fallback = True
        fallback_reason = "insufficient_border_band_candidates"

    final_matrix = initial_matrix
    candidate_count = int(candidates.sum())
    if candidate_count >= config.min_background_candidates:
        refit_matrix, _ = _fit_partial_affine(src[candidates], dst[candidates], config)
        if refit_matrix is not None:
            final_matrix = refit_matrix
        else:
            fallback = True
            fallback_reason = (
                f"{fallback_reason}; refit_failed_using_initial"
                if fallback_reason
                else "refit_failed_using_initial"
            )
    else:
        fallback = True
        fallback_reason = (
            f"{fallback_reason}; insufficient_consistent_candidates_using_initial"
            if fallback_reason
            else "insufficient_consistent_candidates_using_initial"
        )

    final_predicted = cv2.transform(src[:, None, :], final_matrix).reshape(-1, 2)
    final_residual = np.linalg.norm(dst - final_predicted, axis=1)
    final_residual_median = float(np.median(final_residual))
    final_residual_mad = float(np.median(np.abs(final_residual - final_residual_median)))
    final_inlier_ratio = float((final_residual <= config.ransac_reprojection_threshold).mean())
    global_flow = affine_flow(final_matrix, height, width)
    residual_flow = flow - global_flow
    diagnostics = AffineFitDiagnostics(
        affine_parameters=tuple(tuple(float(value) for value in row) for row in final_matrix),
        inlier_ratio=final_inlier_ratio,
        candidate_count=candidate_count,
        initial_candidate_count=initial_count,
        candidate_strategy=candidate_strategy,
        fallback=fallback,
        fallback_reason=fallback_reason,
        residual_median=final_residual_median,
        residual_mad=final_residual_mad,
    )
    return MotionDecomposition(
        apparent_flow=flow,
        global_flow=global_flow,
        residual_flow=residual_flow,
        apparent_statistics=motion_statistics(flow),
        global_statistics=motion_statistics(global_flow),
        residual_statistics=motion_statistics(residual_flow),
        affine=diagnostics,
    )
