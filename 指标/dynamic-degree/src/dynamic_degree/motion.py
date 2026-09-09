from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

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
