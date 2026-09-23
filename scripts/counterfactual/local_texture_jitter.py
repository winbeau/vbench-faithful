"""Small, rapid LOCAL coordinate displacements, not additive RGB noise.

Each original frame is resampled exactly once at x + a(t)*u(x), where u is a
smooth bounded spatial field. The field has zero spatial mean and vanishes at
the image boundary; a(t) alternates rapidly and has zero temporal mean. There
is no cumulative warp, whole-frame translation, or replacement of source motion.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def displacement_field(height: int, width: int, *, scale: float, border: float, seed: int):
    if min(height, width) < 4 or not np.isfinite(scale) or scale <= 0 or not np.isfinite(border) or border <= 0:
        raise ValueError("positive spatial scale/border and at least 4x4 pixels required")
    rng = np.random.default_rng(seed)
    coarse = rng.standard_normal((max(3, int(np.ceil(height / scale)) + 1),
                                  max(3, int(np.ceil(width / scale)) + 1), 2)).astype(np.float32)
    smooth = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (0, 0), max(1.0, scale / 8))
    x, y = np.arange(width), np.arange(height)
    wx = (1 - np.cos(np.pi * np.clip(np.minimum(x, width - 1 - x) / border, 0, 1))) / 2
    wy = (1 - np.cos(np.pi * np.clip(np.minimum(y, height - 1 - y) / border, 0, 1))) / 2
    window = wy[:, None] * wx[None, :]
    mean = (smooth * window[..., None]).sum(axis=(0, 1)) / window.sum()
    field = (smooth - mean) * window[..., None]
    field /= np.linalg.norm(field, axis=-1).max()
    return field.astype(np.float32)


def alternating_phase(count: int, seed: int):
    if count < 2:
        raise ValueError("local jitter requires at least two source frames")
    phase = (-1.0) ** (np.arange(count) + seed % 2)
    phase -= phase.mean()
    return (phase / abs(phase).max()).astype(np.float32)


def local_texture_jitter(frames: np.ndarray, amplitude: float, seed: int, config: dict):
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("displacement amplitude must be finite and nonnegative")
    count, height, width, channels = frames.shape
    if channels != 3 or frames.dtype != np.uint8:
        raise ValueError("native uint8 RGB frames required")
    field = displacement_field(height, width, scale=config["spatial_scale_pixels"],
                               border=config["border_taper_pixels"], seed=seed) * amplitude
    phase = alternating_phase(count, seed)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    dx_y, dx_x = np.gradient(field[..., 0])
    dy_y, dy_x = np.gradient(field[..., 1])
    jacobian_min = float(min(np.min((1 + a * dx_x) * (1 + a * dy_y) - a * a * dx_y * dy_x)
                             for a in np.unique(phase)))
    output = []
    inside = True
    for frame, coefficient in zip(frames, phase):
        mx, my = xx + coefficient * field[..., 0], yy + coefficient * field[..., 1]
        inside &= bool(mx.min() >= 0 and my.min() >= 0 and mx.max() <= width - 1 and my.max() <= height - 1)
        output.append(cv2.remap(frame, mx, my, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101))
    result = np.stack(output)
    magnitudes = np.linalg.norm(field, axis=-1)
    temporal_step = abs(np.diff(phase))[:, None, None] * magnitudes[None]
    info = {
        "coordinate_map": "bounded_local_displacement_field", "intensity_noise_added": False,
        "interpolation": "opencv_linear", "field_convention": "output(x,y)=source(x+dx,y+dy)",
        "amplitude_units": "native_pixels", "max_displacement_pixels": float(magnitudes.max()),
        "mean_displacement_pixels": float(magnitudes.mean()),
        "spatial_mean_displacement_pixels": field.astype(float).mean(axis=(0, 1)).tolist(),
        "temporal_mean_displacement_max_pixels": float(abs(phase.mean()) * magnitudes.max()),
        "mean_interframe_jitter_step_pixels": float(temporal_step.mean()),
        "max_interframe_jitter_step_pixels": float(temporal_step.max()),
        "minimum_warp_jacobian": jacobian_min, "sampling_coordinates_in_bounds": inside,
        "edge_displacement_max_pixels": float(max(magnitudes[0].max(), magnitudes[-1].max(),
                                                   magnitudes[:, 0].max(), magnitudes[:, -1].max())),
        "temporal_phase": phase.tolist(), "clipped_fraction": 0.0,
        "preencode_difference": float(np.mean(abs(result.astype(np.float32) - frames))),
        "preencode_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "displacement_field_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
    }
    info["geometry_qualified"] = bool(inside and jacobian_min >= config["min_warp_jacobian"]
        and np.linalg.norm(info["spatial_mean_displacement_pixels"]) <= config["max_global_translation_pixels"]
        and info["max_displacement_pixels"] <= amplitude + 1e-5
        and info["temporal_mean_displacement_max_pixels"] <= 1e-5
        and info["edge_displacement_max_pixels"] == 0)
    return result, info, {"displacement_xy": field, "temporal_phase": phase}
