"""DEV hypothesis: video-inferred reversing modes with spatial safeguards.

This module does not certify tracking or infer physical causes. It returns
trajectory diagnostics, not a valid score. A coherent temporal component alone
is NOT evidence of nuisance: affine/camera motion, same-direction parallax,
localized motion and locally coordinated parts are protected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .structural_decomposition import DecompositionConfig, smooth_trajectories, spatial_support


@dataclass(frozen=True)
class CommonModeConfig:
    temporal_scale_seconds: float = .20
    minimum_reversal_fraction: float = .5
    minimum_mode_energy_fraction: float = .1
    minimum_participation_fraction: float = .1
    minimum_evidence_fraction: float = .6
    structure_retention_threshold: float = .8
    local_supported_energy_threshold: float = .5
    minimum_mode_separation: float = .05
    spatial_radius_fraction: float = .2
    minimum_neighbors: int = 4
    coherence_error_scale: float = .25

    def __post_init__(self):
        for value in (self.temporal_scale_seconds, self.spatial_radius_fraction,
                      self.coherence_error_scale):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("positive finite scales required")
        for value in (self.minimum_reversal_fraction, self.minimum_mode_energy_fraction,
                      self.minimum_participation_fraction, self.minimum_evidence_fraction,
                      self.structure_retention_threshold, self.local_supported_energy_threshold,
                      self.minimum_mode_separation):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("fractions must be finite in (0,1]")
        if not isinstance(self.minimum_neighbors, int) or self.minimum_neighbors < 4:
            raise ValueError("at least four neighbors required")


def decompose_common_modes(xy, timestamps, short_side, config, *, colors=None, reliable=None):
    """Separate residual modes using ALL timestamps, without a frequency notch.

    SVD uses the time integral of squared velocity. Mode signs cancel in outer
    products; nearly degenerate modes are retained rather than making a decision
    from an arbitrary SVD basis. Reliability is an additional necessary condition,
    never evidence that a mode is nuisance. Missing tracks remain uncertified.
    """
    xy = np.asarray(xy, float)
    times = np.asarray(timestamps, float)
    if not np.isfinite(short_side) or short_side <= 0:
        raise ValueError("positive short side required")
    smooth = smooth_trajectories(xy, times, config.temporal_scale_seconds)
    raw = np.diff(xy, axis=0)
    trend = np.diff(smooth, axis=0)
    residual = raw - trend
    if raw.shape[1] == 0:
        raise ValueError("at least one trajectory required")
    if reliable is None:
        reliable = np.ones(raw.shape[:2], bool)
    reliable = np.asarray(reliable, bool)
    if reliable.shape != raw.shape[:2]:
        raise ValueError("reliability must be T-1,N")
    # The v2 hypothesis deliberately does not use the v1 RGB affinity. Accept
    # that argument only for the shared probe interface; do not claim semantics.
    del colors
    dt = np.diff(times)
    normalized = residual.reshape(len(raw), -1) / np.sqrt(dt[:, None])
    u, singular, vectors = np.linalg.svd(normalized, full_matrices=False)
    energy = singular ** 2
    total_energy = float(energy.sum())
    positions = np.median(xy, axis=0) / short_side
    design = np.c_[np.ones(len(positions)), positions - positions.mean(axis=0)]
    neighborhood = DecompositionConfig(spatial_radius_fraction=config.spatial_radius_fraction,
                                        minimum_neighbors=config.minimum_neighbors,
                                        coherence_error_scale=config.coherence_error_scale)
    full = raw.copy()
    structure_only = raw.copy()
    diagnostics = []
    dots = np.sum(raw[:-1] * raw[1:], axis=-1)
    norms = np.linalg.norm(raw[:-1], axis=-1) * np.linalg.norm(raw[1:], axis=-1)
    observed_reversal = ((dots < -.25 * norms) & (norms > 1e-10)).mean(axis=0)
    # Roundoff from exact constant velocity must not create candidate modes.
    numerical_floor = np.finfo(float).eps ** 2 * max(1., float(np.sum(xy ** 2))) * 1e4
    if total_energy > numerical_floor:
        for index, (s, vector) in enumerate(zip(singular, vectors)):
            fraction = float(energy[index] / total_energy)
            if fraction < config.minimum_mode_energy_fraction:
                continue
            temporal = u[:, index] * s / np.sqrt(dt)
            field = vector.reshape(-1, 2)
            point_energy = np.sum(field ** 2, axis=1)
            point_energy /= point_energy.sum()
            participation = float(1 / (len(field) * np.sum(point_energy ** 2)))
            predicted = design @ np.linalg.lstsq(design, field, rcond=None)[0]
            affine = float(np.clip(1 - np.sum((field - predicted) ** 2) / np.sum(field ** 2), 0, 1))
            directional = float(np.linalg.norm(field.sum(axis=0)) ** 2
                                / np.linalg.norm(field, axis=1).sum() ** 2)
            point_reliable = reliable.mean(axis=0) >= config.minimum_evidence_fraction
            support, tested = spatial_support(field[None], np.median(xy, axis=0), None,
                                               point_reliable[None], short_side, neighborhood)
            tested_energy = float(np.sum(point_energy * tested[0]))
            supported_energy = float(np.sum(point_energy * tested[0]
                                             * (support[0] >= config.structure_retention_threshold)))
            evidence = float(np.mean(reliable @ point_energy))
            # Zeros near a turning point do not count as observations of reversal.
            active = np.abs(temporal) > .1 * np.max(np.abs(temporal))
            adjacent = active[:-1] & active[1:]
            reversal = float(np.mean(temporal[:-1][adjacent] * temporal[1:][adjacent] < 0)) if adjacent.any() else 0.
            raw_reversal = float(np.sum(observed_reversal * point_energy))
            gap = min((abs(float(s - other)) / float(s) for j, other in enumerate(singular) if j != index), default=1.)
            reasons = []
            if gap < config.minimum_mode_separation:
                reasons.append("degenerate_mode_basis")
            if participation < config.minimum_participation_fraction:
                reasons.append("localized_motion")
            if evidence < config.minimum_evidence_fraction or tested_energy < config.minimum_evidence_fraction:
                reasons.append("insufficient_correspondence_or_neighborhood")
            if affine >= config.structure_retention_threshold:
                reasons.append("affine_motion")
            if directional >= config.structure_retention_threshold:
                reasons.append("same_direction_motion")
            if supported_energy >= config.local_supported_energy_threshold:
                reasons.append("locally_coordinated_motion")
            structure_eligible = not reasons
            repeated = (reversal > config.minimum_reversal_fraction
                        and raw_reversal > config.minimum_reversal_fraction)
            if not repeated:
                reasons.append("no_repeated_observed_reversal")
            # A nuisance mode can share its waveform with genuine camera motion.
            # Even when the mode is eligible, retain its affine projection; do
            # not remove the whole component to improve counterfactual numbers.
            component = (u[:, index] * s * np.sqrt(dt))[:, None, None] * (field - predicted)[None]
            if structure_eligible:
                structure_only -= component
            if structure_eligible and repeated:
                full -= component
            diagnostics.append({"index": index, "energy_fraction": fraction,
                                "participation": participation, "affine_explained": affine,
                                "directional_coherence_squared": directional,
                                "reliable_energy_fraction": evidence, "tested_energy_fraction": tested_energy,
                                "locally_supported_energy_fraction": supported_energy,
                                "temporal_reversal_fraction": reversal, "observed_reversal_fraction": raw_reversal,
                                "relative_singular_gap": gap, "removed": structure_eligible and repeated,
                                "affine_projection_retained": True,
                                "retention_reasons": reasons})
    return {"raw_deltas": raw, "temporal_only_deltas": trend,
            "structure_only_deltas": structure_only, "full_deltas": full,
            "modes": diagnostics, "reversal_fraction": observed_reversal,
            "lag_speeds": {str(lag): float(np.mean(np.linalg.norm(xy[lag:] - xy[:-lag], axis=-1)
                                                   / (times[lag:] - times[:-lag])[:, None]) / short_side)
                           for lag in range(1, min(4, len(times) - 1) + 1)}}
