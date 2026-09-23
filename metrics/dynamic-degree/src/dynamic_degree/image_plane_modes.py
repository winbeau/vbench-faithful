"""Single-video, leave-region-out checks of image-plane motion explanations.

This diagnostic does not remove any motion. Broad reversing nonaffine motion is
a hypothesis to test, not a physical-cause label; natural controls are required.
No construction field, phase, amplitude, reference video or prompt is accepted.
"""
from __future__ import annotations

import numpy as np


def region_partition(masks, xy):
    """Smallest containing SAM region owns a point; uncovered = explicit -1."""
    masks, xy = np.asarray(masks, bool), np.asarray(xy, float)
    if masks.ndim != 3 or xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("region stack and finite point locations required")
    pixels = np.rint(xy).astype(int)
    if (pixels < 0).any() or (pixels >= [masks.shape[2], masks.shape[1]]).any():
        raise ValueError("point outside region geometry")
    owners = np.full(len(xy), -1, int)
    # Stable ties, not persistent identities across frames/videos.
    for index in np.argsort(masks.sum(axis=(1, 2)), kind="stable"):
        member = masks[index, pixels[:, 1], pixels[:, 0]] & (owners == -1)
        owners[member] = index
    return owners


def spatial_design(xy, shape, grid):
    xy = np.asarray(xy, float) / (np.array(shape[::-1]) - 1)
    affine = np.c_[np.ones(len(xy)), xy - .5]
    if grid == 0:
        return affine
    if not isinstance(grid, int) or grid < 2:
        raise ValueError("zero for affine, or at least two spatial centers per axis")
    xx, yy = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
    centers = np.c_[xx.ravel(), yy.ravel()]
    gaussian = np.exp(-np.sum((xy[:, None] - centers[None]) ** 2, axis=-1) / (2 * (1 / (grid - 1)) ** 2))
    return np.c_[affine, gaussian]


def leave_region_prediction(values, xy, owners, valid, shape, *, grid, ridge=1e-3):
    """Predict each held-out region using no values from that region.

    Fit every temporal/channel response jointly with the same spatial design.
    The affine baseline is unpenalized and retained as a distinct explanation.
    A field being spatially predictable is not proof that it should be removed.
    """
    values, xy = np.asarray(values, float), np.asarray(xy, float)
    owners, valid = np.asarray(owners), np.asarray(valid, bool)
    if (values.ndim != 3 or values.shape[1] != len(xy) or values.shape[-1] != 2
            or valid.shape != (len(xy),) or owners.shape != valid.shape or not np.isfinite(values).all()
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("finite T,N,2 motion and point-level evidence required")
    design = spatial_design(xy, shape, grid)
    flat = values.transpose(1, 0, 2).reshape(len(xy), -1)
    prediction = np.full_like(flat, np.nan)
    folds = []
    for region in np.unique(owners):
        test = (owners == region) & valid
        train = (owners != region) & valid
        row = {"held_region": int(region), "train_points": int(train.sum()), "test_points": int(test.sum())}
        if not test.any() or train.sum() <= design.shape[1] or np.linalg.matrix_rank(design[train, :3]) < 3:
            folds.append(row | {"status": "insufficient_evidence"})
            continue
        normal = design[train].T @ design[train]
        penalty = np.zeros(normal.shape[0])
        penalty[3:] = ridge * train.sum()
        beta = np.linalg.solve(normal + np.diag(penalty), design[train].T @ flat[train])
        prediction[test] = design[test] @ beta
        residual = prediction[test] - flat[test]
        folds.append(row | {"status": "diagnostic_only", "squared_error": float(np.sum(residual ** 2)),
                            "observed_energy": float(np.sum(flat[test] ** 2))})
    return prediction.reshape(len(xy), values.shape[0], 2).transpose(1, 0, 2), folds


def temporal_modes(values, times, xy, shape, owners, *, max_modes=3):
    """Describe all-phase Eulerian velocity modes; never infer static truth."""
    values, times, xy = np.asarray(values, float), np.asarray(times, float), np.asarray(xy, float)
    dt = np.diff(times)
    if (values.ndim != 3 or values.shape != (len(dt), len(xy), 2) or len(dt) < 2
            or not np.isfinite(values).all() or not np.isfinite(times).all() or not (dt > 0).all()
            or len(owners) != len(xy)):
        raise ValueError("finite velocity samples and matching native times required")
    if not len(xy):
        return []
    centered = values - np.average(values, weights=dt, axis=0)
    weighted = centered.reshape(len(dt), -1) * np.sqrt(dt[:, None])
    u, singular, right = np.linalg.svd(weighted, full_matrices=False)
    total = float(np.sum(singular ** 2))
    if total <= np.finfo(float).eps ** 2 * max(1., float(np.sum(values ** 2))) * 1e4:
        return []
    design = spatial_design(xy, shape, 0)
    result = []
    for index in range(min(max_modes, len(singular))):
        field = right[index].reshape(-1, 2)
        fitted = design @ np.linalg.lstsq(design, field, rcond=None)[0]
        energy = np.sum(field ** 2, axis=1)
        temporal = u[:, index] * singular[index] / np.sqrt(dt)
        # Use nonzero signs at observed phases; no construction frequency/notch.
        active = abs(temporal) > .1 * np.max(abs(temporal))
        adjacent = active[:-1] & active[1:]
        reversal = float(np.mean(temporal[:-1][adjacent] * temporal[1:][adjacent] < 0)) if adjacent.any() else None
        by_region = [{"region": int(r), "energy_fraction": float(energy[np.asarray(owners) == r].sum() / energy.sum())}
                     for r in np.unique(owners)]
        result.append({"index": index, "energy_fraction": float(singular[index] ** 2 / total),
                       "reversal_fraction": reversal, "temporal_velocity_coefficients": temporal.tolist(),
                       "affine_explained_fraction": float(1 - np.sum((field - fitted) ** 2) / energy.sum()),
                       "participation_fraction": float(energy.sum() ** 2 / (len(energy) * np.sum(energy ** 2))),
                       "region_energy": by_region,
                       "warning": "descriptive full-input basis; no held-out certification or nuisance label"})
    return result


def leave_region_time_prediction(values, times, owners, reliable, *, rank, folds=3):
    """Two-way cross-check: temporal basis from OTHER regions, amplitudes from
    OTHER times of this point. Every phase is held out exactly once.

    Unlike spatial interpolation this does not invent an unseen region's field
    from remote regions. Predicted values are still model-flow evidence, not
    physical ground truth and not a permission to suppress an oscillation.
    """
    values, times = np.asarray(values, float), np.asarray(times, float)
    owners, reliable = np.asarray(owners), np.asarray(reliable, bool)
    dt = np.diff(times)
    if (values.ndim != 3 or values.shape != (*reliable.shape, 2) or len(values) != len(dt)
            or len(owners) != values.shape[1] or not np.isfinite(values).all() or not np.isfinite(times).all()
            or not (dt > 0).all() or not isinstance(rank, int) or rank < 0
            or not isinstance(folds, int) or not 2 <= folds < len(values)):
        raise ValueError("finite T,N,2 fields with evidence, timestamps and valid CV plan required")
    eligible = np.average(reliable, axis=0, weights=dt) >= .6
    prediction = np.full_like(values, np.nan)
    records = []
    for region in np.unique(owners):
        other = (owners != region) & eligible
        target = (owners == region) & eligible
        if not other.any() or not target.any():
            records.append({"held_region": int(region), "status": "insufficient_evidence"})
            continue
        basis_data = values[:, other] - np.average(values[:, other], weights=dt, axis=0)
        matrix = basis_data.reshape(len(dt), -1) * np.sqrt(dt[:, None])
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False) if rank else (np.zeros((len(dt), 0)), np.empty(0), None)
        numerical_floor = np.finfo(float).eps * max(1., float(np.linalg.norm(values[:, other]))) * 100
        available_rank = min(rank, int(np.sum(singular > numerical_floor)))
        basis = np.c_[np.ones(len(dt)), u[:, :available_rank] / np.sqrt(dt[:, None])]
        predicted_count, observed_count = 0, 0
        for fold in range(folds):
            held_times = np.arange(len(dt)) % folds == fold
            for point in np.flatnonzero(target):
                training = ~held_times & reliable[:, point]
                testing = held_times & reliable[:, point]
                observed_count += int(testing.sum())
                if training.sum() <= basis.shape[1] or np.linalg.matrix_rank(basis[training]) != basis.shape[1]:
                    continue
                weighted = basis[training] * np.sqrt(dt[training, None])
                beta = np.linalg.lstsq(weighted, values[training, point] * np.sqrt(dt[training, None]), rcond=None)[0]
                prediction[testing, point] = basis[testing] @ beta
                predicted_count += int(testing.sum())
        records.append({"held_region": int(region), "status": "diagnostic_only", "basis_source_points": int(other.sum()),
                        "target_points": int(target.sum()), "temporal_rank": available_rank,
                        "target_reliable_pairs": observed_count, "predicted_pairs": predicted_count})
    return prediction, records


def fit_temporal_field(values, times, wave, reliable):
    """Fit a constant and one temporal mode with observed pairs only.

    Coefficients at points with insufficient time coverage stay NaN. The wave
    must come from the current video; this fit is explanatory, not held-out.
    It does not certify that a component is an artifact.
    """
    values, times, wave = map(lambda a: np.asarray(a, float), (values, times, wave))
    reliable = np.asarray(reliable, bool)
    dt = np.diff(times)
    if (values.ndim != 3 or values.shape != (*reliable.shape, 2)
            or wave.shape != (len(dt),) or len(values) != len(dt) or len(dt) < 3
            or not all(np.isfinite(a).all() for a in (values, times, wave)) or not (dt > 0).all()):
        raise ValueError("finite temporal mode, T,N,2 velocities and pair evidence required")
    weight = dt[:, None] * reliable
    total = weight.sum(axis=0)
    w = wave[:, None]
    mean_wave = np.divide((weight * w).sum(axis=0), total, out=np.zeros_like(total), where=total > 0)
    mean_values = np.divide((weight[..., None] * values).sum(axis=0), total[:, None],
                            out=np.zeros_like(values[0]), where=total[:, None] > 0)
    variation = w - mean_wave
    energy = (weight * variation ** 2).sum(axis=0)
    valid = ((total >= .6 * dt.sum()) & (reliable.sum(axis=0) >= 3)
             & (energy > np.finfo(float).eps * max(1., float(np.sum(dt * wave ** 2))) * 100))
    field = np.full_like(values[0], np.nan)
    numerator = (weight[..., None] * variation[..., None] * (values - mean_values)).sum(axis=0)
    field[valid] = numerator[valid] / energy[valid, None]
    constant = mean_values - mean_wave[:, None] * field
    return field, constant, valid


def affine_field_split(field, xy, shape, valid):
    """Project onto global camera/affine motion; unknown points stay unknown.

    Any eventual subtraction must keep this affine projection. Projection is
    algebraic, not proof that the residual is nonphysical.
    """
    field, xy, valid = np.asarray(field, float), np.asarray(xy, float), np.asarray(valid, bool)
    if (field.shape != xy.shape or field.ndim != 2 or field.shape[1] != 2
            or valid.shape != (len(xy),) or not np.isfinite(xy).all()
            or not np.isfinite(field[valid]).all()):
        raise ValueError("finite observed N,2 field and coordinates required")
    design = spatial_design(xy, shape, 0)
    affine = np.full_like(field, np.nan)
    if valid.sum() < 4 or np.linalg.matrix_rank(design[valid]) < 3:
        return affine, affine.copy()
    affine[valid] = design[valid] @ np.linalg.lstsq(design[valid], field[valid], rcond=None)[0]
    return affine, field - affine


def boundary_motion(field, owners, valid, *, stride=1):
    """Test one-sided motion extrapolation across reference-region boundaries.

    Four equally spaced points straddle an edge. The central displacement
    difference minus the average of its two flanking differences annihilates
    affine AND quadratic fields. A piecewise translation instead leaves its
    jump. Every tested point must have evidence, and both points on each side
    must belong to that side's region. Missing support is reported, not zero.

    The result describes a sampled model field and a reference-frame SAM
    partition, NOT persistent parts, physical ground truth or a suppression
    rule. Smooth natural deformations can have the same statistic as jitter.
    """
    field, owners, valid = np.asarray(field, float), np.asarray(owners), np.asarray(valid, bool)
    if (field.ndim != 3 or field.shape[-1] != 2 or owners.shape != field.shape[:2]
            or valid.shape != owners.shape or not isinstance(stride, int) or stride < 1
            or min(owners.shape) <= 3 * stride or not np.isfinite(field[valid]).all()):
        raise ValueError("regular H,W,2 field with owners, evidence and a supported stride required")
    samples = {key: [] for key in ("jump", "difference", "left", "right", "available", "crossing", "inside")}
    for axis in (0, 1):
        n = field.shape[axis] - 3 * stride
        positions = []
        for k in range(4):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(k * stride, k * stride + n)
            positions.append(tuple(slices))
        a, b, c, d = [field[s] for s in positions]
        oa, ob, oc, od = [owners[s] for s in positions]
        available = np.logical_and.reduce([valid[s] for s in positions])
        crossing = (oa == ob) & (oc == od) & (ob != oc)
        inside = (oa == ob) & (ob == oc) & (oc == od)
        for name, value in (("jump", c - b - .5 * ((b - a) + (d - c))), ("difference", c - b),
                            ("left", ob), ("right", oc), ("available", available),
                            ("crossing", crossing), ("inside", inside)):
            samples[name].append(value.reshape(-1, 2) if name in ("jump", "difference") else value.ravel())
    arrays = {key: np.concatenate(value) for key, value in samples.items()}

    def summarize(selected):
        tested = selected & arrays["available"]
        count = int(tested.sum())
        jump = float(np.sum(arrays["jump"][tested] ** 2)) if count else None
        difference = float(np.sum(arrays["difference"][tested] ** 2)) if count else None
        return {"possible_stencils": int(selected.sum()), "observed_stencils": count,
                "jump_energy": jump, "central_difference_energy": difference,
                "jump_rms": float(np.sqrt(jump / count)) if count else None,
                "jump_over_difference_energy": jump / difference if count and difference > 1e-20 else None}

    result = {"stride": stride, "cross_region": summarize(arrays["crossing"]),
              "same_region": summarize(arrays["inside"]), "regions": []}
    for owner in np.unique(owners):
        touching = arrays["crossing"] & ((arrays["left"] == owner) | (arrays["right"] == owner))
        result["regions"].append({"region": int(owner), **summarize(touching)})
    return result, arrays


def constrained_mode_residual(component, times, xy, owners, observed, shape, *, iterations=400, tolerance=1e-9):
    """Candidate removal orthogonal to regional affine AND temporal-mean motion.

    Alternating orthogonal projections use the dt-weighted observed-pair inner
    product. Thus deletion cannot change each region's affine projection at a
    given time or a point's mean velocity. Unsupported/degenerate small regions
    are wholly protected. Missing entries are zero REMOVAL, not zero motion.
    Nonconvergence must veto removal. These constraints do not protect every
    physical nonaffine deformation and do not establish artifact causality.
    """
    component, times, xy = map(lambda a: np.asarray(a, float), (component, times, xy))
    owners, observed = np.asarray(owners), np.asarray(observed, bool)
    dt = np.diff(times)
    if (component.shape != (*observed.shape, 2) or observed.shape != (len(dt), len(xy))
            or owners.shape != (len(xy),) or not all(np.isfinite(a).all() for a in (component, times, xy))
            or not (dt > 0).all() or not isinstance(iterations, int) or iterations < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("finite component, matching observations and projection parameters required")
    design = spatial_design(xy, shape, 0)
    groups = []
    removable = observed.copy()
    for t in range(len(dt)):
        for region in np.unique(owners):
            selected = np.flatnonzero(observed[t] & (owners == region))
            if len(selected):
                q = None
                if len(selected) > 3 and np.linalg.matrix_rank(design[selected]) == 3:
                    q, _ = np.linalg.qr(design[selected], mode="reduced")
                else:
                    # This is an exact support constraint, not just a small
                    # residual tolerance after the next temporal projection.
                    removable[t, selected] = False
                groups.append((t, selected, q))
    weights = removable * dt[:, None]
    totals = weights.sum(axis=0)

    def spatial_null(values):
        result = np.zeros_like(values)
        for t, selected, q in groups:
            if q is not None:
                local = values[t, selected]
                result[t, selected] = local - q @ (q.T @ local)
        return result

    def temporal_null(values):
        mean = np.divide((weights[..., None] * values).sum(axis=0), totals[:, None],
                         out=np.zeros_like(values[0]), where=totals[:, None] > 0)
        return np.where(removable[..., None], values - mean, 0)

    def norm(values):
        return float(np.sqrt(np.sum(values ** 2 * dt[:, None, None])))

    residual = np.where(removable[..., None], component, 0)
    original_norm = norm(residual)
    limit = tolerance * max(original_norm, 1e-20)
    spatial_error = temporal_error = float("inf")
    for count in range(1, iterations + 1):
        residual = temporal_null(spatial_null(residual))
        spatial_error = norm(residual - spatial_null(residual))
        temporal_error = norm(residual - temporal_null(residual))
        if max(spatial_error, temporal_error) <= limit:
            break
    if norm(residual) <= np.finfo(float).eps * max(original_norm, 1e-20) * 100:
        residual[:] = 0
        spatial_error = temporal_error = 0.
    return residual, {"converged": bool(max(spatial_error, temporal_error) <= limit), "iterations": count,
                      "original_component_norm": original_norm, "removal_norm": norm(residual),
                      "affine_constraint_error": spatial_error, "mean_velocity_constraint_error": temporal_error,
                      "constraint_limit": limit, "observed_pairs": int(observed.sum()),
                      "removable_pairs": int(removable.sum()),
                      "total_pairs": int(observed.size), "protected_degenerate_region_times": sum(q is None for _, _, q in groups)}


def regional_reversal_ablation(values, times, xy, owners, reliable, shape):
    """DEV ablation, not a completed Repair or a physical-motion classifier.

    Posthoc DEV gates: leading mode energy >= .5, reversal > .5, participation
    >= .1, full-input reliable-pair fraction >= .6 and held-out prediction gain
    > .5. No pixel-amplitude, construction-phase or reference-video input.
    Only its region-affine/mean-velocity-orthogonal component may be subtracted.
    All unknown pairs stay unknown, and a conditional speed is NOT a score.
    """
    values, times, xy = map(lambda a: np.asarray(a, float), (values, times, xy))
    reliable, owners = np.asarray(reliable, bool), np.asarray(owners)
    dt = np.diff(times)
    if (values.ndim != 3 or values.shape != (*reliable.shape, 2) or reliable.shape != (len(dt), len(xy))
            or owners.shape != (len(xy),) or len(dt) < 4 or len(shape) != 2 or min(shape) <= 1
            or not all(np.isfinite(a).all() for a in (values, times, xy)) or not (dt > 0).all()):
        raise ValueError("finite full-phase velocities, geometry and evidence required")
    point_valid = np.average(reliable, axis=0, weights=dt) >= .6
    modes = temporal_modes(values[:, point_valid], times, xy[point_valid], shape, owners[point_valid])
    proposed = np.zeros_like(values)
    removal = np.zeros_like(values)
    projection_observed = np.zeros_like(reliable)
    reasons, prediction, constraints = [], None, None
    coverage = float(np.average(reliable.mean(axis=1), weights=dt))
    if not modes:
        reasons.append("no_temporal_mode")
    else:
        mode = modes[0]
        if mode["energy_fraction"] < .5:
            reasons.append("no_dominant_mode")
        if mode["reversal_fraction"] is None or mode["reversal_fraction"] <= .5:
            reasons.append("no_repeated_reversal")
        if mode["participation_fraction"] < .1:
            reasons.append("localized_mode")
        if coverage < .6:
            reasons.append("insufficient_pair_evidence")
        if len(modes) > 1 and 1 - np.sqrt(modes[1]["energy_fraction"] / mode["energy_fraction"]) < .05:
            reasons.append("nearly_degenerate_mode_basis")
        baseline, _ = leave_region_time_prediction(values, times, owners, reliable, rank=0)
        predicted, _ = leave_region_time_prediction(values, times, owners, reliable, rank=1)
        common = reliable & np.isfinite(baseline).all(axis=-1) & np.isfinite(predicted).all(axis=-1)
        errors = [float(np.sum(np.where(common, np.sum((np.nan_to_num(p) - values) ** 2, axis=-1), 0) * dt[:, None]))
                  for p in (baseline, predicted)]
        gain = 1 - errors[1] / errors[0] if common.any() and errors[0] > 1e-20 else None
        prediction = {"gain": gain, "common_pairs": int(common.sum()), "total_pairs": int(common.size)}
        if gain is None or gain <= .5:
            reasons.append("insufficient_cross_region_time_predictability")
        if not reasons:
            wave = np.asarray(mode["temporal_velocity_coefficients"])
            field, _, fitted = fit_temporal_field(values, times, wave, reliable)
            observed = reliable & fitted[None]
            projection_observed = observed
            proposed = wave[:, None, None] * np.nan_to_num(field)[None]
            tested_removal, constraints = constrained_mode_residual(proposed, times, xy, owners, observed, shape)
            if constraints["converged"]:
                removal = tested_removal
            else:
                reasons.append("constraint_projection_not_converged")
    corrected = values - removal
    denominator = float(np.sum(reliable * dt[:, None]) * min(shape))

    def conditional_speed(velocity):
        return float(np.sum(np.linalg.norm(velocity, axis=-1) * reliable * dt[:, None]) / denominator) if denominator else None

    result = {"status": "diagnostic_only", "score": None, "subtracted": bool(np.any(removal)),
              "retention_reasons": reasons, "reliable_pair_fraction": coverage, "heldout_prediction": prediction,
              "constraints": constraints, "conditional_raw_speed": conditional_speed(values),
              "conditional_guarded_speed": conditional_speed(corrected),
              "units": "short_side_lengths_per_second_on_observed_pairs_only",
              "warning": "conditional observed model motion; not a full score, artifact label or physical-motion validation"}
    return result, {"raw_velocity": values, "corrected_velocity": corrected, "proposed_component": proposed,
                    "removal": removal, "reliable_pair": reliable, "projection_observed": projection_observed,
                    "timestamps": times, "xy": xy, "owners": owners}
