"""CPU replay of local Lagrangian paths from cached forward/backward flows.

Pair-local geometric/photometric reliability is a proxy, not learned visibility.
No construction fields, reference video or metadata are accepted.
"""
from __future__ import annotations

import numpy as np

from .dense_correspondence import dense_evidence, sample_field
from .local_trajectory import plan_windows


def local_dense_paths(frames, timestamps, forward, backward, queries, config):
    windows, owner = plan_windows(timestamps, config)
    queries = np.asarray(queries, np.float32)
    if queries.ndim != 2 or queries.shape[-1] != 2 or not np.isfinite(queries).all():
        raise ValueError("finite N,2 initial queries required")
    if forward.shape != backward.shape or forward.shape != (len(frames) - 1, *frames.shape[1:3], 2):
        raise ValueError("flow geometry does not match frames")
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("nonfinite flow cannot define a path")
    result = {"queries": queries, "windows": np.asarray(windows), "transition_owner": owner}
    for k, (a, b) in enumerate(windows):
        points = [queries.copy()]
        for flow in forward[a:b - 1]:
            points.append(points[-1] + sample_field(flow, points[-1]))
        xy = np.stack(points)
        evidence = dense_evidence(frames[a:b], forward[a:b - 1], backward[a:b - 1], config,
                                  query_positions=xy[:-1])
        reliable = (evidence["inside_pair"] & evidence["visible_pair"]
                    & (evidence["cycle_error"] <= config.cycle_error_max)
                    & (evidence["moving_error"] <= config.match_error_max))
        result[f"window_{k}_tracks"] = xy
        result[f"window_{k}_reliable_pair"] = reliable
    return result
