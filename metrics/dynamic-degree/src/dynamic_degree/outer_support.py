"""A development ablation of SAM partition support, not new motion evidence.

Prefer the largest containing proposal instead of the smallest containing
proposal. Tiny nested masks otherwise leave too few grid points to distinguish
an affine part from arbitrary texture motion. Masks are proposals in ONE frame;
the resulting groups are not certified objects or persistent part identities.
"""
from __future__ import annotations

import numpy as np

from .image_plane_modes import region_partition, spatial_design
from .residual_reversal import residual_first_reversal


def outer_support_partition(masks, xy):
    masks, xy = np.asarray(masks, bool), np.asarray(xy, float)
    # Reuse all geometry validation and the explicit uncovered=-1 convention.
    owners = region_partition(masks, xy)
    pixels = np.rint(xy).astype(int)
    owners[:] = -1
    areas = masks.sum(axis=(1, 2))
    for index in np.argsort(-areas, kind='stable'):
        member = masks[index, pixels[:, 1], pixels[:, 0]] & (owners == -1)
        owners[member] = index
    return owners


def partition_support(owners, xy, shape, reliable):
    """Count geometrically unidentifiable affine groups; never label motion."""
    design = spatial_design(xy, shape, 0)
    groups, protected = 0, 0
    for observed in reliable:
        for owner in np.unique(owners):
            selected = observed & (owners == owner)
            count = int(selected.sum())
            if count:
                groups += 1
                if count <= 3 or np.linalg.matrix_rank(design[selected]) < 3:
                    protected += count
    return {'regions_with_points': int(len(np.unique(owners))),
            'region_times_with_observations': groups,
            'fully_protected_degenerate_point_pairs': protected,
            'reliable_point_pairs': int(reliable.sum())}


def outer_support_reversal(values, times, xy, masks, reliable, shape):
    """Change only partition ownership; keep temporal gates/head untouched.

    No video identity, paired base, known warp, phase or score target is used.
    Affine protection applies to the NEW larger groups, not every small part.
    Natural articulated/periodic/small-motion response therefore needs testing.
    """
    old_owners = region_partition(masks, xy)
    owners = outer_support_partition(masks, xy)
    result, arrays = residual_first_reversal(values, times, xy, owners, reliable, shape)
    result.update(version='outer-support-residual-reversal-v1',
                  ownership_changed_points=int(np.sum(owners != old_owners)),
                  smallest_support=partition_support(old_owners, xy, shape, reliable),
                  outer_support=partition_support(owners, xy, shape, reliable),
                  limitation='larger single-frame SAM proposals may merge real moving parts; '
                  'affine preservation is not motion correctness or physical jitter classification')
    arrays['smallest_owners'] = old_owners
    return result, arrays
