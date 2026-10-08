"""Preserve the pinned Temporal Flickering computation and native aggregate."""
from vbench_audit_core.upstream import import_official_module


def compute(full_info, device, submodules):
    """Called in this dimension's isolated visual worker after asset preflight."""
    module, _ = import_official_module("temporal_flickering")
    return module.compute_temporal_flickering(full_info, device, submodules)
