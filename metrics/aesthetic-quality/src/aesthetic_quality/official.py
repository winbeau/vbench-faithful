"""Preserve the pinned Aesthetic Quality computation and native aggregate."""
from vbench_audit_core.upstream import import_official_module


def compute(full_info, device, submodules):
    """Called in this dimension's isolated visual worker after asset preflight."""
    module, _ = import_official_module("aesthetic_quality")
    return module.compute_aesthetic_quality(full_info, device, submodules)
