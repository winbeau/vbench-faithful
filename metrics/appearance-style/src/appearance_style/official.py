"""Preserve the pinned Appearance Style computation and native aggregate."""
from vbench_audit_core.upstream import import_official_module


def compute(full_info, device, submodules):
    """Called in this dimension's isolated visual worker after asset preflight."""
    module, _ = import_official_module("appearance_style")
    return module.compute_appearance_style(full_info, device, submodules)
