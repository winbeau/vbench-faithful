from __future__ import annotations

from pathlib import Path

from vbench_audit_core.upstream import (
    UPSTREAM_SHA,
    UPSTREAM_URL,
    load_config,
    resolve_upstream_path,
    verify_upstream,
)


def test_locked_config_lists_all_dimensions_and_single_identity():
    config = load_config()
    assert config.url == UPSTREAM_URL
    assert config.sha == UPSTREAM_SHA
    assert set(config.dimensions) == {
        "dynamic_degree",
        "motion_smoothness",
        "subject_consistency",
        "scene",
        "human_action",
        "spatial_relationship",
        "overall_consistency",
        "multiple_objects",
        "background_consistency",
        "temporal_style",
        "object_class",
        "color",
        "aesthetic_quality",
        "imaging_quality",
        "temporal_flickering",
        "appearance_style",
    }


def test_default_checkout_is_sibling_and_clean_official_source_matches():
    selected = resolve_upstream_path()
    assert selected == Path(__file__).resolve().parents[4] / "VBench"
    state = verify_upstream(dimension="dynamic_degree")
    assert state.sha == UPSTREAM_SHA
    assert state.remote == UPSTREAM_URL
    assert state.source_hashes["dynamic_degree"] == load_config().dimensions["dynamic_degree"].source_sha256
