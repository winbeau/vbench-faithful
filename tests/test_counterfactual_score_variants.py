"""Contract tests for the counterfactual repair-variant plumbing.

The Spatial Relationship repair has more than one scoring contract (plan
section 9.6), and the counterfactual driver appends to a per-shard JSONL file.
These tests pin properties that are easy to lose silently:

* one shard file must not end up holding rows from two repair configurations,
  because the CPA instrument would average incompatible scores;
* the report has to state which variant produced its numbers;
* a repair that floors at zero must stay distinguishable from one whose
  direction is inverted, which requires keeping the per-frame evidence.
"""
from __future__ import annotations

import json

from scripts.counterfactual.run_dimension import (
    evidence_summary,
    render_evidence_section,
    render_report,
)
from scripts.counterfactual.score import existing_derived_ids, frame_evidence


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_resume_returns_only_this_backends_ids(tmp_path):
    output = tmp_path / "spatial__repair__shard0.jsonl"
    _write_rows(output, [
        {"derived_id": "a", "backend": "repair", "repair_mode": "ordered_role",
         "detection_conditioned": True},
        {"derived_id": "b", "backend": "official"},
    ])
    config = {"repair_mode": "ordered_role", "detection_conditioned": True}
    assert existing_derived_ids(output, "repair", config) == {"a"}
    assert existing_derived_ids(output, "official", None) == {"b"}


def test_resume_is_empty_for_a_missing_file(tmp_path):
    config = {"repair_mode": "ordered_role", "detection_conditioned": False}
    assert existing_derived_ids(tmp_path / "absent.jsonl", "repair", config) == set()


def test_resume_refuses_a_second_repair_variant_in_one_shard(tmp_path):
    output = tmp_path / "spatial__repair__shard0.jsonl"
    _write_rows(output, [
        {"derived_id": "a", "backend": "repair", "repair_mode": "ordered_role",
         "detection_conditioned": False},
    ])
    requested = {"repair_mode": "ordered_role", "detection_conditioned": True}
    try:
        existing_derived_ids(output, "repair", requested)
    except SystemExit as exc:
        assert "fresh --output" in str(exc)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("a conflicting repair configuration must be fatal")


def test_legacy_rows_without_a_variant_are_read_as_the_identity_first_default(tmp_path):
    output = tmp_path / "spatial__repair__shard0.jsonl"
    _write_rows(output, [{"derived_id": "a", "backend": "repair"}])
    default = {"repair_mode": "ordered_role_identity_assignment", "detection_conditioned": False}
    assert existing_derived_ids(output, "repair", default) == {"a"}


def test_report_states_the_repair_variant():
    rows = [
        {"derived_id": "t0::original", "base_id": "t0", "family": "directional_flip",
         "level": "original", "expected_rank": 1, "split": "test"},
        {"derived_id": "t0::hflip", "base_id": "t0", "family": "directional_flip",
         "level": "hflip", "expected_rank": 0, "split": "test"},
    ]
    report = render_report(
        "spatial_relationship", "directional_flip", rows,
        [{"backend": "official", "scored_clips": 2, "expected_clips": 2, "incomplete_shards": []}],
        {"coverage": [], "profiles": {}, "contracts": {}},
        "deadbeef",
        None,
        {"repair_mode": "ordered_role", "detection_conditioned": True},
    )
    assert "- repair variant: `ordered_role` (detection-conditioned: true)" in report


def test_report_omits_the_variant_line_when_unspecified():
    rows = [
        {"derived_id": "t0::original", "base_id": "t0", "family": "directional_flip",
         "level": "original", "expected_rank": 1, "split": "test"},
    ]
    report = render_report(
        "spatial_relationship", "directional_flip", rows,
        [{"backend": "official", "scored_clips": 1, "expected_clips": 1, "incomplete_shards": []}],
        {"coverage": [], "profiles": {}, "contracts": {}},
        "deadbeef",
    )
    assert "- repair variant:" not in report


def _repair_diagnostics(reasons):
    return {
        "frame_results": [{"frame_index": index, "frame_reason": reason} for index, reason in enumerate(reasons)],
        "detected_frame_count": sum(1 for reason in reasons if reason not in {"missing_subject", "missing_object"}),
        "scored_frame_count": len(reasons),
        "missing_subject_count": reasons.count("missing_subject"),
        "missing_object_count": reasons.count("missing_object"),
        "ambiguous_role_count": 0,
        "valid_frame_count": sum(1 for reason in reasons if reason not in {"missing_subject", "missing_object"}),
        "aggregation_method": "mean_over_16_officially_sampled_frames",
    }


def test_frame_evidence_counts_reasons():
    evidence = frame_evidence(_repair_diagnostics(["missing_object", "direction_mismatch", "relation_satisfied"]))
    assert evidence["frame_reason_counts"] == {
        "direction_mismatch": 1, "missing_object": 1, "relation_satisfied": 1,
    }
    assert evidence["missing_object_count"] == 1
    assert evidence["scored_frame_count"] == 3


def test_frame_evidence_reads_official_zero_frame_rate():
    evidence = frame_evidence({"frame_results": [0.0, 1.0, 0.0, 0.0]})
    assert evidence["frame_count"] == 4
    assert evidence["zero_frame_fraction"] == 0.75
    assert "frame_reason_counts" not in evidence


def test_frame_evidence_is_absent_without_diagnostics():
    assert frame_evidence(None) is None
    assert frame_evidence({"frame_scores": [1.0]}) is not None


def test_evidence_summary_separates_dropouts_from_wrong_direction():
    rows = [{"derived_id": "a", "split": "test"}, {"derived_id": "b", "split": "test"}]
    evidence = {
        "a": {"repair": frame_evidence(_repair_diagnostics(["missing_object", "missing_object"]))},
        "b": {"repair": frame_evidence(_repair_diagnostics(["direction_mismatch", "relation_satisfied"]))},
    }
    summary = evidence_summary(rows, evidence, "repair")
    assert summary["clips"] == 2
    assert summary["frames"] == 4
    assert summary["missing_object"] == 2
    assert summary["direction_mismatch"] == 1
    assert summary["satisfied"] == 1


def test_evidence_section_says_so_when_nothing_was_recorded():
    rows = [{"derived_id": "a", "split": "test"}]
    lines = render_evidence_section(rows, None)
    assert "No per-clip evidence was recorded" in "\n".join(lines)


def test_evidence_section_renders_a_table_when_recorded():
    rows = [{"derived_id": "a", "split": "test"}]
    evidence = {"a": {"repair": frame_evidence(_repair_diagnostics(["relation_satisfied"]))}}
    text = "\n".join(render_evidence_section(rows, evidence))
    assert "## Frame evidence (test split)" in text
    assert "| repair | 1 | 1 |" in text
