from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.counterfactual.local_texture_jitter import local_texture_jitter
from scripts.counterfactual.score_static_jitter import select_candidates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/dynamic-static-jitter/construction.local-texture-dev5-8px-v1.json").read_text())


def records():
    base = {"candidate_id": "base", "status": "qualified", "split": "dev", "protocol": CONFIG["protocol"]}
    rejected = {**base, "candidate_id": "cf", "status": "rejected", "sha256": "abc",
                "family": "local_texture_alternating", "pixel_exact_to_intended": True,
                "native_timeline_preserved": True, "sampling_coordinates_in_bounds": True,
                "minimum_warp_jacobian": .44, "intensity_noise_added": False}
    return [base, rejected]


def test_stress_is_explicit_and_never_changes_quality_qualification():
    rows = records()
    original = deepcopy(rows)
    assert select_candidates(rows) == rows[:1]
    assert select_candidates(rows, stress_test=True) == rows
    assert rows == original and rows[1]["status"] == "rejected"


@pytest.mark.parametrize("key,value", [("split", "test"), ("minimum_warp_jacobian", -.1),
                                      ("sha256", None), ("native_timeline_preserved", False),
                                      ("intensity_noise_added", True)])
def test_stress_cannot_bypass_test_freeze_or_media_integrity(key, value):
    rows = records()
    rows[1][key] = value
    with pytest.raises(ValueError):
        select_candidates(rows, stress_test=True)


def test_eight_pixel_extension_retains_strict_gates_and_same_spatial_field():
    previous = json.loads((ROOT / "configs/dynamic-static-jitter/construction.local-texture-dev5-v1.json").read_text())
    for key in ["min_warp_jacobian", "min_structure_correlation", "spatial_scale_pixels", "border_taper_pixels", "seeds"]:
        assert CONFIG[key] == previous[key]
    frames = np.zeros((4, 256, 256, 3), np.uint8)
    _, small, field4 = local_texture_jitter(frames, 4, 1701, previous)
    _, large, field8 = local_texture_jitter(frames, 8, 1701, CONFIG)
    assert small["geometry_qualified"] and not large["geometry_qualified"]
    assert 0 < large["minimum_warp_jacobian"] < CONFIG["min_warp_jacobian"]
    assert large["sampling_coordinates_in_bounds"]
    assert np.array_equal(field8["displacement_xy"], 2 * field4["displacement_xy"])


def test_strict_analysis_still_excludes_rejects_and_stress_reports_them_separately():
    from scripts.counterfactual.analyze_official_video_jitter import case_table, tables

    rows = records()
    for i, row in enumerate(rows):
        row.update(base_id="source", prompt_id="prompt", amplitude=0 if i == 0 else 8,
                   family="original" if i == 0 else "local_texture_alternating", sha256="abc")
    scores = {"base": {"input_sha256": "abc", "origin": {"status": "succeeded", "score": 0},
                       "repair": {"status": "succeeded", "score": .002}},
              "cf": {"input_sha256": "abc", "origin": {"status": "succeeded", "score": 1},
                     "repair": {"status": "insufficient_evidence", "score": None}}}
    strict = next(t for t in tables(rows, scores) if t["amplitude"] == 8)
    stress = next(t for t in tables(rows, scores, stress_test=True) if t["amplitude"] == 8)
    assert strict["origin:full"]["paired_count"] == 0
    assert stress["qualified"] == 0 and stress["scoring_population"] == 1
    assert stress["origin:full"]["paired_count"] == 1 and stress["origin:full"]["zero_to_one"] == 1
    assert stress["origin:full"]["coverage_qualified"] == 0
    assert stress["origin:full"]["coverage_scoring_population"] == 1
    assert stress["repair:full"]["paired_count"] == 0
    assert case_table(rows, scores)[1]["origin_cf"] is None
    shown = case_table(rows, scores, stress_test=True)[1]
    assert shown["origin_cf"] == 1 and shown["status"] == "rejected" and shown["repair_cf"] is None
