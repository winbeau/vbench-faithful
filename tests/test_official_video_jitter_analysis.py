from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.counterfactual.analyze_official_video_jitter import case_table, tables, validate_population
from scripts.counterfactual.official_video_jitter import interventions


CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "configs/dynamic-static-jitter/construction.official-dev-v1.json").read_text())


def candidates():
    return [{**spec, "candidate_id": str(i), "base_id": "source", "prompt_id": "p", "split": "dev",
             "kind": "official_native", "protocol": CONFIG["protocol"], "status": "qualified", "sha256": str(i),
             "frame_map": "identity_all_source_frames", "coordinate_map": "identity", "native_timeline_preserved": True,
             "pixel_exact_to_intended": True, "source_shape": [16, 512, 512, 3], "decoded_shape": [16, 512, 512, 3],
             "source_format_adaptation": "none", "source_video": "/official.mp4", "source_sha256": "0",
             "video": "/official.mp4" if i == 0 else f"/output/{i}.mp4"}
            for i, spec in enumerate(interventions(CONFIG))]


def test_population_requires_unaltered_original_not_a_replacement():
    rows = candidates()
    assert len(validate_population(rows, CONFIG)) == 1
    changed = deepcopy(rows)
    changed[0]["video"] = "/reencoded_base.mp4"
    with pytest.raises(ValueError, match="baseline was replaced"):
        validate_population(changed, CONFIG)
    with pytest.raises(ValueError, match="incomplete"):
        validate_population(rows[:-1], CONFIG)


def test_synthetic_video_and_unfrozen_test_cannot_be_presented_as_native_development():
    rows = candidates()
    rows[0]["kind"] = "static"
    with pytest.raises(ValueError, match="synthetic/staticized"):
        validate_population(rows, CONFIG)
    rows = candidates()
    for row in rows:
        row["split"] = "test"
    with pytest.raises(ValueError, match="must be frozen"):
        validate_population(rows, CONFIG)


def test_two_different_units_are_reported_and_missing_is_not_zero():
    rows = candidates()
    scores = {"0": {"input_sha256": "0", "origin": {"status": "succeeded", "score": 0.0},
                    "repair": {"status": "succeeded", "score": 0.02}},
              "2": {"input_sha256": "2", "origin": {"status": "succeeded", "score": 1.0},
                    "repair": {"status": "succeeded", "score": 0.021}}}
    table = next(c for c in tables(rows, scores) if c["family"] == "pixel_rgb" and c["amplitude"] == 8)
    assert table["qualified"] == 2
    origin, repair = table["origin:full"], table["repair:full"]
    assert origin["zero_to_one"] == 1 and origin["one_to_zero"] == 0
    assert repair["base"]["mean"] == 0.02 and repair["counterfactual"]["mean"] == 0.021
    assert repair["absolute_delta"]["mean"] == pytest.approx(0.001)
    assert repair["coverage_qualified"] == .5 and repair["scoring_status_counts"]["NOT RUN"] == 1
    assert repair["base"]["ci95"] is None
    assert "static_false_positive_rate" not in origin


def test_case_table_keeps_insufficient_scores_and_checks_sampled_jitter_phase():
    rows = candidates()
    rows[2]["temporal_phase"] = [-1, 1, -1, 1]
    scores = {"0": {"input_sha256": "0", "repair": {"status": "insufficient_evidence", "score": None, "coverage": .3}},
              "2": {"input_sha256": "2", "origin": {"status": "succeeded", "score": 1.0,
                       "diagnostics": {"sampling": {"sampled_source_frame_indices": [0, 2]}}},
                    "repair": {"status": "insufficient_evidence", "score": None, "coverage": .4,
                               "sampling": {"source_indices": [0, 1, 2, 3]}}}}
    cases = case_table(rows, scores)
    assert len(cases) == len(rows)
    assert cases[2]["repair_base"] is None and cases[2]["repair_cf"] is None
    assert cases[2]["repair_cf_status"] == "insufficient_evidence"
    assert cases[2]["repair_cf_coverage"] == .4
    assert cases[2]["origin_sampled_phase_changes"] == 0
    assert cases[2]["repair_sampled_phase_changes"] == 3
