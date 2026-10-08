from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.counterfactual.qualify_static_jitter import qualification
from scripts.counterfactual.static_jitter import variants


CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "tests/fixtures/construction/construction.dev-v2.json").read_text())


def ledger():
    return [{**spec, "candidate_id": f"video_{i:06d}", "base_id": "base", "prompt_id": "prompt",
             "split": "dev", "status": "qualified"} for i, spec in enumerate(variants(CONFIG))]


def test_qualification_requires_complete_cartesian_construction_not_scores():
    rows = ledger()
    result = qualification(rows, CONFIG)
    assert result["candidate_count"] == 191 and result["score_blind"]
    assert all(g["eligible"] for g in result["groups"])
    with pytest.raises(ValueError, match="incomplete"):
        qualification(rows[:-1], CONFIG)
    with pytest.raises(ValueError, match="duplicate candidate"):
        qualification(rows + rows[:1], CONFIG)


def test_rejection_keeps_ledger_and_excludes_complete_family_or_ladder():
    rows = ledger()
    for row in rows:
        if row["family"] == "pixel_rgb" and row["amplitude"] == 40 and row["seed"] == 1701 and row["motion"] == 0:
            row.update(status="rejected", reason="structure_damaged")
    result = qualification(rows, CONFIG)
    assert result["construction_status_counts"] == {"qualified": 189, "rejected": 2}
    rejected = [g for g in result["groups"] if not g["eligible"]]
    assert len(rejected) == 2
    assert {g["kind"] for g in rejected} == {"static", "translation"}
    assert all(g["input_exclusions"][0]["reason"] == "structure_damaged" for g in rejected)


def test_arbitrary_scores_and_filenames_cannot_change_qualification():
    rows = ledger()
    changed = deepcopy(rows)
    for row in changed:
        row.update(score=0, origin=1, repair=999, video="adversarial_filename.mp4")
    assert qualification(rows, CONFIG) == qualification(changed, CONFIG)


def test_clean_reference_rejection_disqualifies_noisy_ladders_before_scoring():
    rows = ledger()
    for row in rows:
        if row["kind"] == "translation" and row["family"] == "clean" and row["motion"] == 4:
            row["status"] = "construction_failed"
    result = qualification(rows, CONFIG)
    assert all(not g["eligible"] for g in result["groups"] if g["kind"] == "translation")
    assert all(g["eligible"] for g in result["groups"] if g["kind"] != "translation")
