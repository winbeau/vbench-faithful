from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.counterfactual.review_selection import select_reviewed_candidates
from scripts.counterfactual.static_jitter import digest


@pytest.fixture
def cohort(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in ("construction", "sources"):
        (tmp_path / "configs" / name).write_text(name)
    row = {"candidate_id": "cf", "video_uid": "source", "status": "rejected", "seed": 1,
           "sha256": "video-sha", "split": "dev", "protocol": "official-video-local-texture-jitter-v1",
           "family": "local_texture_alternating", "amplitude": 8, "pixel_exact_to_intended": True,
           "native_timeline_preserved": True, "sampling_coordinates_in_bounds": True,
           "minimum_warp_jacobian": .44, "intensity_noise_added": False}
    rows = [{**row, "candidate_id": family, "family": family, "status": "qualified"}
            for family in ("original", "encoding_control")] + [row]
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)))
    review = {"schema": "counterfactual-user-review-v1", "decision": "accepted_as_development_counterfactual",
              "manifest_sha256": digest(manifest), "construction_config": "configs/construction",
              "construction_config_sha256": digest(tmp_path / "configs/construction"),
              "sources": "configs/sources", "sources_sha256": digest(tmp_path / "configs/sources"),
              "amplitude": 8, "amplitude_units": "native_pixels", "family": row["family"],
              "candidates": [{k: row[k] for k in ("candidate_id", "video_uid", "seed", "sha256")} | {"automatic_status_at_build": "rejected"}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return rows, review, path, manifest


def test_exact_reviewed_cohort_without_rewriting_automatic_status(cohort, tmp_path):
    rows, _, path, manifest = cohort
    before = deepcopy(rows)
    assert select_reviewed_candidates(rows, path, manifest, tmp_path) == rows
    assert rows == before and rows[-1]["status"] == "rejected"


@pytest.mark.parametrize("field,value", [("sha256", "changed"), ("seed", 2), ("split", "test"),
                                      ("minimum_warp_jacobian", -1), ("native_timeline_preserved", False)])
def test_review_does_not_authorize_other_bytes_cases_or_integrity_failures(cohort, tmp_path, field, value):
    rows, _, path, manifest = cohort
    rows[-1][field] = value
    with pytest.raises(ValueError):
        select_reviewed_candidates(rows, path, manifest, tmp_path)


def test_manifest_and_config_hashes_are_binding(cohort, tmp_path):
    rows, _, path, manifest = cohort
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        select_reviewed_candidates(rows, path, manifest, tmp_path)


def test_user_review_does_not_implicitly_include_unreviewed_counterfactual(cohort, tmp_path):
    rows, _, path, manifest = cohort
    extra = {**rows[-1], "candidate_id": "unreviewed", "seed": 100}
    assert select_reviewed_candidates(rows + [extra], path, manifest, tmp_path) == rows
