import json
from pathlib import Path

import pytest

from scripts.counterfactual.probe_feature_tracks import tracker_reference_identity, make_tracker, main
from scripts.counterfactual.static_jitter import digest


def test_prepared_source_is_complete_but_not_claimed_as_prior_model_execution():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/dynamic-static-jitter/tracker.cotracker3-preparation-v1.json").read_text())
    result = tracker_reference_identity(config, "cotracker3-offline", root)
    assert len(result["tracker_code"]) == 31
    assert result["tracker_reference_role"] == "planned_local_assets_not_previous_model_inference"
    assert result["tracker_weight_sha256"] == config["model"]["expected_sha256"]
    assert result["tracker_source_manifest_sha256"] == digest(root / config["model"]["source_manifest"])
    config["model"]["predictor_args"]["v2"] = True
    with pytest.raises(ValueError, match="architecture identity"):
        tracker_reference_identity(config, "cotracker3-offline", root)


def test_legacy_reference_and_default_factory_remain_v2(monkeypatch):
    reference = {"tracker_code": {"old.py": "x"}, "tracker_weight_sha256": "weight"}
    result = tracker_reference_identity(reference, "cotracker2", Path("."))
    assert result["tracker_reference_role"] == "previous_cotracker2_execution"
    seen = []
    monkeypatch.setattr("vbench_audit_models.point_tracker.CoTracker2Model", lambda *a: seen.append(a) or "v2")
    assert make_tracker({}, "source", "weight", "cpu") == "v2"
    assert seen == [(Path("source"), Path("weight"), "cpu")]


def test_v3_factory_uses_all_asset_guards_without_network(monkeypatch):
    seen = []
    monkeypatch.setattr("vbench_audit_models.cotracker3.CoTracker3OfflineModel", lambda *a, **kw: seen.append((a, kw)) or "v3")
    request = {"tracker_kind": "cotracker3-offline", "tracker_code": {"a.py": "sha"},
               "tracker_weight_sha256": "weightsha", "tracker_weight_size_bytes": 17}
    assert make_tracker(request, "source", "weight", "cpu") == "v3"
    assert seen == [((Path("source"), Path("weight"), "cpu"),
                     {"expected_source": {"a.py": "sha"}, "expected_sha256": "weightsha", "expected_size_bytes": 17})]
    with pytest.raises(ValueError, match="unknown tracker"):
        make_tracker({"tracker_kind": "typo"}, "source", "weight", "cpu")


def test_wrong_model_reference_does_not_fall_back_to_legacy():
    with pytest.raises(ValueError):
        tracker_reference_identity({}, "cotracker3-offline", Path("."))
    with pytest.raises(SystemExit) as exc:
        main(["prepare", "--manifest", "absent", "--review", "absent", "--tracker-reference", "absent",
              "--output", "absent", "--candidates", "x", "--query-frame", "2", "--tracker-kind", "wrong"])
    assert exc.value.code == 2
