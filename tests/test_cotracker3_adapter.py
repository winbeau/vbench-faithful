from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vbench_audit_models.cotracker3 import (
    CoTracker3OfflineModel, file_sha256, source_identity, verify_local_assets,
)


def fixture_assets(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root = tmp_path / "tracker"
    for name in ("cotracker/predictor.py", "cotracker/models/build_cotracker.py",
                 "cotracker/models/core/cotracker/cotracker3_offline.py"):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test-only upstream source identity\n")
    weight = tmp_path / "tiny-test-state.pth"
    torch.save({"model": {"weight": torch.full((2, 2), 7.), "bias": torch.full((2,), 3.)}}, weight)
    imports, constructors, queries_seen = [], [], []

    class Predictor(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__(); constructors.append(kwargs)
            self.model = torch.nn.Linear(2, 2)

        def forward(self, video, *, queries, backward_tracking):
            queries_seen.append((video.detach().clone(), queries.detach().clone(), backward_tracking))
            tracks = queries[:, None, :, 1:].repeat(1, video.shape[1], 1, 1)
            tracks[:, -1, :, 0] = 101  # must not be clamped into the image
            return tracks, torch.ones(tracks.shape[:-1], dtype=torch.bool)

    def importing(name):
        imports.append(name)
        assert name == "cotracker.predictor"
        return SimpleNamespace(__file__=str(root / "cotracker/predictor.py"), CoTrackerPredictor=Predictor)

    monkeypatch.setattr("vbench_audit_models.cotracker3.importlib.import_module", importing)
    for name in list(sys.modules):
        if name == "cotracker" or name.startswith("cotracker."):
            monkeypatch.delitem(sys.modules, name)
    def no_network(*args, **kwargs):
        raise AssertionError("network/torch.hub is forbidden in local-only adapter")
    monkeypatch.setattr(torch.hub, "load", no_network)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_network)
    kwargs = {"expected_source": source_identity(root), "expected_sha256": file_sha256(weight),
              "expected_size_bytes": weight.stat().st_size}
    return root, weight, kwargs, imports, constructors, queries_seen


def test_local_strict_safe_loading_uses_v3_not_v2(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root, weight, kwargs, imports, constructors, _ = fixture_assets(tmp_path, monkeypatch)
    original = torch.load; loads = []
    def load(*args, **options):
        loads.append(options)
        return original(*args, **options)
    monkeypatch.setattr(torch, "load", load)
    before_path = list(sys.path)
    model = CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    assert imports == ["cotracker.predictor"]
    assert constructors == [{"checkpoint": None, "v2": False, "offline": True, "window_len": 60}]
    assert loads == [{"map_location": "cpu", "weights_only": True}]
    assert (model.model.model.weight == 7).all() and (model.model.model.bias == 3).all()
    assert model.identity["checkpoint_sha256"] == file_sha256(weight)
    assert model.identity["query_frame_forced_by_upstream"] is True
    assert model.identity["independent_reverse_endpoint_check"] is False
    assert sys.path == before_path


def test_full_native_query_contract_and_out_of_view_predictions(tmp_path, monkeypatch):
    root, weight, kwargs, _, _, calls = fixture_assets(tmp_path, monkeypatch)
    model = CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    frames = np.arange(4 * 12 * 16 * 3, dtype=np.uint8).reshape(4, 12, 16, 3)
    queries = np.array([[2, 3, 4], [0, 9, 10]], np.float32)
    result = model.track_queries(frames, queries)
    video, observed, backward = calls[0]
    np.testing.assert_array_equal(video[0].permute(0, 2, 3, 1), frames)
    np.testing.assert_array_equal(observed[0], queries)
    assert backward is True and result["tracks"].shape == (4, 2, 2)
    assert (result["tracks"][-1, :, 0] == 101).all() and queries[0, 1] == 3
    assert result["visible"].dtype == np.bool_
    with pytest.raises(ValueError):
        model.track_queries(frames, [[.5, 3, 4]])
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["absent", "weight_sha", "weight_size", "source_changed", "source_extra", "source_incomplete"])
def test_missing_or_mismatched_assets_fail_before_import_or_initialization(tmp_path, monkeypatch, failure):
    root, weight, kwargs, imports, constructors, _ = fixture_assets(tmp_path, monkeypatch)
    if failure == "absent":
        weight = tmp_path / "missing.pth"
    elif failure == "weight_sha":
        kwargs["expected_sha256"] = "0" * 64
    elif failure == "weight_size":
        kwargs["expected_size_bytes"] += 1
    elif failure == "source_changed":
        (root / "cotracker/predictor.py").write_text("# changed after pin\n")
    elif failure == "source_extra":
        (root / "cotracker/new.py").write_text("# unbound module\n")
    else:
        kwargs["expected_source"].pop("cotracker/models/build_cotracker.py")
    with pytest.raises((FileNotFoundError, ValueError)):
        CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    assert imports == constructors == []


def test_foreign_loaded_namespace_rejected_even_if_predictor_not_yet_imported(tmp_path, monkeypatch):
    root, weight, kwargs, imports, _, _ = fixture_assets(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "cotracker.models", SimpleNamespace(__file__=str(tmp_path / "other/models.py")))
    with pytest.raises(RuntimeError, match="different cotracker"):
        CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    assert imports == []


def test_unbound_same_directory_module_is_not_an_execution_identity(tmp_path, monkeypatch):
    root, weight, kwargs, imports, _, _ = fixture_assets(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "cotracker.unpinned", SimpleNamespace(__file__=str(root / "cotracker/unpinned.py")))
    with pytest.raises(RuntimeError, match="unbound cotracker source"):
        CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    assert imports == []


@pytest.mark.parametrize("failure", ["nonfinite_track", "nonboolean_visibility"])
def test_invalid_predictions_are_not_coerced_into_evidence(tmp_path, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    root, weight, kwargs, _, _, _ = fixture_assets(tmp_path, monkeypatch)
    model = CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    def predict(video, *, queries, backward_tracking):
        tracks = queries[:, None, :, 1:].repeat(1, video.shape[1], 1, 1)
        visible = torch.ones(tracks.shape[:-1], dtype=torch.bool)
        if failure == "nonfinite_track":
            tracks[0, -1, 0, 0] = torch.nan
        else:
            visible = visible.float()
        return tracks, visible
    model.model = predict
    with pytest.raises(ValueError):
        model.track_queries(np.zeros((4, 12, 16, 3), np.uint8), [[1, 3, 4]])


def test_predictor_import_escape_and_source_symlink_escape_are_rejected(tmp_path, monkeypatch):
    root, weight, kwargs, _, _, _ = fixture_assets(tmp_path, monkeypatch)
    before = list(sys.path)
    monkeypatch.setattr("vbench_audit_models.cotracker3.importlib.import_module",
                        lambda _: SimpleNamespace(__file__=str(tmp_path / "other.py")))
    with pytest.raises(RuntimeError, match="predictor import escaped"):
        CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    assert before == sys.path
    target = tmp_path / "external.py"; target.write_text("# external source\n")
    (root / "cotracker/linked.py").symlink_to(target)
    with pytest.raises(ValueError, match="symlink escapes"):
        source_identity(root)


def test_malformed_or_incompatible_state_does_not_return_random_network(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root, weight, kwargs, _, _, _ = fixture_assets(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, "load", lambda *a, **k: "not a state")
    with pytest.raises(ValueError, match="state dictionary"):
        CoTracker3OfflineModel(root, weight, "cpu", **kwargs)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})
    with pytest.raises(RuntimeError):
        CoTracker3OfflineModel(root, weight, "cpu", **kwargs)


def test_asset_manifest_is_explicit_and_positive(tmp_path):
    with pytest.raises(ValueError):
        verify_local_assets(tmp_path, tmp_path / "no-weight", expected_source={},
                            expected_sha256="", expected_size_bytes=0)
