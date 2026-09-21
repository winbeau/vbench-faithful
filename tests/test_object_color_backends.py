"""Mock upstream tests prove call/metadata boundaries, not CUDA parity."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vbench_audit_core.contracts import run_batch_contract
from vbench_audit_models.grit import bind_heads


@pytest.mark.parametrize("dimension", ["object_class", "color"])
def test_official_invokes_compute_with_original_strings(monkeypatch, tmp_path, dimension):
    official = importlib.import_module(f"{dimension}.official")
    batch = importlib.import_module(f"{dimension}.metric").evaluate_batch
    videos = [tmp_path / f"video_{i:03}.mp4" for i in range(2)]
    metadata = {p.name: {"prompt": "an RED sofa", "dimension_metadata": {
        dimension: {"object": "SOFAS", "color": "RED"}}} for p in videos}
    calls = []

    def compute(path, device, submodules):
        request = json.loads(Path(path).read_text())
        calls.append(request)
        assert submodules == {"model_weight": str((tmp_path / "grit.pth").resolve())}
        assert request[0]["prompt_en"] == "an RED sofa"
        expected = {"object": "SOFAS"} if dimension == "object_class" else {"color": "RED"}
        assert request[0]["auxiliary_info"][dimension] == expected
        return .5, [{"video_path": str(videos[0].resolve()), "video_results": .5,
                     "frame_count": 16, "success_frame_count": 8}]

    module = SimpleNamespace(**{f"compute_{dimension}": compute})
    monkeypatch.setattr(official, "preflight", lambda *a: {"upstream": {"sha": "fixture-only"}})
    monkeypatch.setattr(official, "import_official_module", lambda *a: (module, None))
    rows = run_batch_contract(batch, "vbench", videos, metadata,
        config={"model": {"grit": {"checkpoint": str(tmp_path / "grit.pth")}},
                "runtime": {"evidence_dir": str(tmp_path / "evidence")}})
    assert len(calls) == 1
    assert len(rows) == 2
    assert rows[0].score == .5
    assert rows[1].status == "dropped_by_official" and rows[1].score is None
    assert Path(rows[0].metric["official_return"]["path"]).is_file()
    summary = importlib.import_module(f"{dimension}.runtime").summarize("vbench", rows)
    assert summary.aggregate is None and summary.counts["official_dropped_count"] == 1


@pytest.mark.parametrize("dimension", ["object_class", "color"])
def test_audit_consumes_instances_without_editing_official_metadata(monkeypatch, tmp_path, dimension):
    runtime = importlib.import_module(f"{dimension}.runtime")
    batch = importlib.import_module(f"{dimension}.metric").evaluate_batch
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"objects": ["car"], "colors": ["red"],
                                 "object_aliases": {"car": ["cars"]}}))
    primary = [{"text": "a red car", "score": .8, "box": [0, 0, 10, 10]}]
    objects = [{**primary[0], "text": "cars"}]
    frames = [{"status": "succeeded", "objects": objects, "primary": primary,
               "binding": bind_heads(primary, objects)}]
    model = SimpleNamespace(provenance={"fixture": True}, detect_video=lambda p: frames)
    monkeypatch.setattr(runtime.models, "build_model", lambda *a, **kw: model)
    video = tmp_path / "video_000.mp4"
    entry = {"prompt": "a red car", "dimension_metadata": {dimension: {"object": "DOG", "color": "BLUE"}}}
    rows = run_batch_contract(batch, "audit", [video], {video.name: entry}, config={
        "model": {"labels": {"path": str(labels)}}, "runtime": {"evidence_dir": str(tmp_path / "evidence")}})
    assert rows[0].score == 1
    assert rows[0].metric["query_source"] == "deterministic"
    assert entry["dimension_metadata"][dimension]["object"] == "DOG"
    artifact = json.loads(Path(rows[0].metric["evidence"]["path"]).read_text())
    assert artifact["raw_frames"][0]["objects"][0]["score"] == .8
    assert artifact["scoring"]["denominator_kind"] == "all_sampled_frames"
