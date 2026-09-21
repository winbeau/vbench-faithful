from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from scripts.object_color_natural import cached_evaluator
from vbench_audit_core.schemas import VideoResult
from vbench_audit_core.inputs import sha256_file


def test_natural_resume_keeps_official_drops_and_avoids_reinference(tmp_path, monkeypatch):
    videos = [tmp_path / "one.gif", tmp_path / "two.mp4"]
    metadata = {p.name: {"video_uid": p.stem} for p in videos}
    calls = []

    def evaluate(backend, paths, metadata, device, config):
        calls.append(list(paths))
        return [VideoResult(str(p), "dropped_by_official", None,
                            {"video_uid": metadata[p.name]["video_uid"]}) for p in paths]

    monkeypatch.setattr("scripts.object_color_natural.importlib.import_module",
                        lambda name: SimpleNamespace(evaluate_batch=evaluate))
    config = {"experiment": {"dimension": "color", "checkpoints": str(tmp_path / "checkpoints"),
                            "context_sha256": "fixture", "chunk_size": 1}}
    first = cached_evaluator("vbench", videos, metadata, "cuda:0", config)
    second = cached_evaluator("vbench", list(reversed(videos)), metadata, "cuda:0", config)
    assert len(calls) == 2
    assert [asdict(r) for r in first] == [asdict(r) for r in reversed(second)]
    assert all(r.status == "dropped_by_official" and r.score is None for r in second)
    config["experiment"]["context_sha256"] = "different-formula"
    with pytest.raises(ValueError, match="context"):
        cached_evaluator("vbench", videos, metadata, "cuda:0", config)


def test_natural_checkpoint_rejects_duplicate_uids(tmp_path, monkeypatch):
    videos = [tmp_path / "one.gif", tmp_path / "two.mp4"]
    metadata = {p.name: {"video_uid": p.stem} for p in videos}
    monkeypatch.setattr("scripts.object_color_natural.importlib.import_module", lambda name:
        SimpleNamespace(evaluate_batch=lambda *a, **k: [VideoResult(str(p), "succeeded", .5,
            {"video_uid": "duplicate"}) for p in videos]))
    config = {"experiment": {"dimension": "color", "checkpoints": str(tmp_path / "checkpoints"),
                            "context_sha256": "fixture", "chunk_size": 2}}
    with pytest.raises(ValueError, match="duplicates"):
        cached_evaluator("vbench", videos, metadata, "cuda:0", config)
    assert not (tmp_path / "checkpoints").exists()


def test_natural_score_writes_common_results_after_checkpoint_creation(tmp_path, monkeypatch):
    from scripts.object_color_natural import score
    root = tmp_path / "output" / "natural"
    media = root / "media" / "cogvideo/object_class/a car-0.gif"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"mock-only-gif")
    row = {"video_uid": "original-uid", "relative_video_path": "cogvideo/object_class/a car-0.gif",
           "prompt": "a car", "official_auxiliary_info": {"object": "car"}}
    (root / "object_class-manifest.json").write_text(json.dumps({"rows": [row]}))
    (root / "media-ledger.json").write_text(json.dumps({"expected_videos": 1,
        "materialized_videos": 1, "errors": [], "files": [{"relative_path": row["relative_video_path"],
                                                              "sha256": sha256_file(media)}]}))
    (root / "protocol.json").write_text("{}")
    weights = root / "mock.weights"
    weights.write_bytes(b"not-a-model")
    labels = root / "vocab.json"
    labels.write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")

    def run(evaluator, backend, videos, metadata, gpu_ids, **kwargs):
        assert backend == "vbench" and gpu_ids == [0, 1]
        assert videos[0].suffix == ".gif" and videos[0].read_bytes() == media.read_bytes()
        assert metadata[videos[0].name]["prompt"] == "a car"
        return [VideoResult(str(videos[0]), "succeeded", .5,
            {"video_uid": "original-uid", "frame_count": 16, "success_frame_count": 8})]

    monkeypatch.setattr("scripts.object_color_natural.run_batch_sharded", run)
    score(SimpleNamespace(root=root, dimension="object_class", backend="official",
                          checkpoint=weights, vocabulary=labels, upstream=tmp_path, chunk_size=192))
    result = root / "scores/object_class/official"
    assert json.loads((result / "results.json").read_text())[0]["video_uid"] == "original-uid"
    assert json.loads((result / "summary.json").read_text())["aggregate"] == .5
    assert (root / "checkpoints/object_class/official/context.json").is_file()


def test_human_agreement_retains_missing_pairs_and_separates_common_coverage():
    from scripts.object_color_natural_report import METHODS, population
    videos = {}
    for uid, value in (("a", 1.), ("b", 0.), ("c", 1.), ("d", 0.)):
        videos[uid] = {"scores": {m: {"status": "succeeded", "score": value} for m in METHODS}}
    videos["c"]["scores"]["official"] = {"status": "dropped_by_official", "score": None}
    pairs = [
        {"prompt_id": "first", "human_label": "1", "video_a_uid": "a", "video_b_uid": "b",
         "model_a": "lavie", "model_b": "modelscope"},
        {"prompt_id": "second", "human_label": "1", "video_a_uid": "c", "video_b_uid": "d",
         "model_a": "cogvideo", "model_b": "videocraft"},
    ]
    report = population(pairs, videos, dict.fromkeys(METHODS, 0.), resamples=100, seed=1)
    origin, repair = (report["methods"][m] for m in METHODS[:2])
    assert origin["expected_pairs"] == 2 and origin["defined_pairs"] == 1
    assert origin["tie_aware_agreement_full_denominator"] == .5
    assert origin["tie_aware_agreement_defined_pairs"] == 1.
    assert repair["paired_delta_vs_official_full_denominator"] == .5
    assert repair["common_defined_pairs_with_official"] == 1
    assert repair["paired_delta_common_defined_pairs"] == 0.
    assert origin["paper_alignment"]["pearson_n4_defined_pairs"] is None
    unavailable = [r for r in origin["paper_alignment"]["models"] if r["generator"] == "cogvideo"][0]
    assert unavailable["metric_win_ratio_full_population"] is None
    assert unavailable["full_population_lower_bound"] == 0.
    assert unavailable["full_population_upper_bound"] == 1.


def test_paper_win_ratio_counts_ties_as_half_wins():
    from scripts.object_color_natural_report import paper_win_ratios
    videos = {uid: {"scores": {"official": {"status": "succeeded", "score": 1.}}} for uid in ("a", "b")}
    pairs = [{"video_a_uid": "a", "video_b_uid": "b", "model_a": "lavie",
              "model_b": "modelscope", "human_label": "0.5"}]
    report = paper_win_ratios(pairs, videos, "official")
    assert report["full_pair_population_covered"]
    assert all(r["metric_win_ratio_full_population"] == .5 for r in report["models"])
    assert report["pearson_n4_defined_pairs"] is None
