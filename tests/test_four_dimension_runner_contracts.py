from pathlib import Path
import os

from vbench_audit_core.contracts import load_model_config, run_batch_contract, run_batch_sharded
from vbench_audit_core.schemas import VideoResult


def shard_evaluator(backend, items, metadata, device, config):
    assert backend == "audit"
    assert device == "cuda:0"
    assert "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    return [VideoResult(video=str(item), status="succeeded", score=float(index)) for index, item in enumerate(items)]


def test_common_batch_contract_restores_order_and_preserves_metadata(tmp_path):
    videos = [tmp_path / "video_000.mp4", tmp_path / "video_001.mp4"]
    seen = {}

    def mock_evaluator(backend, items, metadata, device, config):
        seen.update(backend=backend, metadata=metadata, device=device, config=config)
        return [
            VideoResult(video=str(items[1]), status="succeeded", score=0.2),
            VideoResult(video=str(items[0]), status="succeeded", score=0.1),
        ]

    results = run_batch_contract(
        mock_evaluator,
        "audit",
        videos,
        {videos[0].name: {"prompt": "a cat"}},
        device="cpu",
        config={"runtime": {"audit_variant": "diagnostic"}},
    )
    assert [item.video for item in results] == [str(video) for video in videos]
    assert [item.score for item in results] == [0.1, 0.2]
    assert seen == {
        "backend": "audit",
        "metadata": {videos[0].name: {"prompt": "a cat"}},
        "device": "cpu",
        "config": {"runtime": {"audit_variant": "diagnostic"}},
    }


def test_common_batch_contract_rejects_missing_video(tmp_path):
    video = tmp_path / "video_000.mp4"

    def bad_evaluator(*_args):
        return [VideoResult(video="elsewhere.mp4", status="succeeded", score=1.0)]

    try:
        run_batch_contract(bad_evaluator, "audit", [video], {}, config={})
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("missing batch coverage must fail")


def test_common_four_shard_runner_deduplicates_and_restores_full_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,GPU-test-uuid,2,1")
    videos = [tmp_path / f"video_{index:03d}.mp4" for index in range(8)]
    results = run_batch_sharded(shard_evaluator, "audit", videos, {}, [0, 1, 2, 3], label="four-dimension-test")
    assert [item.video for item in results] == [str(video) for video in videos]
    assert len({item.video for item in results}) == len(videos)
    assert all(item.status == "succeeded" for item in results)
    assert {item.metric["worker_device"]["cuda_visible_devices"] for item in results} == {"5", "GPU-test-uuid", "2", "1"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,GPU-test-uuid,2,1"


def test_batch_rejects_overlapping_video_uids(tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    def duplicate_uids(*_args):
        return [VideoResult(str(p), "succeeded", 0., {"video_uid": "same"}) for p in videos]
    import pytest
    with pytest.raises(ValueError, match="video_uid duplicates"):
        run_batch_contract(duplicate_uids, "audit", videos, {})


def test_model_config_contract_is_toml_only_and_preserves_nested_tables(tmp_path):
    config_path = tmp_path / "models.toml"
    config_path.write_text("[models]\nclip = '/weights/local.pt'\n[dimensions.color]\nhead = 'densecap'\n", encoding="utf-8")
    assert load_model_config(config_path) == {
        "models": {"clip": "/weights/local.pt"},
        "dimensions": {"color": {"head": "densecap"}},
    }
