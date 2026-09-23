from dataclasses import asdict
import json
from pathlib import Path

import pytest

from dynamic_degree import cli
from dynamic_degree.trajectory import TrajectoryConfig


@pytest.fixture
def assets(tmp_path, monkeypatch):
    source = tmp_path / "tracker"
    (source / "cotracker").mkdir(parents=True)
    (source / "cotracker/predictor.py").write_text("# mock source, never imported\n")
    checkpoint = tmp_path / "tracker.pth"
    checkpoint.write_bytes(b"mock checkpoint, never loaded")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(TrajectoryConfig())))
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("video_000.mp4", "video_002.mp4", "video_003.mp4"):
        (videos / name).write_bytes(b"mock video, never decoded")
    # Invalid metadata must never be read by the video-only candidate.
    (videos / "metadata.json").write_text("this is not json")
    monkeypatch.setattr(cli, "check_cuda", lambda ids: {"requested_gpu_ids": ids})
    monkeypatch.setattr(cli, "release_cuda_resources", lambda: None)
    monkeypatch.setattr(cli, "set_seed", lambda _: None)
    monkeypatch.setattr(cli, "find_metadata", lambda *args: pytest.fail("candidate tried to read metadata"))
    options = ["--audit-variant", "trajectory", "--trajectory-config", str(config),
               "--tracker-root", str(source), "--tracker-weight", str(checkpoint),
               "--output", str(tmp_path / "output")]
    return videos, options, tmp_path / "output"


def test_candidate_audit_is_video_only_and_has_code_model_input_provenance(assets, monkeypatch):
    videos, options, output = assets
    from dynamic_degree.backends import vbench
    monkeypatch.setattr(vbench, "verify_upstream", lambda _: pytest.fail("audit candidate needs no upstream"))
    monkeypatch.setattr(cli, "weight_path", lambda: pytest.fail("audit candidate needs no RAFT"))
    calls = []

    def fake(backend, paths, metadata, gpu_ids, weight, level, seed, **kwargs):
        calls.append(kwargs)
        assert backend == "audit" and all(not x["prompt"] for x in metadata.values())
        assert gpu_ids == [0, 2]
        return [{"video": str(p), "status": "succeeded", "score": 0.1, "backend": backend} for p in paths]

    monkeypatch.setattr(cli, "evaluate_backend_sharded", fake)
    assert cli.main(["--audit", "--video-dir", str(videos), "--gpu", "0,2", *options]) == 0
    assert calls[0]["trajectory_options"]["config"] == asdict(TrajectoryConfig())
    run = json.loads(next(output.glob("dynamic-degree/audit/*/run.json")).read_text())
    assert run["metadata_used"] is False and run["metadata_sha256"] is None
    assert len(run["input_sha256"]) == 3
    assert run["source_file_sha256"]["metrics/dynamic-degree/src/dynamic_degree/trajectory.py"]
    assert run["tracker_source_sha256"]["cotracker/predictor.py"]
    assert run["tracker_weight_sha256"] and run["uv_lock_sha256"]


def test_candidate_both_routes_options_only_to_repair_and_preserves_failure(assets, monkeypatch):
    videos, options, output = assets
    from dynamic_degree.backends import vbench
    monkeypatch.setattr(vbench, "verify_upstream", lambda _: None)
    monkeypatch.setattr(cli, "weight_path", lambda: videos / "video_000.mp4")
    monkeypatch.setattr(cli, "environment_record", lambda *args: {"official_marker": True})
    calls = []

    def fake(backend, paths, metadata, gpu_ids, weight, level, seed, **kwargs):
        calls.append((backend, kwargs))
        return [{"video": str(p), "backend": backend,
                 "status": "succeeded" if backend == "vbench" else "insufficient_evidence",
                 "score": 1.0 if backend == "vbench" else None} for p in paths]

    monkeypatch.setattr(cli, "evaluate_backend_sharded", fake)
    assert cli.main(["--both", "--video", str(videos / "video_000.mp4"), *options]) == 1
    assert calls[0] == ("vbench", {}) and "trajectory_options" in calls[1][1]
    summaries = {p.parent.parent.name: json.loads(p.read_text()) for p in output.glob("dynamic-degree/*/*/summary.json")}
    assert summaries["vbench"]["status"] == "complete"
    assert summaries["audit"]["status"] == "failed" and summaries["audit"]["aggregate"] is None
    dirs = list(output.glob("dynamic-degree/*/*"))
    assert dirs[0].name == dirs[1].name


def test_missing_candidate_config_fails_before_cuda_or_output(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setattr(cli, "check_cuda", lambda _: pytest.fail("invalid input reached CUDA"))
    assert cli.main(["--audit", "--audit-variant", "trajectory", "--video", str(video)]) == 2


def test_local_reliability_variant_routes_its_own_config_and_formula(assets, monkeypatch):
    from dynamic_degree.local_trajectory import LocalTrajectoryConfig

    videos, old_options, output = assets
    options = ["local-trajectory" if x == "trajectory" else x for x in old_options]
    config_path = Path(options[options.index("--trajectory-config") + 1])
    config_path.write_text(json.dumps(asdict(LocalTrajectoryConfig())))

    def fake(backend, paths, metadata, gpu_ids, weight, level, seed, **kwargs):
        assert kwargs["trajectory_options"]["variant"] == "local-trajectory"
        assert kwargs["trajectory_options"]["config"]["window_seconds"] == 1.0
        return [{"video": str(p), "status": "succeeded", "score": .1} for p in paths]

    monkeypatch.setattr(cli, "evaluate_backend_sharded", fake)
    assert cli.main(["--audit", "--video-dir", str(videos), *options]) == 0
    summary = json.loads(next(output.glob("dynamic-degree/audit/*/summary.json")).read_text())
    assert summary["formula_version"] == "local-trajectory-reliability-v1-observed-motion-lower-bound"
