from background_consistency.backends.repair import evaluate_batch


def test_repair_boundary_does_not_fabricate_score(tmp_path):
    result = evaluate_batch("audit", [tmp_path / "video.mp4"], {}, None, {})[0]
    assert result.metric["variant"] == "patch_frame_calibrated"
    assert result.status == "failed"
    assert result.score is None
