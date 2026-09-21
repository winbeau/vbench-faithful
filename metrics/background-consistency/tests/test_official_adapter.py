from background_consistency.backends.vbench import evaluate_batch


def test_official_adapter_requires_local_assets_without_fabricating_scores(tmp_path):
    video = tmp_path / "video.mp4"
    result = evaluate_batch("vbench", [video], {}, None, {})[0]
    assert result.status == "failed"
    assert result.score is None
    assert "checkpoint" in (result.error or "")
