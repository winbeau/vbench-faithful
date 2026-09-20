from background_consistency.backends.vbench import evaluate_batch


def test_official_adapter_is_explicitly_deferred(tmp_path):
    video = tmp_path / "video.mp4"
    result = evaluate_batch("vbench", [video], {}, None, {})[0]
    assert result.status == "not_implemented"
    assert result.score is None
    assert "compute_background_consistency" in (result.error or "")
