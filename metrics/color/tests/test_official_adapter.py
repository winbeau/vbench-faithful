from color.backends.vbench import evaluate_batch


def test_official_adapter_is_explicitly_deferred(tmp_path):
    result = evaluate_batch("vbench", [tmp_path / "video.mp4"], {}, None, {})[0]
    assert result.status == "not_implemented"
    assert result.score is None
    assert "compute_color" in (result.error or "")
