from color.backends.vbench import evaluate_batch


def test_official_adapter_requires_explicit_local_configuration(tmp_path):
    result = evaluate_batch("vbench", [tmp_path / "video.mp4"], {}, None, {})[0]
    assert result.status == "failed"
    assert result.score is None
    assert "grit" in (result.error or "")
