from object_class.metric import evaluate_batch


def test_batch_contract_has_one_explicit_null_result_per_video(tmp_path):
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    results = evaluate_batch("audit", videos, {}, None, {"runtime": {"audit_variant": "diagnostic"}})
    assert [item.video for item in results] == [str(item) for item in videos]
    assert all(item.status == "not_implemented" and item.score is None for item in results)
