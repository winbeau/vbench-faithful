import inspect

from background_consistency.metric import evaluate_batch


def test_shared_batch_signature():
    assert list(inspect.signature(evaluate_batch).parameters) == ["backend", "videos", "metadata", "device", "config"]


def test_public_entry_pins_inference_before_loading_the_model(tmp_path, monkeypatch):
    from background_consistency import inference, metric
    from vbench_audit_core.schemas import VideoResult
    called = []
    def setup(seed):
        called.append(seed)
        return {"seed": seed, "cudnn_tf32": False}
    def backend(*args):
        assert called == [0]
        return [VideoResult(str(tmp_path/"a.mp4"), "succeeded", score=.8)]
    monkeypatch.setattr(inference, "configure_inference", setup)
    monkeypatch.setattr(metric.vbench, "evaluate_batch", backend)
    result = evaluate_batch("vbench", [tmp_path/"a.mp4"], {}, "cpu", {"model": {"clip": {"checkpoint": "test"}}})
    assert result[0].metric["inference"]["cudnn_tf32"] is False
