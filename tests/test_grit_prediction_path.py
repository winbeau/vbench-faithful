"""Scoring consumes original predictions without constructing display images."""

from types import SimpleNamespace

import numpy as np

from vbench_audit_models.grit import predict_instances


def test_prediction_path_preserves_input_and_return_identity():
    frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    prediction = {"instances": object()}
    calls = []

    def predictor(image):
        calls.append(image)
        return prediction

    def discarded_visualization(*args):
        raise AssertionError("scoring must not draw a visualization")

    model = SimpleNamespace(demo=SimpleNamespace(
        predictor=predictor, run_on_image=discarded_visualization))
    assert predict_instances(model, frame) is prediction
    assert len(calls) == 1 and calls[0] is frame


def test_shared_caption_and_instance_consumers_do_not_reload_or_infer(tmp_path):
    from vbench_audit_core.run_inference_cache import RunInferenceCache
    from vbench_audit_models.grit import GritEvidenceModel

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = {"evidence": {"status": "succeeded", "objects": [{"text": "car"}]},
               "captions": [["car", [0, 0, 1, 1], ["car"]]]}
    producer = GritEvidenceModel.__new__(GritEvidenceModel)
    producer.cache = RunInferenceCache(tmp_path, {"task": "ObjectDet"})
    producer._predict = lambda image: payload
    assert producer.detect(frame) == payload["evidence"]

    def forbidden(image):
        raise AssertionError("another dimension reloaded or reran an identical predictor")

    consumer = GritEvidenceModel.__new__(GritEvidenceModel)
    consumer.cache = RunInferenceCache(tmp_path, {"task": "ObjectDet"})
    consumer._predict = forbidden
    assert consumer.caption(frame) == payload["captions"]
    assert consumer.cache.hits == 1 and consumer.cache.misses == 0
