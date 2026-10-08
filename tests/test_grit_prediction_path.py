"""Scoring consumes original predictions without constructing display images."""

from types import SimpleNamespace

import numpy as np
import pytest

from vbench_audit_models.grit import predict_instances


def object_roi():
    decoder = SimpleNamespace(training=False, textual=object(), tokenizer=object(),
                              beamsearch_decode=object(), begin_token_id=101, padding_idx=0)
    calls = []

    def forward(*args, **kwargs):
        import torch
        calls.append((args, kwargs))
        return [{"boxes": torch.tensor([[1., 2., 3., 4.]]), "labels": ["person"]}]

    return SimpleNamespace(training=False, test_task="ObjectDet", beam_size=1,
        text_decoder=decoder, text_decoder_det=SimpleNamespace(**vars(decoder)),
        _forward_box=forward), calls


def test_same_frame_reuse_keeps_outputs_independent_and_consumes_once():
    from vbench_audit_models.grit import SameFrameObjectDet
    roi, calls = object_roi()
    reuse = SameFrameObjectDet(roi)
    features, proposals = {}, []
    first = reuse(features, proposals)
    # Detectron heads/postprocessing mutate returned Instances in place.
    first[0]["boxes"].mul_(2)
    first[0]["labels"][0] = "changed"
    second = reuse(features, proposals, det_box=True)
    assert second[0]["boxes"].tolist() == [[1., 2., 3., 4.]]
    assert second[0]["labels"] == ["person"]
    assert len(calls) == 1 and reuse.reuses == 1
    reuse(features, proposals, det_box=True)
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["features", "proposals", "frame", "densecap",
    "token", "weights", "beamsearch", "training", "targets"])
def test_non_equivalent_or_different_frame_heads_keep_original_calls(change):
    from vbench_audit_models.grit import SameFrameObjectDet
    roi, calls = object_roi()
    reuse = SameFrameObjectDet(roi)
    features, proposals = {}, []
    reuse(features, proposals)
    targets = None
    if change == "features":
        features = {}
    elif change == "proposals":
        proposals = []
    elif change == "frame":
        reuse.reset()
    elif change == "densecap":
        roi.test_task = "DenseCap"
    elif change == "token":
        roi.text_decoder_det.begin_token_id = 104
    elif change == "weights":
        roi.text_decoder_det.textual = object()
    elif change == "beamsearch":
        roi.text_decoder_det.beamsearch_decode = object()
    elif change == "training":
        roi.training = True
    elif change == "targets":
        targets = []
    reuse(features, proposals, targets, det_box=True)
    assert len(calls) == 2 and reuse.reuses == 0


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
