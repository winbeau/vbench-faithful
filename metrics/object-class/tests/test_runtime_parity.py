import inspect

from object_class.metric import evaluate_batch


def test_shared_batch_signature():
    assert list(inspect.signature(evaluate_batch).parameters) == ["backend", "videos", "metadata", "device", "config"]
