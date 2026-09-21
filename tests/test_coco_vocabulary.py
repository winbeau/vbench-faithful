import pytest
import torch

from vbench_audit_models.coco_vocabulary import CocoVocabularyBoxDetector
from vbench_audit_models.foreground import COCO_SUBJECTS


def fake_detector(*, fail=False):
    detector = CocoVocabularyBoxDetector.__new__(CocoVocabularyBoxDetector)
    detector.device, detector.threshold = 'cpu', .8
    detector.classes = {'person': 1, 'dog': 18, 'bed': 65}

    def model(images):
        assert not torch.are_deterministic_algorithms_enabled()
        if fail:
            raise RuntimeError('test detector failure')
        return [{'boxes': torch.tensor([[1, 1, 6, 8], [2, 2, 7, 8], [1, 1, 5, 5], [0, 0, 9, 9]], dtype=torch.float32),
                 'labels': torch.tensor([65, 1, 18, 0]), 'scores': torch.tensor([.95, .7, .9, .99])} for _ in images]
    detector.model = model
    return detector


def test_expansion_keeps_furniture_without_relaxing_confidence_or_background_filter():
    detector = fake_detector()
    results = detector.boxes_for(torch.zeros(5, 3, 10, 10, dtype=torch.uint8), None)
    assert len(results) == 5
    assert all(row['labels'] == ['bed', 'dog'] for row in results)
    assert all(row['label_ids'] == [65, 18] for row in results)
    assert all(len(row['boxes']) == 2 for row in results)
    # The frozen subject vocabulary remains unchanged for prior experiments.
    assert 'bed' not in COCO_SUBJECTS and len(COCO_SUBJECTS) == 19


def test_explicit_object_filter_and_unsupported_structure_are_distinct():
    detector = fake_detector(); frames = torch.zeros(1, 3, 10, 10, dtype=torch.uint8)
    assert detector.boxes_for(frames, 'bed')[0]['labels'] == ['bed']
    with pytest.raises(ValueError, match='unsupported scoring object class: tower'):
        detector.boxes_for(frames, 'tower')


def test_expanded_box_failure_restores_determinism_settings():
    strict = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(RuntimeError, match='test detector failure'):
            fake_detector(fail=True).boxes_for(torch.zeros(1, 3, 10, 10, dtype=torch.uint8), None)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn)
