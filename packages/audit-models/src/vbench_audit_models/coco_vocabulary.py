"""Explicit COCO vocabulary extension using the existing box/SAM weights.

Kept separate so the earlier frozen 19-class adapter remains reproducible.
This module contains no metric formulas and reads no construction artifacts.
"""
from __future__ import annotations

from pathlib import Path

from .foreground import CocoSubjectBoxDetector, MobileSamForegroundProvider, load_mobile_sam_predictor


class CocoVocabularyBoxDetector(CocoSubjectBoxDetector):
    def __init__(self, checkpoint, *, classes, **kwargs):
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        categories = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta['categories']
        supported = {name: index for index, name in enumerate(categories) if index and name != 'N/A'}
        if classes != supported or len(classes) != 80:
            raise ValueError('this experiment requires the complete pinned COCO 80-class vocabulary')
        super().__init__(checkpoint, **kwargs)
        self.classes = dict(classes)
        self.provenance = {**self.provenance, 'vocabulary': self.classes, 'vocabulary_size': len(self.classes),
                           'adapter': 'coco80_development_extension', 'base_weights_changed': False}

    def boxes_for(self, frames, phrase=None):
        import torch
        if phrase is not None and phrase not in self.classes:
            raise ValueError(f'unsupported scoring object class: {phrase}')
        allowed = [self.classes[phrase]] if phrase is not None else list(self.classes.values())
        names = {index: name for name, index in self.classes.items()}
        result = []
        strict = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            with torch.inference_mode():
                for start in range(0, len(frames), 4):
                    images = [frame.to(self.device, dtype=torch.float32)/255 for frame in frames[start:start+4]]
                    for prediction in self.model(images):
                        wanted = torch.as_tensor(allowed, device=prediction['labels'].device)
                        keep = torch.isin(prediction['labels'], wanted) & (prediction['scores'] >= self.threshold)
                        label_ids = prediction['labels'][keep].cpu().tolist()
                        result.append({'boxes': prediction['boxes'][keep].cpu().tolist(),
                                       'scores': prediction['scores'][keep].cpu().tolist(),
                                       'label_ids': label_ids, 'labels': [names[index] for index in label_ids]})
        finally:
            torch.use_deterministic_algorithms(strict, warn_only=warn)
        return result


def build_coco80_provider(localizer, vocabulary, *, device):
    from vbench_audit_core.inputs import sha256_file
    sam_path = Path(localizer['sam_checkpoint'])
    if sha256_file(sam_path) != localizer['sam_sha256']:
        raise ValueError('MobileSAM checkpoint SHA256 mismatch')
    detector = CocoVocabularyBoxDetector(Path(localizer['detector_checkpoint']), classes=vocabulary['classes'],
        expected_sha256=localizer['detector_sha256'], device=device,
        threshold=float(localizer['threshold']), size=int(localizer['size']))
    provider = MobileSamForegroundProvider(detector, load_mobile_sam_predictor(sam_path, device),
                                          weights_sha256=localizer['sam_sha256'])
    provider.provenance = {**provider.provenance, 'classes': dict(detector.classes),
                           'vocabulary_size': len(detector.classes), 'development_candidate': vocabulary['name']}
    return provider
