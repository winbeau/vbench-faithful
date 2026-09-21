"""Automatic per-frame boxes and MobileSAM masks for a separate experiment.

This provider does not fabricate human confirmations or consume construction
masks. All detections of the official subject class are retained as a union;
misses stay empty. It is not the human-prompted provider's protocol.
"""
from __future__ import annotations

from pathlib import Path

from vbench_audit_models.foreground import COCO_SUBJECTS, CocoSubjectBoxDetector


class AutomaticMobileSamSubjectMaskProvider:
    def __init__(self, detector, predictor, *, weights_sha256: str):
        self.detector, self.predictor = detector, predictor
        self.provenance = {'role': 'scoring', 'family': 'mobilesam', 'model': 'vit_t',
                           'weights_sha256': weights_sha256, 'box_detector': detector.provenance,
                           'prompt_source': 'automatic_per_frame_detection', 'human_confirmed': False,
                           'construction_masks_reused': False, 'instances': 'union_all_target_class',
                           'missing_frames': 'empty_mask_no_interpolation', 'multimask_output': False}
        self.last_diagnostics = None

    def masks_for(self, video: Path, frames, phrase: str):
        import numpy as np
        import torch
        from .models import SubjectMasks, native_rgb_uint8

        if phrase not in COCO_SUBJECTS:
            raise ValueError(f'unsupported official subject: {phrase}')
        frames = native_rgb_uint8(frames)
        detections = self.detector.boxes_for(frames, phrase)
        if len(detections) != len(frames):
            raise ValueError('detector frame count mismatch')
        images = frames.cpu().permute(0, 2, 3, 1).numpy()
        masks, traces = [], []
        with torch.inference_mode():
            for image, detection in zip(images, detections):
                boxes = np.asarray(detection['boxes'], dtype=np.float32).reshape(-1, 4)
                height, width = image.shape[:2]
                if (not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any()
                        or (boxes < 0).any() or (boxes[:, [0, 2]] > width).any()
                        or (boxes[:, [1, 3]] > height).any()):
                    raise ValueError('invalid detector box')
                union = np.zeros((height, width), dtype=np.uint8)
                qualities = []
                if len(boxes):
                    self.predictor.set_image(np.ascontiguousarray(image), image_format='RGB')
                    for box in boxes:
                        output, quality, _ = self.predictor.predict(box=box, multimask_output=False)
                        output = np.asarray(output)
                        if output.shape != (1, height, width) or not np.isin(output, [0, 1]).all():
                            raise ValueError('SAM must return one native binary mask per box')
                        union |= output[0].astype(np.uint8)
                        qualities.append(float(np.asarray(quality).reshape(-1)[0]))
                masks.append(torch.from_numpy(union))
                traces.append({**detection, 'sam_predicted_iou': qualities})
        stacked = torch.stack(masks).unsqueeze(1).float()
        present = stacked.flatten(2).sum(-1) > 0
        self.last_diagnostics = {'detections': traces, 'num_frames': len(masks),
                                 'num_missing_frames': int((~present).sum()),
                                 'num_multi_instance_frames': sum(len(r['boxes']) > 1 for r in traces)}
        return SubjectMasks(stacked, present, phrase, 'automatic-box-mobilesam-union')
