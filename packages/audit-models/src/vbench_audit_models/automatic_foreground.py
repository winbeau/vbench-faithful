"""MobileSAM object proposals for frames with no COCO foreground detection.

No category, box, or mask from intervention construction is accepted here.
"""
from __future__ import annotations

import numpy as np


def select_automatic_mask(records, shape, policy):
    candidates, diagnostics = [], []
    for index, record in enumerate(records):
        mask = np.asarray(record['segmentation'])
        if mask.shape != shape or not np.isin(mask, (0, 1)).all():
            raise ValueError('automatic mask must be native binary pixels')
        fraction = float(mask.mean())
        edges = sum(bool(edge.any()) for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))
        eligible = (policy['minimum_area'] <= fraction <= policy['maximum_area']
                    and edges <= policy['maximum_touched_edges'])
        diagnostics.append({'index': index, 'area_fraction': fraction, 'touched_edges': edges,
                            'predicted_iou': float(record['predicted_iou']),
                            'stability_score': float(record['stability_score']), 'eligible': bool(eligible),
                            'bbox': record['bbox']})
        if eligible:
            candidates.append(index)
    if not candidates:
        return np.zeros(shape, np.uint8), {'selected_index': None, 'candidates': diagnostics}
    selected = min(candidates, key=lambda i: (-diagnostics[i]['area_fraction'],
        -diagnostics[i]['stability_score'], -diagnostics[i]['predicted_iou'], i))
    return np.asarray(records[selected]['segmentation'], dtype=np.uint8), {'selected_index': selected, 'candidates': diagnostics}


def build_automatic_generator(model, policy):
    from mobile_sam import SamAutomaticMaskGenerator
    return SamAutomaticMaskGenerator(model, points_per_side=policy['points_per_side'],
        points_per_batch=policy['points_per_batch'], pred_iou_thresh=policy['pred_iou_threshold'],
        stability_score_thresh=policy['stability_threshold'], crop_n_layers=0,
        min_mask_region_area=0, output_mode='binary_mask')


class SamFallbackForegroundProvider:
    def __init__(self, provider, policy):
        self.provider, self.policy = provider, dict(policy)
        self.generator = build_automatic_generator(provider.predictor.model, policy)
        self.provenance = {**provider.provenance, 'automatic_fallback': dict(policy),
                           'fallback_trigger': 'zero foreground pixels from COCO80 on this actual frame',
                           'fallback_selection': 'largest eligible confident mask; no construction target/box/mask or clean-mask propagation'}
        self.last_diagnostics = None

    def masks_for(self, frames):
        masks = self.provider.masks_for(frames)
        base = self.provider.last_diagnostics
        images = frames.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        fallback = []
        for index, (frame, mask) in enumerate(zip(images, masks)):
            if mask.any():
                continue
            records = self.generator.generate(np.ascontiguousarray(frame))
            selected, trace = select_automatic_mask(records, frame.shape[:2], self.policy)
            masks[index] = selected
            fallback.append({'frame': index, **trace})
        self.last_diagnostics = {**base,
            'num_empty_before_fallback': base['num_empty_foreground_frames'],
            'num_empty_foreground_frames': int((masks.sum((1, 2)) == 0).sum()),
            'foreground_fraction': masks.mean((1, 2)).tolist(),
            'automatic_fallback_frames': fallback}
        return masks
