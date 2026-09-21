"""Development probe: independent caption objects augment any existing mask.

Frozen caption v1/v2 providers remain unchanged. This helper consumes only
descriptions/boxes predicted from the supplied actual image, never targets or
masks from another video variant. It contains no scoring formula.
"""
from __future__ import annotations

import numpy as np

from .caption_relations import select_caption_box_with_relations


def select_caption_boxes(instances, shape, policy):
    _, trace = select_caption_box_with_relations(instances, shape, policy)
    selected = [row for row in trace['captions'] if row['eligible']]
    return [row['box'] for row in selected], {
        'captions': trace['captions'], 'selected_indices': [row['index'] for row in selected]}


def caption_sam_union(predictor, image, instances, policy, *, base_mask=None):
    """Union every eligible object with an optional mask of this same image.

    Eligibility retains the fixed semantic head, score and box-area rules.
    There is no largest-object restriction, nonempty short-circuit, target-area
    optimization, temporal mask carry-over, or cross-variant propagation.
    """
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('native uint8 RGB image required')
    if base_mask is None:
        mask = np.zeros(image.shape[:2], np.uint8)
    else:
        base_mask = np.asarray(base_mask)
        if base_mask.shape != image.shape[:2] or not np.isin(base_mask, (0, 1)).all():
            raise ValueError('base mask must be binary in the same native geometry')
        mask = base_mask.astype(np.uint8, copy=True)
    boxes, trace = select_caption_boxes(instances, image.shape[:2], policy)
    trace['base_fraction'] = float(mask.mean())
    trace['sam_objects'] = []
    if boxes:
        predictor.set_image(np.ascontiguousarray(image), image_format='RGB')
    for index, box in zip(trace['selected_indices'], boxes):
        prediction, quality, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
        prediction, quality = np.asarray(prediction), np.asarray(quality)
        if (prediction.shape != (1, *image.shape[:2]) or not np.isin(prediction, (0, 1)).all()
                or quality.size != 1 or not np.isfinite(quality).all()):
            raise ValueError('caption-prompted SAM returned invalid native mask or quality')
        piece = prediction[0].astype(np.uint8)
        before = int(mask.sum())
        mask |= piece
        trace['sam_objects'].append({'caption_index': index,
            'sam_predicted_iou': float(quality.ravel()[0]),
            'mask_fraction': float(piece.mean()), 'added_pixels': int(mask.sum()) - before})
    trace['foreground_fraction'] = float(mask.mean())
    return mask, trace
