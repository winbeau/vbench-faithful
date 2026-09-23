"""Image-coordinate relocation prompts from single-video motion hypotheses.

The shifted region proposes where to ask SAM, never a target mask or identity.
Wrong hypotheses and out-of-view proposals must remain explicit downstream.
"""
from __future__ import annotations

import cv2
import numpy as np


def relocation_prompts(source_mask, hypotheses):
    mask = np.asarray(source_mask)
    if mask.ndim != 2 or not np.isin(mask, (0, 1)).all() or min(mask.shape) < 2:
        raise ValueError('nondegenerate binary source mask required')
    h, w = mask.shape
    records = []
    for i, proposal in enumerate(hypotheses):
        d = np.asarray(proposal['displacement_pixels'], float)
        if d.shape != (2,) or not np.isfinite(d).all():
            raise ValueError('finite translation hypothesis required')
        moved = cv2.warpAffine(mask.astype(np.uint8), np.array([[1.,0.,d[0]],[0.,1.,d[1]]]),
                              (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        yy, xx = np.nonzero(moved)
        result = {'source_hypothesis': i, 'origin': proposal['origin'],
                  'displacement_pixels': d.tolist(), 'visible_prompt_pixels': len(xx),
                  'status': 'proposal_only', 'box_xyxy': None, 'point_xy': None}
        if not len(xx):
            result['status'] = 'out_of_view_or_empty'
        else:
            # Use a visible interior point; do not clamp an out-of-view source
            # point onto a boundary and call it foreground. Padding handles a
            # whole-frame source mask without an infinite distance transform.
            distance = cv2.distanceTransform(np.pad(moved,1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
            y, x = np.unravel_index(np.argmax(distance), moved.shape)
            result.update(box_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                          point_xy=[int(x),int(y)])
        records.append(result)
    return records
