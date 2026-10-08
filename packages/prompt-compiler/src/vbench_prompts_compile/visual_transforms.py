"""Deterministic pixel masks and box mirrors, independent of detector inference.

Mask geometry is not a visibility annotation. Missing localization or unavailable
controls are explicit failures, never converted into evidence that an object is
absent. The zero level is an exact no-op, including for unlocalized objects.
"""
from __future__ import annotations
import math


def rectangle(box, width, height):
    x0, y0, x1, y1 = box
    return [max(0, min(width, math.floor(x0))), max(0, min(height, math.floor(y0))),
            max(0, min(width, math.ceil(x1))), max(0, min(height, math.ceil(y1)))]


def box_mask(shape, boxes, level=1.0):
    import numpy as np
    if not 0 <= level <= 1:
        raise ValueError('occlusion fraction outside [0,1]')
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    rects = []
    if level == 0:
        return mask, rects
    for box in boxes:
        x0, y0, x1, y1 = rectangle(box, width, height)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        mw, mh = math.ceil(w * level), math.ceil(h * level)
        x, y = x0 + (w - mw) // 2, y0 + (h - mh) // 2
        rect = [x, y, x + mw, y + mh]
        mask[y:y + mh, x:x + mw] = True
        rects.append(rect)
    return mask, rects


def background_mask(target_mask, forbidden, rects, *, stride=16):
    """Translate the entire target mask to an empty detected region, same area.

    Search a fixed grid plus far edges; no clipping, reshaping or resizing. A
    conservative empty bounding rectangle is required. Failure stays explicit.
    """
    import numpy as np
    ys, xs = np.where(target_mask)
    if not len(xs):
        return target_mask.copy(), []
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    w, h = x1 - x0, y1 - y0
    height, width = target_mask.shape
    integral = np.pad(forbidden.astype('int64'), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = sorted(set(range(0, height - h + 1, stride)) | {height - h})
    xs = sorted(set(range(0, width - w + 1, stride)) | {width - w})
    for y in ys:
        for x in xs:
            occupied = integral[y+h, x+w] - integral[y, x+w] - integral[y+h, x] + integral[y, x]
            if occupied:
                continue
            mask = np.zeros_like(target_mask)
            mask[y:y+h, x:x+w] = target_mask[y0:y1, x0:x1]
            if int(mask.sum()) != int(target_mask.sum()):
                raise AssertionError('background control changed patch area')
            return mask, [[a+x-x0, b+y-y0, c+x-x0, d+y-y0] for a,b,c,d in rects]
    return None, []


def patch_frame(frame, detections, required, control, level):
    import numpy as np
    target = [r['box'] for r in detections if r['label'] == required[0]]
    non_target = [r['box'] for r in detections if r['label'] == required[1]]
    mask, rects = box_mask(frame.shape, target, level)
    full_target, _ = box_mask(frame.shape, target)
    other_mask, _ = box_mask(frame.shape, non_target)
    details = {'control': control, 'level': level, 'target_boxes': target,
               'non_target_boxes': non_target, 'visibility_verified': False}
    if level == 0:
        details.update(status='ok', rectangles=[], mask_pixels=0, changed_pixels=0,
                       outside_unchanged=True, target_box_coverage=0.0, non_target_box_coverage=0.0)
        return frame.copy(), details
    if not full_target.any():
        return None, {**details, 'status': 'target_unlocalized'}
    if control == 'background':
        forbidden, _ = box_mask(frame.shape, [r['box'] for r in detections])
        mask, rects = background_mask(mask, forbidden, rects)
        if mask is None:
            return None, {**details, 'status': 'equal_area_background_unavailable'}
    elif control == 'non_target':
        mask, rects = box_mask(frame.shape, non_target, level)
        if not mask.any():
            return None, {**details, 'status': 'non_target_unlocalized'}
        if np.any(mask & full_target):
            return None, {**details, 'status': 'non_target_control_overlaps_target'}
    elif control != 'target':
        raise ValueError('unknown occlusion control')
    patched = frame.copy()
    patched[mask] = 128
    outside_unchanged = np.array_equal(frame[~mask], patched[~mask])
    if not outside_unchanged:
        raise AssertionError('changed pixels outside mask')
    details.update(status='ok', rectangles=rects, mask_pixels=int(mask.sum()),
        changed_pixels=int(np.any(frame != patched, axis=-1).sum()), outside_unchanged=True,
        target_box_coverage=float((mask & full_target).sum() / full_target.sum()),
        non_target_box_coverage=float((mask & other_mask).sum() / other_mask.sum()) if other_mask.any() else None)
    if control == 'target' and level == 1 and details['target_box_coverage'] != 1.0:
        raise AssertionError('full endpoint failed to cover localized target boxes')
    return patched, details


def mirror_detections(frames, size, axis):
    width, height = size
    if axis not in {'hflip', 'vflip'}:
        raise ValueError('unknown mirror axis')
    result = []
    for frame in frames:
        output = []
        for item in frame:
            x0, y0, x1, y1 = item['box']
            box = [width-x1, y0, width-x0, y1] if axis == 'hflip' else [x0, height-y1, x1, height-y0]
            output.append({**item, 'box': box})
        result.append(output)
    return result
