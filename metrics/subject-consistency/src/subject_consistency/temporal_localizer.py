"""Clip-local box tracks followed by fresh MobileSAM segmentation on each frame.

Only boxes from the actual input clip enter tracking. No construction mask,
reference clip, DINO score, or previous call participates. Interpolated prompts
are not interpolated masks: SAM must supply current-frame evidence.
"""
from __future__ import annotations

import math
from statistics import median


def box_iou(left, right):
    width = max(0., min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0., min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return intersection / union if union > 0 else 0.


def temporal_box_prompts(detections, *, minimum_iou=.2, max_gap=2, smooth_radius=1):
    """Keep all target tracks; bridge bounded gaps, then median smooth prompts."""
    if not 0 < minimum_iou <= 1 or max_gap < 0 or smooth_radius < 0:
        raise ValueError('invalid tracking parameters')
    tracks = []
    for frame, detection in enumerate(detections):
        boxes = [tuple(float(v) for v in box) for box in detection['boxes']]
        if any(len(b) != 4 or not all(math.isfinite(v) for v in b)
               or min(b) < 0 or b[2] <= b[0] or b[3] <= b[1] for b in boxes):
            raise ValueError('invalid target box')
        candidates = []
        for track_id, track in enumerate(tracks):
            last = max(track)
            if frame - last > max_gap + 1:
                continue
            for box_id, box in enumerate(boxes):
                overlap = box_iou(track[last], box)
                if overlap >= minimum_iou:
                    candidates.append((-overlap, track_id, box_id))
        used_tracks, used_boxes = set(), set()
        for _, track_id, box_id in sorted(candidates):
            if track_id in used_tracks or box_id in used_boxes:
                continue
            tracks[track_id][frame] = boxes[box_id]
            used_tracks.add(track_id); used_boxes.add(box_id)
        for box_id, box in enumerate(boxes):
            if box_id not in used_boxes:
                tracks.append({frame: box})
    result = [[] for _ in detections]
    for track_id, observed in enumerate(tracks):
        prompts = dict(observed)
        indices = sorted(observed)
        for left, right in zip(indices, indices[1:]):
            if right - left - 1 > max_gap:
                continue
            for index in range(left + 1, right):
                fraction = (index - left) / (right - left)
                prompts[index] = tuple(a + fraction * (b - a)
                                       for a, b in zip(observed[left], observed[right]))
        for index in sorted(prompts):
            neighbors = [prompts[j] for j in range(index - smooth_radius, index + smooth_radius + 1)
                         if j in prompts]
            box = tuple(median(b[k] for b in neighbors) for k in range(4))
            result[index].append({'box': list(box), 'track_id': track_id,
                                  'source': 'detected' if index in observed else 'bounded_gap',
                                  'raw_box': list(prompts[index])})
    return result


def clip_anchor_prompts(detections):
    """Choose one prompt frame by detector confidence, independently per clip.

    This reproduces the fixed human-box prompting pattern without a human box
    or a clean reference. Current-image SAM evidence is still required on every
    frame. Absence of any detection remains explicit absence of an anchor.
    """
    eligible = []
    for index, row in enumerate(detections):
        scores = row['scores']
        if len(scores) != len(row['boxes']) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError('invalid detection confidences')
        if scores:
            eligible.append((max(scores), sum(scores), -index))
    if not eligible:
        return [[] for _ in detections]
    anchor = -max(eligible)[2]
    prompts = [{'box': list(box), 'track_id': i, 'source': 'clip_anchor', 'anchor_frame': anchor}
               for i, box in enumerate(detections[anchor]['boxes'])]
    return [[{**p, 'box': list(p['box'])} for p in prompts] for _ in detections]


def direct_anchor_prompts(detections):
    """Preserve current detections; prompt SAM with a clip anchor only on gaps."""
    anchors = clip_anchor_prompts(detections)
    return [[{'box': list(box), 'track_id': i, 'source': 'detected'}
             for i, box in enumerate(row['boxes'])] if row['boxes'] else anchor
            for row, anchor in zip(detections, anchors)]


class TemporalMobileSamSubjectMaskProvider:
    def __init__(self, detector, predictor, *, weights_sha256, minimum_iou=.2,
                 max_gap=2, smooth_radius=1, prompt_policy='bounded_tracks'):
        if prompt_policy not in ('bounded_tracks', 'clip_anchor', 'direct_anchor'):
            raise ValueError('unknown temporal prompt policy')
        self.detector, self.predictor = detector, predictor
        self.options = dict(minimum_iou=minimum_iou, max_gap=max_gap, smooth_radius=smooth_radius)
        self.prompt_policy = prompt_policy
        self.provenance = {'role': 'scoring', 'family': 'mobilesam', 'model': 'vit_t',
                           'weights_sha256': weights_sha256, 'box_detector': detector.provenance,
                           'prompt_source': 'clip_local_target_box_tracks', 'tracking': self.options,
                           'prompt_policy': prompt_policy,
                           'human_confirmed': False, 'construction_masks_reused': False,
                           'cross_variant_state': False, 'instances': 'union_all_target_tracks',
                           'multimask_output': False}
        self.last_diagnostics = None
        self.last_direct_masks = None

    def masks_for(self, video, frames, phrase):
        import numpy as np
        import torch
        from .models import SubjectMasks, native_rgb_uint8

        self.last_diagnostics = self.last_direct_masks = None
        frames = native_rgb_uint8(frames)
        detections = self.detector.boxes_for(frames, phrase)
        if len(detections) != len(frames):
            raise ValueError('detector frame count mismatch')
        if self.prompt_policy == 'clip_anchor':
            prompts = clip_anchor_prompts(detections)
        elif self.prompt_policy == 'direct_anchor':
            prompts = direct_anchor_prompts(detections)
        else:
            prompts = temporal_box_prompts(detections, **self.options)
        images = frames.cpu().permute(0, 2, 3, 1).numpy()
        tracked, direct, traces = [], [], []
        with torch.inference_mode():
            for image, detection, tracked_prompts in zip(images, detections, prompts):
                height, width = image.shape[:2]
                direct_prompts = [dict(box=box, source='detected') for box in detection['boxes']]
                unions = []
                qualities = []
                if direct_prompts or tracked_prompts:
                    self.predictor.set_image(np.ascontiguousarray(image), image_format='RGB')
                cache = {}
                for group in (direct_prompts, tracked_prompts):
                    union = np.zeros((height, width), np.uint8)
                    group_quality = []
                    for prompt in group:
                        box = np.asarray(prompt['box'], np.float32)
                        if (box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any()
                                or box[2] <= box[0] or box[3] <= box[1]
                                or box[[0, 2]].max() > width or box[[1, 3]].max() > height):
                            raise ValueError('box exceeds actual frame')
                        key = tuple(box.tolist())
                        if key not in cache:
                            mask, quality, _ = self.predictor.predict(box=box, multimask_output=False)
                            if np.shape(mask) != (1, height, width) or not np.isin(mask, (0, 1)).all():
                                raise ValueError('SAM must return one current-frame binary mask')
                            cache[key] = (np.asarray(mask[0], np.uint8), float(np.asarray(quality).reshape(-1)[0]))
                        mask, quality = cache[key]
                        union |= mask
                        group_quality.append(quality)
                    unions.append(torch.from_numpy(union)); qualities.append(group_quality)
                direct.append(unions[0]); tracked.append(unions[1])
                traces.append({'detections': detection, 'tracked_prompts': tracked_prompts,
                               'direct_sam_predicted_iou': qualities[0], 'tracked_sam_predicted_iou': qualities[1]})
        def result(values, source):
            masks = torch.stack(values).unsqueeze(1).float()
            return SubjectMasks(masks, masks.flatten(2).any(-1), phrase, source)
        self.last_direct_masks = result(direct, 'automatic-box-mobilesam-union-threshold05')
        masks = result(tracked, 'clip-local-tracked-box-mobilesam-union')
        self.last_diagnostics = {'frames': traces, 'num_frames': len(frames),
                                 'direct_empty_frames': int((~self.last_direct_masks.instance_present).sum()),
                                 'tracked_empty_frames': int((~masks.instance_present).sum()),
                                 'gap_prompt_count': sum(p['source'] == 'bounded_gap' for ps in prompts for p in ps),
                                 'anchor_prompt_count': sum(p['source'] == 'clip_anchor' for ps in prompts for p in ps),
                                 'prompt_policy': self.prompt_policy}
        return masks
