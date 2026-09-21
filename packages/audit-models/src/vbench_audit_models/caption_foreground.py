"""Existing GRiT captions prompt independent MobileSAM scoring localization."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def caption_head(text, policy):
    # Restrict matching to the described noun phrase, before its attributes or
    # context. 'window on a tower' describes a window; 'glass shower door' has
    # head door, not glass or shower. Unknown heads remain explicit.
    text = text.lower()
    protected = [m.span() for aliases in (policy['object_aliases'], policy['scene_aliases'])
                 for alias in aliases if ' ' in alias
                 for m in re.finditer(r'\b'+re.escape(alias)+r'(?:s|es)?\b', text)]
    stops = [m.start() for m in re.finditer(policy['head_stop_regex'], text)
             if not any(start <= m.start() < end for start, end in protected)]
    phrase = text[:min(stops) if stops else len(text)].strip()
    matches = []
    for kind, labels in [('object', policy['object_aliases']), ('scene', policy['scene_aliases'])]:
        for alias, canonical in labels.items():
            for match in re.finditer(r'\b'+re.escape(alias)+r'(?:s|es)?\b', phrase):
                matches.append((match.end(), -match.start(), kind == 'scene', kind, canonical, alias))
    if not matches:
        return {'phrase': phrase, 'kind': 'unknown', 'label': None}
    chosen = max(matches)
    return {'phrase': phrase, 'kind': chosen[3], 'label': chosen[4], 'matched_alias': chosen[5]}


def select_caption_box(instances, shape, policy):
    height, width = shape
    records = []
    for index, instance in enumerate(instances):
        box = np.asarray(instance['box'], dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any()
                or (box[2:] <= box[:2]).any() or box[2] > width+1e-4 or box[3] > height+1e-4):
            raise ValueError('caption box must be in native image coordinates')
        area = float(np.prod(box[2:]-box[:2])/(height*width))
        head = caption_head(instance['text'], policy)
        eligible = (head['kind'] == 'object' and instance['score'] >= policy['threshold']
                    and policy['minimum_box_area'] <= area <= policy['maximum_box_area'])
        records.append({**instance, 'index': index, 'head': head, 'box_area_fraction': area, 'eligible': eligible})
    eligible = [row for row in records if row['eligible']]
    if not eligible:
        return None, {'selected_index': None, 'captions': records}
    chosen = min(eligible, key=lambda x: (-x['box_area_fraction'], -x['score'], x['index']))
    return chosen['box'], {'selected_index': chosen['index'], 'captions': records}


class GritCaptionBoxDetector:
    def __init__(self, policy, *, device):
        from vbench_audit_core.inputs import sha256_file
        from .grit import GritEvidenceModel
        path = Path(policy['checkpoint'])
        if sha256_file(path) != policy['checkpoint_sha256']:
            raise ValueError('GRiT caption weights changed')
        self.policy = policy
        self.model = GritEvidenceModel('color', path, device=device)
        self.provenance = {**self.model.provenance, 'role': 'scoring_caption_localizer',
            'head': 'DenseCap postprocessed native boxes; no color metric computation',
            'policy': policy, 'strict_determinism_disabled_only_inside_grit_forward': True}

    def box_for(self, frame):
        import torch
        strict = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            output = self.model.detect(frame)
        finally:
            torch.use_deterministic_algorithms(strict, warn_only=warn)
        if output['status'] != 'succeeded':
            raise ValueError('GRiT caption localization failed: '+output['error'])
        return select_caption_box(output['postprocessed'], frame.shape[:2], self.policy)


class GritSamFallbackForegroundProvider:
    def __init__(self, provider, policy, *, device):
        self.provider = provider
        self.caption_detector = GritCaptionBoxDetector(policy, device=device)
        self.provenance = {**provider.provenance, 'caption_fallback': self.caption_detector.provenance,
            'fallback_trigger': 'zero foreground pixels on this actual frame',
            'fallback_selection': 'largest eligible caption head object box, independently prompted SAM; no construction inputs or clean-mask propagation'}
        self.last_diagnostics = None

    def masks_for(self, frames):
        import torch
        masks = self.provider.masks_for(frames)
        base = self.provider.last_diagnostics
        images = frames.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        traces = []
        with torch.inference_mode():
            for index, (frame, mask) in enumerate(zip(images, masks)):
                if mask.any():
                    continue
                box, trace = self.caption_detector.box_for(np.ascontiguousarray(frame))
                if box is not None:
                    predictor = self.provider.predictor
                    predictor.set_image(np.ascontiguousarray(frame), image_format='RGB')
                    prediction, quality, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
                    if prediction.shape != (1, *frame.shape[:2]) or not np.isin(prediction, (0, 1)).all():
                        raise ValueError('caption-prompted SAM returned invalid pixels')
                    masks[index] = prediction[0].astype(np.uint8)
                    trace['sam_predicted_iou'] = float(np.asarray(quality).ravel()[0])
                traces.append({'frame': index, **trace})
        self.last_diagnostics = {**base, 'num_empty_before_fallback': base['num_empty_foreground_frames'],
            'num_empty_foreground_frames': int((masks.sum((1, 2)) == 0).sum()),
            'foreground_fraction': masks.mean((1, 2)).tolist(), 'caption_fallback_frames': traces}
        return masks
