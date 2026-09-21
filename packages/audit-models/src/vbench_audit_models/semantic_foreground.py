"""Input-only CLIP semantics for MobileSAM proposals, with existing weights.

No construction target, construction mask, preference label or metric score is
accepted. Object and scene vocabularies are fixed across all actual inputs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .automatic_foreground import select_automatic_mask


def proposal_views(frame, mask, *, padding_fraction, fill_rgb):
    frame, mask = np.asarray(frame), np.asarray(mask)
    if (frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8
            or mask.shape != frame.shape[:2] or not np.isin(mask, (0, 1)).all() or not mask.any()):
        raise ValueError('proposal needs native RGB bytes and a nonempty binary mask')
    y, x = np.nonzero(mask)
    x0, x1, y0, y1 = int(x.min()), int(x.max()+1), int(y.min()), int(y.max()+1)
    pad = int(np.ceil(max(x1-x0, y1-y0)*padding_fraction))
    x0, x1, y0, y1 = max(0, x0-pad), min(frame.shape[1], x1+pad), max(0, y0-pad), min(frame.shape[0], y1+pad)
    crop = frame[y0:y1, x0:x1].copy()
    isolated = np.full_like(crop, fill_rgb)
    visible = mask[y0:y1, x0:x1].astype(bool)
    isolated[visible] = crop[visible]
    return Image.fromarray(crop), Image.fromarray(isolated)


class ClipProposalClassifier:
    def __init__(self, model, preprocess, tokenize, policy, *, device):
        import torch
        from torch.nn import functional as F
        self.model, self.preprocess, self.device, self.policy = model, preprocess, device, dict(policy)
        self.objects, self.surfaces = policy['object_labels'], policy['scene_labels']
        self.labels = self.objects+self.surfaces
        if not self.objects or not self.surfaces or len(set(self.labels)) != len(self.labels):
            raise ValueError('object and scene labels must be nonempty, disjoint and unique')
        prompts = [template.format(label=label) for label in self.labels for template in policy['templates']]
        with torch.inference_mode():
            encoded = F.normalize(model.encode_text(tokenize(prompts).to(device)).float(), dim=-1)
            self.text_features = F.normalize(encoded.reshape(len(self.labels), len(policy['templates']), -1).mean(1), dim=-1)

    def classify(self, frame, records):
        import torch
        from torch.nn import functional as F
        if not records:
            return []
        pictures = [picture for row in records for picture in proposal_views(frame, row['segmentation'],
            padding_fraction=self.policy['padding_fraction'], fill_rgb=self.policy['fill_rgb'])]
        similarities = []
        with torch.inference_mode():
            for start in range(0, len(pictures), self.policy['batch_size']):
                batch = torch.stack([self.preprocess(picture) for picture in pictures[start:start+self.policy['batch_size']]]).to(self.device)
                features = F.normalize(self.model.encode_image(batch).float(), dim=-1)
                similarities.append((features@self.text_features.T).cpu())
        values = torch.cat(similarities).reshape(len(records), 2, len(self.labels)).mean(1).numpy()
        if not np.isfinite(values).all():
            raise ValueError('nonfinite proposal text/image similarities')
        result = []
        boundary = len(self.objects)
        for row in values:
            obj, scene = int(np.argmax(row[:boundary])), boundary+int(np.argmax(row[boundary:]))
            best = np.argsort(-row, kind='stable')[:5]
            result.append({'object_label': self.labels[obj], 'object_similarity': float(row[obj]),
                'scene_label': self.labels[scene], 'scene_similarity': float(row[scene]),
                'object_margin': float(row[obj]-row[scene]),
                'top_labels': [{'label': self.labels[i], 'similarity': float(row[i])} for i in best]})
        return result


def select_semantic_mask(records, classifications, shape, geometry, semantic):
    if len(records) != len(classifications):
        raise ValueError('one semantic classification required per proposal')
    _, trace = select_automatic_mask(records, shape, geometry)
    eligible = []
    for row, label in zip(trace['candidates'], classifications):
        margin = float(label['object_margin'])
        if not np.isfinite(margin):
            raise ValueError('nonfinite semantic margin')
        row.update(label)
        row['geometry_eligible'] = row['eligible']
        row['eligible'] = row['geometry_eligible'] and margin >= semantic['minimum_object_margin']
        if row['eligible']:
            eligible.append(row['index'])
    if not eligible:
        trace['selected_index'] = None
        return np.zeros(shape, np.uint8), trace
    chosen = min(eligible, key=lambda i: (-trace['candidates'][i]['area_fraction'],
        -trace['candidates'][i]['object_margin'], -trace['candidates'][i]['stability_score'], i))
    trace['selected_index'] = chosen
    return np.asarray(records[chosen]['segmentation'], dtype=np.uint8), trace
