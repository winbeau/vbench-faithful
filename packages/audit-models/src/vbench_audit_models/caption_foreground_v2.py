"""Caption viewpoint normalization; preserve the earlier frozen candidate."""
from __future__ import annotations

import re

from .caption_foreground import GritCaptionBoxDetector, GritSamFallbackForegroundProvider, select_caption_box


def normalize_caption(text, policy):
    return re.sub(policy['viewpoint_prefix_regex'], '', text.lower(), count=1).strip()


def select_caption_box_v2(instances, shape, policy):
    normalized = [{**row, 'original_text': row['text'], 'text': normalize_caption(row['text'], policy)}
                  for row in instances]
    return select_caption_box(normalized, shape, policy)


class GritCaptionBoxDetectorV2(GritCaptionBoxDetector):
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
        return select_caption_box_v2(output['postprocessed'], frame.shape[:2], self.policy)


class GritSamFallbackForegroundProviderV2(GritSamFallbackForegroundProvider):
    def __init__(self, provider, policy, *, device):
        self.provider = provider
        self.caption_detector = GritCaptionBoxDetectorV2(policy, device=device)
        self.provenance = {**provider.provenance, 'caption_fallback': self.caption_detector.provenance,
            'caption_policy_revision': 2,
            'fallback_trigger': 'zero foreground pixels on this actual frame',
            'fallback_selection': 'largest eligible native caption object box after viewpoint normalization; existing GRiT proposal threshold retained; no second threshold on language-rescored confidence; independent SAM'}
        self.last_diagnostics = None
