"""Development parser probe for noun heads before relational predicates.

This does not enable a new scoring provider or change either frozen caption
candidate. Keep the existing scene/object vocabulary and box criteria.
"""
from __future__ import annotations

import re

from .caption_foreground import select_caption_box
from .caption_foreground_v2 import normalize_caption


RELATION_CLAUSE = re.compile(
    r'\s+(?:covering|covers|containing|contains|covered\s+(?:by|with|in)|surrounded\s+by)\b'
)


def normalize_caption_relations(text, policy):
    phrase = normalize_caption(text, policy)
    relation = RELATION_CLAUSE.search(phrase)
    return phrase[:relation.start()].rstrip() if relation else phrase


def select_caption_box_with_relations(instances, shape, policy):
    normalized = [{**row, 'original_text': row['text'],
                   'text': normalize_caption_relations(row['text'], policy)} for row in instances]
    return select_caption_box(normalized, shape, policy)
