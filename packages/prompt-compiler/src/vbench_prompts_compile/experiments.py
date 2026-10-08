"""Frozen experiment identities, official metadata and paired cluster estimates."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
import re

SCENE_SYNONYMS = {
    'ocean': 'sea', 'phone booth': 'telephone booth', 'carrousel': 'carousel',
    'pharmacy': 'drugstore', 'gas station': 'petrol station',
    'indoor movie theater': 'indoor cinema', 'harbor': 'harbour',
    'staircase': 'stairway', 'alley': 'alleyway',
    'indoor gymnasium': 'indoor gym', 'train station platform': 'railway station platform',
}
OFFICIAL_DIMENSIONS = {'scene': 'scene', 'spatial': 'spatial_relationship',
                       'action': 'human_action', 'objects': 'multiple_objects'}
RELATIONS = {'left': 'on the left of', 'right': 'on the right of',
             'above': 'on the top of', 'below': 'on the bottom of'}


def text_key(task, prompt, caption=None):
    return hashlib.sha256(json.dumps([task, prompt, caption], ensure_ascii=False).encode()).hexdigest()


def bare_scene(prompt):
    return re.sub(r'^(?:a|an|the)\s+', '', prompt.strip(), flags=re.I)


def scene_rule_key(key):
    reverse = {v: k for k, v in SCENE_SYNONYMS.items()}
    return reverse.get(key.lower(), key.lower())


def scene_rule_caption(caption):
    text = caption.lower()
    for canonical, alias in sorted(SCENE_SYNONYMS.items(), key=lambda p: -len(p[1])):
        text = re.sub(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', canonical, text)
    return text


def metadata_index(rows):
    result = {d: {} for d in OFFICIAL_DIMENSIONS}
    for task, dimension in OFFICIAL_DIMENSIONS.items():
        for row in rows:
            if dimension not in row['dimension']:
                continue
            if task == 'action':
                value = row['prompt_en'].lower().split('person is ')[-1]
            else:
                value = row.get('auxiliary_info', {}).get(dimension)
                # load_dimension_info and each dimension unwrap these nested keys.
                while isinstance(value, dict) and set(value) == {dimension}:
                    value = value[dimension]
                if task == 'scene':
                    while isinstance(value, dict) and set(value) == {'scene'}:
                        value = value['scene']
                if task == 'objects' and isinstance(value, dict):
                    value = value['object']
            result[task][row['prompt_en']] = value
    return result


def official_target(index, task, prompt):
    key = bare_scene(prompt) if task == 'scene' else prompt
    return index[task].get(key)


def estimate(rows, key, *, strata=None):
    """Full-denominator mean, or macro mean over a fixed set of label strata."""
    if not rows:
        return None
    if strata is None:
        return sum(float(r.get(key) or 0) for r in rows) / len(rows)
    groups = {s: [r for r in rows if r.get('expected') == s] for s in strata}
    if any(not v for v in groups.values()):
        return None
    return sum(estimate(v, key) for v in groups.values()) / len(groups)


def cluster_summary(rows, key, *, strata=None, rounds=2000, seed=20260919):
    """Resample whole families and recompute the SAME reported estimator.

    Missing categories make a macro estimate undefined, never silently reduce its
    category count. The undefined bootstrap fraction is reported explicitly.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r['family']].append(r)
    point = estimate(rows, key, strata=strata)
    result = {'n': len(rows), 'families': len(groups), 'estimate': point,
              'ci': None, 'bootstrap_valid': 0, 'bootstrap_rounds': rounds}
    if len(groups) < 2 or point is None:
        return result
    categories = tuple(strata) if strata is not None else (None,)
    sufficient = []
    for values in groups.values():
        sufficient.append([(sum(float(r.get(key) or 0) for r in values if s is None or r.get('expected') == s),
                            sum(s is None or r.get('expected') == s for r in values)) for s in categories])
    rng = random.Random(seed)
    distribution = []
    for _ in range(rounds):
        draws = rng.choices(sufficient, k=len(sufficient))
        totals = [(sum(d[i][0] for d in draws), sum(d[i][1] for d in draws)) for i in range(len(categories))]
        if all(n for _, n in totals):
            distribution.append(sum(v / n for v, n in totals) / len(categories))
    distribution.sort()
    result['bootstrap_valid'] = len(distribution)
    if distribution:
        result['ci'] = [distribution[int(.025 * (len(distribution) - 1))],
                        distribution[int(.975 * (len(distribution) - 1))]]
    return result
