"""Full-denominator behavioral correctness; never reward wrong invariant predictions."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import NormalDist

TASKS = ('spatial', 'action', 'objects', 'scene')
TRANSFORMS = ('paraphrase', 'changed', 'composition', 'long_head', 'long_middle', 'long_tail')


def semantic_key(task, value):
    """Set-valued exact target match, ignoring order/case and equivalent inverse triples only.

    No fuzzy noun, synonym, or evidence-dependent matching. Articles and action
    synonyms belong in the separately ablated production interface.
    """
    if task == 'scene':
        return value if isinstance(value, str) and value in ('supported', 'contradicted', 'insufficient') else None
    if not isinstance(value, dict):
        return None
    key = {'spatial': 'relationships', 'action': 'actions', 'objects': 'entities'}[task]
    if set(value) != {key} or not isinstance(value[key], list):
        return None
    if task != 'spatial':
        if any(not isinstance(v, str) or not v.strip() for v in value[key]):
            return None
        return tuple(sorted({' '.join(v.lower().split()) for v in value[key]}))
    triples = []
    opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}
    for triple in value[key]:
        if not isinstance(triple, dict) or set(triple) != {'subject', 'relation', 'object'}:
            return None
        if any(not isinstance(v, str) or not v.strip() for v in triple.values()) or triple['relation'] not in opposite:
            return None
        a, r, b = triple['subject'].lower().strip(), triple['relation'], triple['object'].lower().strip()
        triples.append(min((a, r, b), (b, opposite[r], a)))
    return tuple(sorted(set(triples)))


def correct(task, prediction, target):
    expected = semantic_key(task, target)
    if expected is None:
        raise ValueError('invalid evaluation target')
    actual = semantic_key(task, prediction)
    return actual is not None and actual == expected


def wilson_lower(successes, n, *, groups=1, alpha=.05):
    if n == 0:
        return None
    z = NormalDist().inv_cdf(1 - alpha / groups)
    p = successes / n
    return max(0., (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


def paired_delta(values, *, seed=20260919, rounds=2000):
    """Values are paired per-family accuracy differences, not repeated frames."""
    if not values:
        return {'n': 0, 'mean': None, 'ci95': None, 'noninferior_at_minus_003': False}
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(rounds))
    ci = [draws[int(.025 * rounds)], draws[min(rounds - 1, int(.975 * rounds))]]
    return {'n': len(values), 'mean': sum(values) / len(values), 'ci95': ci,
            'noninferior_at_minus_003': ci[0] >= -.03}


def behavioral_score(rows, predictions, *, expected_tasks=TASKS, expected_families=24):
    """All expected task/transform groups must exist; missing predictions count wrong.

    Each family contains exactly one canonical item and one per transform.
    Inference failures stay in the denominator. Duplicate variants are an error.
    """
    families = defaultdict(dict)
    for row in rows:
        if row['source'] != 'engineered' or row['task'] not in expected_tasks:
            continue
        key = (row['task'], row['family'])
        if row['variant'] in families[key]:
            raise ValueError('duplicate family variant')
        families[key][row['variant']] = row
    expected_groups = len(expected_tasks) * len(TRANSFORMS)
    groups, complete = {}, True
    for task in expected_tasks:
        fs = [v for (t, _), v in families.items() if t == task]
        if len(fs) != expected_families:
            complete = False
        for variant in TRANSFORMS:
            success, individual, deltas = [], [], []
            for family in fs:
                base, changed = family.get('canonical'), family.get(variant)
                if base is None or changed is None:
                    complete = False
                b = bool(base and correct(task, predictions.get(base['id']), base['target']))
                c = bool(changed and correct(task, predictions.get(changed['id']), changed['target']))
                success.append(b and c)
                individual.append(c)
                deltas.append(int(c) - int(b))
            n = len(fs)
            groups[f'{task}/{variant}'] = {'families': n, 'pair_successes': sum(success),
                'cc': sum(success)/n if n else None, 'variant_accuracy': sum(individual)/n if n else None,
                'simultaneous_lower95': wilson_lower(sum(success), n, groups=expected_groups),
                'paired_accuracy_delta': paired_delta(deltas)}
    worst = min((v['cc'] for v in groups.values() if v['cc'] is not None), default=None)
    lower = min((v['simultaneous_lower95'] for v in groups.values() if v['simultaneous_lower95'] is not None), default=None)
    return {'complete_groups': complete, 'wg_cc': worst if complete else None,
            'wg_cc_simultaneous_lower95': lower if complete else None,
            'worst_groups': [k for k, v in groups.items() if v['cc'] == worst], 'groups': groups}
