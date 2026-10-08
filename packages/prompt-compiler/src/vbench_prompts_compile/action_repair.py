"""Auditable Action interface: v9 output plus declared text equivalences.

Only the current prompt, model output and public vocabulary are inputs. No
video evidence, original-prompt prediction, transform ID or expected class is
available here. Closed-contract resolution is reported separately from the
model's open-text path; it is not a claim of learned synonym generalization.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import re

from .action_lexicon import ACTION_SYNONYMS
from .records import canonical_json, normalize_phrase

INTERFACE_VERSION = 'repair-v2.1'
LEXICON_SHA256 = hashlib.sha256(canonical_json(ACTION_SYNONYMS).encode()).hexdigest()
DETERMINERS = re.compile(r'\b(?:a|an|the|my|your|his|her|its|our|their)\b')
SUBJECT = re.compile(r'^(?:(?:a|an|the) (?:person|man|woman|child|baby)|someone|somebody|he|she|they|people) (?:is|are)\s+')
# Whole declared class/alias matches take precedence: "petting animal (not cat)"
# and "pretending to drum in the air" have explicitly declared meanings.
UNSUPPORTED_CONTEXT = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|isn't|aren't|wasn't|weren't|if|may|might|would|could|will|"
    r"watching|observing|imagining|planning|intending|pretending|avoiding|refusing|"
    r"wants? to|hopes? to|used to|going to|about to|thinking about)\b|n't\b")


def relaxed_key(value):
    """Grammar-only equivalence; no stemming, word reordering or fuzzy match."""
    value = re.sub(r'[-\u2010-\u2015]', ' ', normalize_phrase(value))
    return ' '.join(DETERMINERS.sub(' ', value).split())


def valid_target(value):
    return (isinstance(value, dict) and set(value) == {'actions'}
            and isinstance(value['actions'], list)
            and all(isinstance(v, str) and v.strip() for v in value['actions']))


def unique(values):
    return list(dict.fromkeys(values))


class ActionNormalizer:
    def __init__(self, vocabulary, synonyms=None):
        self.vocabulary = vocabulary
        self.exact = defaultdict(set)
        self.relaxed = defaultdict(set)
        names = {normalize_phrase(v): v for v in vocabulary.labels}
        aliases = [(v, v) for v in vocabulary.labels]
        aliases += [(v, names[normalize_phrase(k)]) for k, v in (ACTION_SYNONYMS if synonyms is None else synonyms).items()
                    if normalize_phrase(k) in names]
        aliases += [(k, names[normalize_phrase(v)]) for k, v in vocabulary.aliases.items() if normalize_phrase(v) in names]
        for phrase, label in aliases:
            self.exact[normalize_phrase(phrase)].add(label)
            self.relaxed[relaxed_key(phrase)].add(label)

    def candidates(self, phrase):
        exact = self.exact.get(normalize_phrase(phrase))
        return exact if exact else self.relaxed.get(relaxed_key(phrase), set())

    def resolve(self, phrase):
        values = self.candidates(phrase)
        return next(iter(values)) if len(values) == 1 else None

    def normalize_output(self, value):
        if not valid_target(value):
            return None
        mapped = []
        for phrase in value['actions']:
            candidates = self.candidates(phrase)
            label = next(iter(candidates)) if len(candidates) == 1 else None
            # Preserve the existing model-output resolver outside the closed
            # contract (e.g. "climbing a tall tree"). Ambiguity and unsafe
            # scope must never fall through to token containment.
            if not candidates and not UNSUPPORTED_CONTEXT.search(normalize_phrase(phrase)):
                label = self.vocabulary.resolve(phrase)
            mapped.append(label or 'other')
        return {'actions': unique(mapped)}

    def prompt_slots(self, prompt):
        phrase = SUBJECT.sub('', normalize_phrase(prompt), count=1)
        if not phrase:
            return []
        # Resolve whole class names first, including those containing "and".
        if self.candidates(phrase):
            return [self.slot(phrase)]
        pieces = re.split(r'\s+and\s+', phrase)
        slots, start = [], 0
        while start < len(pieces):
            # Longest complete class at a conjunction boundary. This preserves
            # "clean and jerk and <unknown action>" without discarding unknowns.
            end = next((j for j in range(len(pieces), start, -1)
                        if self.candidates(' and '.join(pieces[start:j]))), start + 1)
            slots.append(self.slot(' and '.join(pieces[start:end])))
            start = end
        return slots

    def slot(self, phrase):
        values = self.candidates(phrase)
        label = next(iter(values)) if len(values) == 1 else None
        return {'text': phrase, 'label': label, 'ambiguous': len(values) > 1,
                'blocked_context': not label and bool(UNSUPPORTED_CONTEXT.search(phrase))}

    def compile(self, prompt, model_target, *, scope_guard=True):
        result = self.compile_frozen_v2(prompt, model_target)
        if not scope_guard:
            return result
        result['version'] = INTERFACE_VERSION
        slots = result['prompt_slots']
        if result['normalized_model_target'] is None or (slots and all(s['label'] for s in slots)):
            return result
        # The closed prompt grammar is one subject assertion, not a parser for
        # arbitrary prose. A global "without" or "and" in camera/style prose
        # must never replace a correct model parse or invent an extra action.
        reasons = []
        if not result['normalized_model_target']['actions'] and not any(s['label'] for s in slots):
            reasons.append('preserve_model_empty_outside_positive_contract')
        if not SUBJECT.match(normalize_phrase(prompt)):
            reasons.append('outside_declared_subject_form')
        if re.search(r'[.!?;\n]', prompt.strip().rstrip('.!?')):
            reasons.append('multiple_clauses')
        if len(slots) > 1 and not any(s['label'] for s in slots):
            reasons.append('unresolved_conjunction')
        if len(slots) > 1 and any(not s['label'] and not s['ambiguous'] and not s['blocked_context']
                                  and not re.match(r'^[a-z]+ing\b', s['text']) for s in slots):
            reasons.append('outside_declared_action_conjunction')
        if reasons:
            target = result['normalized_model_target']
            result.update(target=target, method='model_output', changed=target != model_target,
                          scope_guard={'applied': True, 'reasons': reasons})
        return result

    def compile_frozen_v2(self, prompt, model_target):
        """Original v2 behavior retained for immutable pre-audit comparisons."""
        normalized = self.normalize_output(model_target)
        slots = self.prompt_slots(prompt)
        info = {'version': 'repair-v2', 'lexicon_sha256': LEXICON_SHA256,
                'input_target': model_target, 'normalized_model_target': normalized, 'prompt_slots': slots}
        if normalized is None:
            return {**info, 'target': None, 'method': 'invalid_model_output', 'changed': False}
        if slots and all(slot['label'] for slot in slots):
            target = {'actions': unique(slot['label'] for slot in slots)}
            method = 'complete_prompt_contract'
        elif any(slot['blocked_context'] or slot['ambiguous'] for slot in slots):
            # Scope is not licensed by merely mentioning a known action. Keep
            # independently resolved conjunctions and reject the unsafe pieces.
            target = {'actions': unique(slot['label'] or 'other' for slot in slots)}
            method = 'scoped_prompt_contract'
        elif len(slots) > 1 and any(slot['label'] for slot in slots):
            target = {'actions': unique(slot['label'] or 'other' for slot in slots)}
            method = 'partial_prompt_contract'
        else:
            target = normalized
            if len(slots) > 1 and target['actions']:
                target = {'actions': unique([*target['actions'], 'other'])}
            method = 'model_output'
        return {**info, 'target': target, 'method': method, 'changed': target != model_target}


@lru_cache(maxsize=8)
def normalizer(vocabulary):
    return ActionNormalizer(vocabulary)


def compile_action(prompt, model_target, vocabulary, *, scope_guard=True):
    return normalizer(vocabulary).compile(prompt, model_target, scope_guard=scope_guard)
