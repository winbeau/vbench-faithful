"""Paper semantic scoring over fixed evidence, integrated from the published scorer.

The scoring functions retain the published formulas, denominators and abstentions.
Historical matrix construction and table replay use the pinned HF reproduction bundle.
"""
from __future__ import annotations

from collections import defaultdict
import re

from . import records as R
from . import spatial_repair, objects_repair
from .experiments import RELATIONS, metadata_index, scene_rule_key, scene_rule_caption, text_key
from .official_replay import action_score, scene_scores, spatial_scores
from .action_repair import compile_action
from .action_lexicon import ACTION_SYNONYMS


def native_entity_codec(metadata):
    """Invert the declared text canonicalizer over the full public label inventory.

    This is a text-side serialization adapter, independent of the test targets,
    predictions, detections and labels. Ambiguous inverses are never guessed.
    """
    index = metadata_index(metadata)
    inventory = set()
    for value in index['spatial'].values():
        if isinstance(value, dict):
            inventory.update([value['object_a'], value['object_b']])
    for value in index['objects'].values():
        if isinstance(value, str):
            inventory.update(name.strip() for name in value.split(' and '))
    inverse = defaultdict(set)
    for name in inventory:
        inverse[R.canonical_entity_name(name)].add(name)
    aliases = {name: next(iter(values)) for name, values in inverse.items() if len(values)==1}
    # A literal native label is already serialized correctly.
    aliases.update({name: name for name in inventory})
    return {'aliases': aliases, 'inventory': sorted(inventory),
            'ambiguous': {k: sorted(v) for k,v in inverse.items() if len(v)>1}}


def backend_target(task, target, codec, *, normalize_spatial=False):
    if not isinstance(target, dict) or task not in {'objects','spatial'}:
        return target
    aliases = (codec or {}).get('aliases', {})
    if task == 'objects':
        return {**target, 'entities': [aliases.get(v,v) for v in target.get('entities',[])]}
    name = (lambda value: spatial_repair.entity_name(value, aliases)) if normalize_spatial else (lambda value: aliases.get(value, value))
    return {**target, 'relationships': [
        {**triple, 'subject': name(triple['subject']),
         'object': name(triple['object'])}
        for triple in target.get('relationships',[])]}


def rule_target(task, prompt, vocab):
    if task == 'spatial':
        for relation, phrase in RELATIONS.items():
            if phrase in prompt.lower():
                a, b = prompt.lower().split(phrase, 1)
                return {'relationships': [{'subject': R.canonical_entity_name(a.strip()), 'relation': relation,
                    'object': R.canonical_entity_name(b.split(',')[0].strip().rstrip('.'))}]}
        return {'relationships': []}
    if task == 'objects':
        entities = [R.canonical_entity_name(p.strip()) for p in prompt.lower().split(',')[0].rstrip('.').split(' and ')]
        return {'entities': entities} if len(entities) >= 2 and all(entities) else {'entities': []}
    if task == 'action':
        phrase = re.sub(r'^a person is\s+', '', prompt.strip(), flags=re.I).rstrip('.')
        aliases = {v: vocab.resolve(k) for k, v in ACTION_SYNONYMS.items()}
        # Resolve a whole named class before interpreting an explicit conjunction.
        full = aliases.get(phrase.lower()) or vocab.resolve(phrase)
        if full:
            return {'actions': [full]}
        actions = [aliases.get(p.lower()) or vocab.resolve(p) or 'other' for p in phrase.split(' and ')]
        return {'actions': list(dict.fromkeys(actions))}
    raise ValueError(task)


def origin_compiled(task, target):
    if target is None:
        return None
    if task == 'spatial':
        reverse = {v:k for k,v in RELATIONS.items()}
        return {'relationships': [{'subject':target['object_a'], 'relation':reverse[target['relationship']], 'object':target['object_b']}]}
    return {('actions' if task == 'action' else 'entities'): [target] if task == 'action' else target.split(' and ')}


def parsed_target(task, row, scheme, predictions, vocab):
    if scheme == 'Origin':
        return origin_compiled(task, row['official_target'])
    if scheme == 'Repair-rule':
        return rule_target(task, row['prompt'], vocab)
    return predictions.get(text_key(task, row['prompt']))


def score_one(row, scheme, evidence, predictions, vocab, codec=None, *, spatial_backend='repair-v2', action_interface='repair-v2.1', objects_backend='repair-v2'):
    if spatial_backend not in {'repair-v2', 'legacy-official'}:
        raise ValueError('unknown spatial backend: ' + str(spatial_backend))
    if action_interface not in {'repair-v2.1', 'repair-v2', 'legacy-model'}:
        raise ValueError('unknown Action interface: ' + str(action_interface))
    if objects_backend not in {'repair-v2', 'legacy-official'}:
        raise ValueError('unknown Objects backend: ' + str(objects_backend))
    task = row['task']
    frames = 1 if task == 'action' else 16
    result = {'score':0., 'coverage':0., 'abstention':0., 'missing':1., 'target':None,
              'status':'missing_evidence', 'frame_scores':[0.] * frames}
    if not row['eligible']:
        return {**result, 'status':'ineligible'}
    good = evidence.get('status') in {'ok','partial'}
    if task == 'scene':
        captions = evidence.get('frame_captions', []) if good else []
        labels, scores, covered, abstained = [], [], 0, 0
        for i in range(16):
            caption = captions[i] if i < len(captions) else None
            label = None
            if isinstance(caption, str):
                if scheme == 'Repair-model':
                    label = predictions.get(text_key('scene', row['prompt'], caption))
                else:
                    key = row['official_target']
                    if scheme == 'Repair-rule':
                        key, caption = scene_rule_key(key), scene_rule_caption(caption)
                    label = 'supported' if scene_scores(key, [caption])[0] else 'contradicted'
            labels.append(label)
            covered += label in R.SCENE_LABELS
            abstained += label == 'insufficient'
            scores.append(float(label == 'supported'))
        return {**result, 'score':sum(scores)/16, 'coverage':covered/16, 'abstention':abstained/16,
            'missing':1-covered/16, 'target':labels, 'status':'ok' if covered==16 else 'partial', 'frame_scores':scores}
    target = parsed_target(task, row, scheme, predictions, vocab)
    if task == 'action' and scheme == 'Repair-model' and action_interface in {'repair-v2.1', 'repair-v2'}:
        resolution = compile_action(row['prompt'], target, vocab, scope_guard=action_interface == 'repair-v2.1')
        result['action_resolution'] = resolution
        target = resolution['target']
    result['target'] = target
    if not isinstance(target, dict):
        result['status'] = 'missing_parse'; return result
    key = {'spatial':'relationships','objects':'entities','action':'actions'}[task]
    values = target.get(key)
    if not isinstance(values, list) or not values:
        result['status'] = 'empty_parse'; return result
    if scheme != 'Origin':
        target = backend_target(task, target, codec, normalize_spatial=spatial_backend=='repair-v2')
        values = target[key]
    result['backend_target'] = target
    if not good:
        return result
    if task == 'objects' and scheme != 'Origin' and objects_backend == 'repair-v2':
        try:
            return {**result, **objects_repair.video_scores(values, evidence.get('frame_detections', []))}
        except (ValueError, TypeError) as error:
            return {**result, 'status': 'invalid_objects_evidence', 'errors': [str(error)]}
    if task == 'action':
        if not evidence.get('top5'):
            return result
        scores = [float(action_score(label, evidence['top5'])) if label != 'other' else 0. for label in values]
        return {**result, 'score':sum(scores)/len(scores), 'coverage':1., 'missing':0.,
            'abstention':values.count('other')/len(values), 'status':'ok', 'frame_scores':[sum(scores)/len(scores)],
            'known_class_score':sum(s for l,s in zip(values,scores) if l!='other')/max(1,sum(l!='other' for l in values))}
    evidence_frames = evidence.get('frame_detections' if task == 'spatial' else 'frame_labels', [])
    scores, covered, errors = [], 0, []
    for i in range(16):
        frame = evidence_frames[i] if i < len(evidence_frames) else None
        if frame is None:
            scores.append(0.); errors.append('missing_frame'); continue
        try:
            if task == 'objects':
                value = float(all(name in frame for name in values))
            else:
                components = []
                for triple in values:
                    if scheme != 'Origin' and spatial_backend == 'repair-v2':
                        components.append(spatial_repair.frame_score(triple, frame))
                    else:
                        mapped = {'object_a':triple['subject'], 'object_b':triple['object'], 'relationship':RELATIONS[triple['relation']]}
                        components.append(spatial_scores(mapped, [frame])[0])
                value = sum(components)/len(components)
            scores.append(value); covered += 1
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as error:
            scores.append(0.); errors.append(type(error).__name__)
    return {**result, 'score':sum(scores)/16, 'coverage':covered/16, 'missing':1-covered/16,
            'status':'ok' if covered==16 else 'partial', 'frame_scores':scores, 'errors':errors}
