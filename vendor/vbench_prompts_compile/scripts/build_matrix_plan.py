#!/usr/bin/env python3
"""Freeze VBench test transformations from real manifests, before model prediction.

Unsupported transformations stay in the plan with an explicit eligibility reason.
The lexicon is a declared closed dictionary control, not independently labelled
human gold. Class replacement alone does not verify absence in the source video.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import records as R
from vbench_prompts_compile.experiments import SCENE_SYNONYMS, metadata_index, official_target, bare_scene
from vbench_prompts_compile.sources import load_k400
from cache_backend_outputs import rows_from_manifest, DIMENSIONS, sha
import csv

# Semantic paraphrases of individual K400 classes; no model/visual outputs used.
from vbench_prompts_compile.action_lexicon import ACTION_SYNONYMS
OPPOSITE = {'on the left of': 'on the right of', 'on the right of': 'on the left of',
            'on the top of': 'on the bottom of', 'on the bottom of': 'on the top of'}


def variants(task, row, index, vocab):
    prompt = row['prompt']
    target = official_target(index, task, prompt)
    base = {'prompt': prompt, 'official_target': target, 'eligible': target is not None,
            'eligibility_reason': None if target is not None else 'no_official_auxiliary_target',
            'evidence_kind': 'original', 'human_expectation_verified': False}
    result = [{**base, 'transform': 'identity'}]
    if task == 'scene':
        key = bare_scene(prompt)
        synonym = SCENE_SYNONYMS.get(key)
        result.append({**base, 'transform': 'synonym', 'eligible': synonym is not None,
                       'eligibility_reason': None if synonym else 'no_strict_synonym_in_frozen_lexicon',
                       'prompt': (('an ' if synonym[0] in 'aeiou' else 'a ') + synonym) if synonym else prompt, 'official_target': synonym,
                       'expected': 'invariant', 'semantic_basis': 'declared_contextual_synonym_lexicon'})
        # Fixed disjoint category, not chosen after inspecting a caption.
        replacement = 'kitchen' if key != 'kitchen' else 'ocean'
        result.append({**base, 'transform': 'class_swap', 'prompt': 'a ' + replacement,
                       'official_target': replacement, 'expected': 'sensitivity_if_replacement_absent'})
    elif task == 'action':
        label = target
        canonical = vocab.resolve(label) if label else None
        ids = {e['label']: e['id'] for e in vocab.entries}
        synonym = ACTION_SYNONYMS.get(label)
        result.append({**base, 'transform': 'synonym', 'eligible': synonym is not None and canonical is not None,
                       'eligibility_reason': None if synonym and canonical else 'unmapped_class_or_synonym',
                       'prompt': 'A person is ' + synonym if synonym else prompt,
                       'official_target': synonym, 'expected_actions': [canonical] if canonical else None,
                       'expected_class_ids': [ids[canonical]] if canonical else None, 'expected': 'invariant'})
        other = vocab.by_id((ids[canonical] + 97) % 400) if canonical else None
        result.append({**base, 'transform': 'class_swap', 'prompt': 'A person is ' + other if other else prompt,
                       'official_target': other, 'expected_actions': [other] if other else None,
                       'expected': 'sensitivity_if_replacement_absent', 'eligible': other is not None})
        result.append({**base, 'transform': 'oov', 'prompt': 'A person is assembling furniture',
                       'official_target': 'assembling furniture', 'expected_actions': ['other'], 'expected': 'abstain'})
        result.append({**base, 'transform': 'mixed_oov', 'prompt': prompt + ' and assembling furniture',
                       'official_target': str(label) + ' and assembling furniture',
                       'expected_actions': [canonical, 'other'] if canonical else None,
                       'expected': 'preserve_known_and_abstain_unknown'})
    elif task == 'spatial':
        valid = isinstance(target, dict) and target.get('relationship') in OPPOSITE
        axis = ('hflip' if 'left' in target['relationship'] or 'right' in target['relationship'] else 'vflip') if valid else None
        swapped = {**target, 'relationship': OPPOSITE[target['relationship']]} if valid else None
        swapped_prompt = prompt.replace(target['relationship'], swapped['relationship'], 1) if valid else prompt
        for kind, evidence, change_prompt, expectation in [
            ('evidence_mirror', 'evidence_mirror', False, 'reverse_direction'),
            ('video_mirror', 'video_mirror', False, 'reverse_direction'),
            ('prompt_swap', 'original', True, 'reverse_direction'),
            ('evidence_mirror_joint', 'evidence_mirror', True, 'invariant'),
            ('video_mirror_joint', 'video_mirror', True, 'invariant'),
        ]:
            result.append({**base, 'transform': kind, 'evidence_kind': evidence,
                'prompt': swapped_prompt if change_prompt else prompt,
                'official_target': swapped if change_prompt else target,
                'eligible': valid, 'axis': axis, 'expected': expectation,
                'eligibility_reason': None if valid else 'outside_four_direction_protocol'})
    else:
        for control in ['target', 'background', 'non_target']:
            for level in [0.0, 0.5, 0.8, 1.0]:
                result.append({**base, 'transform': f'{control}_occlusion_{level}',
                    'evidence_kind': 'occlusion', 'occlusion_control': control, 'level': level,
                    'target_entity': target.split(' and ')[0] if target else None,
                    'expected': 'fail_if_target_verified_invisible' if control == 'target' else 'control'})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', action='append', required=True, help='task=CSV, repeat for every task')
    p.add_argument('--metadata', required=True)
    p.add_argument('--frozen-split', required=True)
    p.add_argument('--split', choices=['dev', 'test'], default='test')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    index = metadata_index(json.loads(Path(args.metadata).read_text()))
    vocab = load_k400()
    manifests = dict(v.split('=', 1) for v in args.manifest)
    with Path(args.frozen_split).open() as f:
        splits = list(csv.DictReader(f))
    rows = []
    for task, path in manifests.items():
        allowed = {r['prompt_id'] for r in splits if r['dimension'] == DIMENSIONS[task] and r['split'] == args.split}
        originals = [r for r in rows_from_manifest(path) if r['prompt'] in allowed]
        for original in originals:
            for variant in variants(task, original, index, vocab):
                row = {**original, **variant, 'task': task, 'original_prompt': original['prompt'],
                    'family': task + ':' + hashlib.sha256(original['prompt'].encode()).hexdigest()[:16],
                    'split': args.split, 'base_official_target': official_target(index, task, original['prompt'])}
                row['sample_id'] = hashlib.sha256(json.dumps([task, row['relative_path'], row['transform']], ensure_ascii=False).encode()).hexdigest()[:24]
                rows.append(row)
    identity = {'schema': 1, 'metadata_sha256': sha(args.metadata), 'split_sha256': sha(args.frozen_split),
        'manifest_sha256': {k: sha(v) for k,v in manifests.items()}, 'builder_sha256': sha(__file__),
        'lexicon_sha256': R.sha256_text(R.canonical_json(SCENE_SYNONYMS)),
        'k400_sha256': sha(ROOT / 'data/raw/k400/labels.json'), 'split': args.split,
        'rows': len(rows), 'eligible': dict(Counter(r['task'] + '/' + r['transform'] for r in rows if r['eligible'])),
        'ineligible': dict(Counter(r['task'] + '/' + r['transform'] + '/' + str(r['eligibility_reason']) for r in rows if not r['eligible'])),
        'note': 'Construction-only semantic expectations, not human visual verification; unknown/missing items remain visible.'}
    J.bind_job(out / 'manifest.json', identity)
    R.write_jsonl(out / 'matrix.jsonl', rows)
    for task in manifests:
        R.write_jsonl(out / f'{task}.jsonl', [r for r in rows if r['task'] == task])
    print(json.dumps(identity, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
