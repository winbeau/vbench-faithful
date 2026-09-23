#!/usr/bin/env python3
"""Freeze a modest behavioral challenge plus SNLI location diagnostics before inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile.sources import load_k400
from vbench_prompts_compile.records import canonical_entity_name

VERSIONS = {'spatial': 'v8', 'action': 'v9', 'objects': 'v6', 'scene': 'v8'}
VARIANTS = ['canonical', 'paraphrase', 'changed', 'composition', 'long_head', 'long_middle', 'long_tail']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def long_prompt(prompt, position):
    # Style-only pressure test, NOT natural long-caption data. No new depicted entities.
    phrases = [
        'Use a restrained visual style with balanced contrast and consistent saturation throughout.',
        'Maintain smooth tonal transitions and natural color grading without abrupt stylistic changes.',
        'Keep the presentation calm and coherent while preserving clarity and fine visual detail.',
        'Prefer subtle highlights and gentle shading rather than excessive contrast or oversaturation.',
        'Preserve a consistent aesthetic and a steady pace throughout the entire visual presentation.',
        'Use realistic rendering with careful tonal balance and avoid distracting visual embellishments.',
        'Keep the overall appearance clean with moderate sharpness and unobtrusive stylistic treatment.',
        'Preserve accurate proportions and use consistent exposure without artificial color shifts.',
        'The desired treatment should remain understated, visually legible, and stylistically consistent.',
    ] * 2
    offset = {'head': 0, 'middle': len(phrases) // 2, 'tail': len(phrases)}[position]
    phrases.insert(offset, prompt)
    text = ' '.join(phrases)
    assert 201 <= len(text.split()) <= 400
    return text


def engineered(seeds):
    vocab = load_k400()
    rows = []
    def family(task, index, prompt, target, paraphrase, changed_prompt, changed_target,
               composition_prompt, composition_target, caption=None, changed_caption=None):
        fid = f'{task}-{index:02d}'
        variants = [('canonical', prompt, target, caption), ('paraphrase', paraphrase, target, caption),
                    ('changed', changed_prompt, changed_target, changed_caption if changed_caption is not None else caption),
                    ('composition', composition_prompt, composition_target, caption)]
        variants += [(f'long_{p}', long_prompt(prompt, p), target, caption) for p in ('head', 'middle', 'tail')]
        for variant, text, expected, cap in variants:
            rows.append({'id': f'{fid}/{variant}', 'family': fid, 'task': task,
                         'source': 'engineered', 'label_quality': 'author_constructed_contract',
                         'variant': variant, 'input': {'prompt': text, **({'caption': cap} if cap is not None else {})},
                         'target': expected, 'prompt_words': len(text.split())})
    for i, (a, b, c) in enumerate(seeds['entities']):
        rel = ('left', 'right', 'above', 'below')[i % 4]
        opp = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}[rel]
        phrase = lambda r: {'left': 'to the left of', 'right': 'to the right of', 'above': 'above', 'below': 'below'}[r]
        triple = lambda r, s=a, o=b: {'subject': s, 'relation': r, 'object': o}
        target = {'relationships': [triple(rel)]}
        para = f'Relative to the {b}, the {a} is positioned {phrase(rel)} it.'
        family('spatial', i, f'The {a} is {phrase(rel)} the {b}.', target, para,
               f'The {a} is {phrase(opp)} the {b}.', {'relationships': [triple(opp)]},
               f'The {a} is {phrase(rel)} the {b}. The {b} is above the {c}.',
               {'relationships': [triple(rel), triple('above', b, c)]})
        names = [canonical_entity_name(x) for x in (a, b, c)]
        family('objects', i, f'A {a} and a {b}.', {'entities': names[:2]},
               f'Both the {b} and the {a} are visible together.',
               f'A {a} and a {c}.', {'entities': [names[0], names[2]]},
               f'A {a}, a {b}, and a {c}.', {'entities': names})
    for i, (name, paraphrase) in enumerate(seeds['actions']):
        other = seeds['actions'][(i + 1) % len(seeds['actions'])][0]
        name, other = vocab.resolve(name), vocab.resolve(other)
        assert name and other
        family('action', i, f'A person is {name}.', {'actions': [name]},
               f'A person is {paraphrase}.', f'A person is {other}.', {'actions': [other]},
               f'One person is {name}, while another person is {other}.', {'actions': [name, other]})
    for i, (scene, paraphrase) in enumerate(seeds['scenes']):
        label = ('supported', 'contradicted', 'insufficient')[i % 3]
        supported = f'This is a {scene}. The lighting is bright.'
        cap = {'supported': supported, 'contradicted': f'This is not a {scene}; that setting is explicitly absent.',
               'insufficient': 'The image is too blurred to determine the setting or the lighting.'}[label]
        changed = 'contradicted' if label == 'supported' else 'supported'
        changed_cap = f'This is not a {scene}; that setting is explicitly absent.' if changed == 'contradicted' else supported
        family('scene', i, f'A {scene}.', label, f'A {paraphrase}.', f'A {scene}.', changed,
               f'A {scene} with bright lighting.', label, cap, changed_cap)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', default='data/ablation-v1/seeds.json')
    p.add_argument('--out', default='data/ablation-v1/eval.jsonl')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() or out.with_suffix('.manifest.json').exists():
        raise SystemExit('frozen evaluation already exists; refusing overwrite')
    seeds = json.loads(Path(args.seeds).read_text())
    assert all(len(v) == 24 for v in seeds.values())
    rows = engineered(seeds)
    hashes, training_text, dev_records = {}, set(), []
    for task, version in VERSIONS.items():
        for split in ('train', 'dev'):
            path = Path(f'data/formal/{version}/{task}/{split}.jsonl')
            hashes[str(path)] = sha(path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for r in records:
                training_text.update(norm(x) for x in r['input'].values() if isinstance(x, str))
            if split == 'dev':
                # Deterministic source-family sample; existing dev is not a fresh blind test.
                seen = set()
                for r in sorted(records, key=lambda r: hashlib.sha256(r['sample_id'].encode()).hexdigest()):
                    if r['group_id'] in seen:
                        continue
                    seen.add(r['group_id'])
                    dev_records.append({'id': f'dev/{task}/{r["sample_id"]}', 'family': r['group_id'],
                                        'task': task, 'source': 'existing_dev', 'variant': 'natural',
                                        'label_quality': r.get('quality'), 'input': r['input'], 'target': r['target'],
                                        'prompt_words': len(r['input']['prompt'].split())})
                    if len(seen) == 48:
                        break
    overlaps = Counter()
    for r in rows:
        r['exact_train_dev_prompt_overlap'] = norm(r['input']['prompt']) in training_text
        if r['exact_train_dev_prompt_overlap']:
            overlaps[f'{r["task"]}/{r["variant"]}'] += 1
    # SNLI location-focused full-hypothesis NLI transfer diagnostic. Original human labels.
    import pyarrow.parquet as pq
    snli = Path('data/raw/snli/test-00000-of-00001.parquet')
    hashes[str(snli)] = sha(snli)
    candidates = pq.read_table(snli).to_pylist()
    locations = r'outside|inside|outdoors|indoors|kitchen|classroom|library|bedroom|bathroom|restaurant|forest|beach|desert|mountain|park|street|room|building|stadium|shop|store|field|water|pool|ocean|lake|river'
    location_form = re.compile(r'^(?:the people|people|a person|the person|they|everyone|someone|the man|a man|the woman|a woman|the men|the women|two people|some people) (?:is|are) (?:in |at |on |inside|outside|indoors|outdoors)', re.I)
    eligible = [r for r in candidates if r['label'] in (0, 1, 2)
                and location_form.search(r['hypothesis']) and re.search(r'\b(?:' + locations + r')\b', r['hypothesis'], re.I)
                and len(r['hypothesis'].split()) <= 14
                and not ({norm(r['premise']), norm(r['hypothesis'])} & training_text)]
    counts, premises = Counter(), set()
    natural = []
    for r in sorted(eligible, key=lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()):
        if counts[r['label']] >= 30 or norm(r['premise']) in premises:
            continue
        counts[r['label']] += 1
        premises.add(norm(r['premise']))
        key = hashlib.sha256(r['premise'].encode()).hexdigest()[:16]
        natural.append({'id': f'snli/{key}', 'family': f'snli-premise/{key}', 'task': 'scene',
                        'source': 'snli_test_locations', 'label_quality': 'original_snli_human_nli_transfer',
                        'variant': 'natural', 'input': {'prompt': r['hypothesis'], 'caption': r['premise']},
                        'target': {0: 'supported', 1: 'insufficient', 2: 'contradicted'}[r['label']],
                        'prompt_words': len(r['hypothesis'].split())})
    rows.extend(natural)
    rows.extend(dev_records)
    assert len({r['id'] for r in rows}) == len(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
    manifest = {'protocol': 'docs/plans/16-ablation-generalization.md', 'protocol_sha256': sha('docs/plans/16-ablation-generalization.md'),
                'builder_sha256': sha(__file__), 'seeds_sha256': sha(args.seeds), 'data_sha256': sha(out),
                'rows': len(rows), 'counts': dict(Counter(f'{r["task"]}/{r["source"]}' for r in rows)),
                'exact_train_dev_prompt_overlap': dict(overlaps), 'input_sha256': hashes,
                'expected_groups': [f'{task}/{v}' for task in VERSIONS for v in VARIANTS if v != 'canonical'],
                'snli_selected_labels': dict(counts), 'snli_original_test_rows': len(candidates),
                'caveats': ['Engineering authored after method development, frozen before new predictions; not human gold.',
                            'Source disjointness does not imply pretrained-model independence.',
                            'SNLI is an NLI transfer diagnostic, not Tag2Text evidence accuracy.',
                            'Long prompts use repeated style prose, not naturally occurring long prompts.',
                            'Existing dev has been logged during training; descriptive only.']}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
