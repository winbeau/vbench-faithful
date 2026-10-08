#!/usr/bin/env python3
"""Quarantine legacy K400 template families exposed to the frozen test matrix.

No labels/predictions/visual scores are read. Exclude every exact test or derived
prompt and its source group, plus K400 template families for the test and class-swap
IDs. Independent natural examples may share a class; the 400-name vocabulary is
still the declared output inventory. Test prompts and splits are never modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/prompt-compiler/src'))
sys.path.insert(0, str(ROOT))
from scripts.semantic.paths import DATA_ROOT, RAW_ROOT, FIXTURES_ROOT
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import records as R
from vbench_prompts_compile.sources import load_k400


def normalized(prompt):
    return ' '.join(re.findall(r'\w+', prompt.lower()))


def template_id(row):
    if row.get('meta', {}).get('k400_id') is not None:
        return int(row['meta']['k400_id'])
    for key, pattern in [('source_id', r'^k400-(\d+)$'), ('group_id', r'^k400-class-(\d+)$')]:
        match = re.match(pattern, row.get(key, ''))
        if match:
            return int(match.group(1))
    return None


def isolate(splits, matrix, vocab):
    reserved_texts = {normalized(r['prompt']) for r in matrix} | {normalized(r['original_prompt']) for r in matrix}
    reserved_labels = set()
    for row in matrix:
        for value in [row.get('base_official_target'), row.get('official_target')]:
            if isinstance(value, str) and (resolved := vocab.resolve(value)):
                reserved_labels.add(resolved)
    reserved_ids = {r['id'] for r in vocab.entries if r['label'] in reserved_labels}
    all_rows = [r for rows in splits.values() for r in rows]
    exposed_groups = {r['group_id'] for r in all_rows if normalized(r['input']['prompt']) in reserved_texts
                      or template_id(r) in reserved_ids}
    kept, excluded = {}, []
    for name, rows in splits.items():
        kept[name] = []
        for row in rows:
            if row['group_id'] in exposed_groups:
                excluded.append({**row, 'original_split': name, 'exclusion_reason': 'frozen_test_prompt_or_template_family'})
            else:
                kept[name].append(row)
    assert not ({r['group_id'] for r in kept['train']} & {r['group_id'] for r in kept['dev']})
    assert not any(normalized(r['input']['prompt']) in reserved_texts for rows in kept.values() for r in rows)
    assert not any(template_id(r) in reserved_ids for rows in kept.values() for r in rows)
    return kept, excluded, {'reserved_template_ids': sorted(reserved_ids), 'reserved_texts': len(reserved_texts),
        'excluded_source_groups': len(exposed_groups),
        'legacy_exact_test_prompt_families': len({r['original_prompt'] for r in matrix if r['transform']=='identity'
            and normalized(r['original_prompt']) in {normalized(t['input']['prompt']) for t in splits['train']}})}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=str(DATA_ROOT / 'formal/v5/action'))
    p.add_argument('--matrix', required=True)
    p.add_argument('--out', default=str(DATA_ROOT / 'formal/v9/action'))
    args = p.parse_args(argv)
    out = Path(args.out)
    if (out / 'manifest.json').exists():
        raise ValueError('output already frozen; choose a new version')
    paths = {name: Path(args.baseline) / (name + '.jsonl') for name in ['train','dev']}
    splits = {k: J.read_jsonl(v) for k,v in paths.items()}
    if any(not v for v in splits.values()):
        raise ValueError('missing training snapshot')
    matrix = J.read_jsonl(Path(args.matrix))
    kept, excluded, audit = isolate(splits, matrix, load_k400())
    manifest = {'policy': 'exclude source groups before retraining; unchanged 300-step v5 hyperparameters; final only; no prediction-driven selection',
        'baseline_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        'matrix_sha256': hashlib.sha256(Path(args.matrix).read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit': audit, 'splits': {}, 'exclusions': dict(Counter(r['original_split'] for r in excluded))}
    for name, rows in kept.items():
        if not rows:
            raise ValueError('source isolation removed entire split')
        for row in rows:
            if not R.validate_record(row).ok:
                raise ValueError('invalid training record')
        path = out / (name + '.jsonl')
        R.write_jsonl(path, rows)
        manifest['splits'][name] = {'rows':len(rows), 'groups':len({r['group_id'] for r in rows}),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'sources':dict(Counter(r['source'] for r in rows))}
    R.write_jsonl(out / 'excluded.jsonl', excluded)
    J.atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
