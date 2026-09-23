#!/usr/bin/env python3
"""Audit Scene observation identity, both pairing endpoints and source components."""
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
from vbench_prompts_compile import annotation_jobs as J


def normalized(text):
    return ' '.join(re.findall(r'\w+', str(text).lower()))


def source_keys(row):
    inputs = row.get('input', row)
    meta = row.get('meta', {})
    keys = {'prompt:' + normalized(inputs['prompt'])}
    if meta.get('frame_source_prompt'):
        keys.add('prompt:' + normalized(meta['frame_source_prompt']))
    if meta.get('frame_video'):
        keys.add('video:' + str(meta['frame_video']).split('/videos/')[-1])
    if meta.get('image') or row.get('image'):
        keys.add('image:' + str(meta.get('image') or row.get('image')))
    return keys


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in rows:
        keys = sorted(source_keys(row))
        find(keys[0])
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    members = {}
    for key in parent:
        members.setdefault(find(key), []).append(key)
    names = {root: 'source-component:' + hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()[:16]
             for root, keys in members.items()}
    return [names[find(sorted(source_keys(row))[0])] for row in rows]


def audit(sets):
    report = {'sets': {}, 'overlaps': {}, 'note': 'shared prompt/video/frame sources define dependency components; observations are not independent families'}
    keys = {}
    for name, rows in sets.items():
        keys[name] = set().union(*(source_keys(row) for row in rows)) if rows else set()
        ids = [r.get('sample_id', r.get('item_id')) for r in rows]
        report['sets'][name] = {'rows': len(rows), 'unique_ids': len(set(ids)),
            'source_prompts': sum(k.startswith('prompt:') for k in keys[name]),
            'evidence_prompts': len({normalized(r.get('meta', {}).get('frame_source_prompt', '')) for r in rows} - {''}),
            'dependency_components': len(set(components(rows))),
            'component_sizes': sorted(Counter(components(rows)).values(), reverse=True)}
    for a, ak in keys.items():
        for b, bk in keys.items():
            if a >= b:
                continue
            overlap = sorted(ak & bk)
            report['overlaps'][f'{a}:{b}'] = {'source_count': len(overlap),
                'by_type': dict(Counter(k.split(':', 1)[0] for k in overlap)),
                'source_hashes': [hashlib.sha256(k.encode()).hexdigest() for k in overlap]}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', action='append', required=True, help='name=JSONL, repeat for each split')
    parser.add_argument('--out', required=True)
    parser.add_argument('--require-disjoint', action='store_true')
    args = parser.parse_args(argv)
    paths = dict(value.split('=', 1) for value in args.set)
    result = audit({name: J.read_jsonl(Path(path)) for name, path in paths.items()})
    result['input_sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}
    J.atomic_json(Path(args.out), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'overlaps'}, indent=2))
    overlaps = {pair: value['source_count'] for pair, value in result['overlaps'].items()}
    print(json.dumps({'overlapping_sources': overlaps}))
    return 2 if args.require_disjoint and any(overlaps.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
