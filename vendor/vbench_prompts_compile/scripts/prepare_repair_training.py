#!/usr/bin/env python3
"""Freeze v8 training data under caption-evidence and four-direction contracts.

No predictions or scores are consumed. Historical files stay untouched. Rejected
source crossings and removed weak relations are retained in the manifest/artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import records as R
from vbench_prompts_compile import training as T
import audit_scene_sources as audit

DIRECTION_MAP = {'under': 'below', 'beneath': 'below', 'underneath': 'below', 'over': 'above'}


def spatial_row(row):
    original = row['target']
    relationships = []
    omitted = []
    for relation in original['relationships']:
        mapped = DIRECTION_MAP.get(relation['relation'], relation['relation'])
        if mapped == 'on':
            obj = re.escape(relation['object'])
            for word, direction in [('top', 'above'), ('bottom', 'below')]:
                if re.search(rf'\bon (?:the )?{word} of (?:a |an |the )?{obj}\b', row['input']['prompt'], re.I):
                    mapped = direction
        if mapped in R.DIRECTION_RELATIONS:
            value = {**relation, 'relation': mapped}
            if value not in relationships:
                relationships.append(value)
        else:
            omitted.append(relation)
    return {**row, 'target': {'relationships': relationships}, 'meta': {
        **row.get('meta', {}), 'spatial_relation_space': 'four_directions',
        'legacy_target': original, 'omitted_unsupported_relations': omitted,
        'target_policy': 'explicit directions; under/beneath/underneath -> below; over -> above; on (the) top/bottom of the named object -> above/below; other weak relations omitted'}}


def prompt_keys(row):
    return {k for k in audit.source_keys(row) if k.startswith('prompt:')}


def scene_partition(rows, test_rows, frozen_test_prompts, seed=20260919):
    excluded_keys = set().union(*(audit.source_keys(r) for r in test_rows))
    excluded_keys.update('prompt:' + audit.normalized(p) for p in frozen_test_prompts)
    excluded, eligible = [], []
    for row in rows:
        overlap = audit.source_keys(row) & excluded_keys
        if overlap:
            excluded.append({**row, 'exclusion_reason': 'reserved_test_source', 'overlap_hashes': [R.sha256_text(k) for k in sorted(overlap)]})
        else:
            eligible.append(row)
    # Allocate source prompts before selecting observations; both pairing endpoints
    # must land in the same partition. Hash order is fixed without reading labels.
    def rank(key):
        return R.sha256_text(f'{seed}|scene-v8|{key}')
    evidence_keys = {'prompt:' + audit.normalized(r['meta']['frame_source_prompt']) for r in eligible if r.get('meta', {}).get('frame_source_prompt')}
    evidence_order = sorted(evidence_keys, key=rank)
    ndev = max(1, round(len(evidence_order) * .2)) if evidence_order else 0
    dev_keys = set(evidence_order[:ndev])
    other_keys = set().union(*(prompt_keys(r) for r in eligible)) - evidence_keys
    dev_keys.update(key for key in other_keys if int(rank(key)[:8], 16) % 5 == 0)
    split = {'train': [], 'dev': []}
    for row in eligible:
        keys = prompt_keys(row)
        if keys & dev_keys and keys - dev_keys:
            excluded.append({**row, 'exclusion_reason': 'train_dev_pair_crossing'})
            continue
        name = 'dev' if keys <= dev_keys else 'train'
        split[name].append(row)
    # Fixtures/paraphrases may share an explicit source group beyond their text.
    # Keep the whole group on the deterministic side and quarantine crossings.
    groups = {name: {r['group_id'] for r in values} for name, values in split.items()}
    shared_groups = groups['train'] & groups['dev']
    if shared_groups:
        for name in split:
            keep = []
            for row in split[name]:
                preferred = 'dev' if int(rank('group:' + row['group_id'])[:8], 16) % 5 == 0 else 'train'
                if row['group_id'] in shared_groups and name != preferred:
                    excluded.append({**row, 'exclusion_reason': 'fixture_group_crossing'})
                else:
                    keep.append(row)
            split[name] = keep
    result = audit.audit({**split, 'test_reserved': test_rows})
    if any(v['source_count'] for v in result['overlaps'].values()):
        raise ValueError('source closure leakage after partition')
    for name, values in split.items():
        components = audit.components(values)
        split[name] = [{**row, 'group_id': component, 'meta': {**row.get('meta', {}), 'legacy_group_id': row['group_id'], 'split': name}} for row, component in zip(values, components)]
    return split, excluded, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['spatial', 'scene'])
    parser.add_argument('--baseline', default='data/formal/v7')
    parser.add_argument('--out', default='data/formal/v8')
    parser.add_argument('--evidence')
    parser.add_argument('--test-plan')
    parser.add_argument('--frozen-split', default='../vbench-audit/splits/e0_prompt_split.csv')
    args = parser.parse_args(argv)
    out = Path(args.out) / args.task
    if (out / 'manifest.json').exists():
        raise SystemExit('frozen output already exists; choose a new version')
    inputs = {}
    baseline = {}
    for split in ['train', 'dev']:
        path = Path(args.baseline) / args.task / f'{split}.jsonl'
        baseline[split] = R.read_jsonl(path)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    excluded = []
    source_audit = None
    if args.task == 'spatial':
        data = {split: [spatial_row(r) for r in rows] for split, rows in baseline.items()}
    else:
        if not args.evidence or not args.test_plan:
            raise SystemExit('scene requires --evidence and --test-plan')
        evidence_path = Path(args.evidence)
        report = json.loads(evidence_path.with_suffix('.report.json').read_text())
        if not report.get('complete') or report.get('missing'):
            raise SystemExit('evidence annotation is incomplete; refuse partial training data')
        rows = R.read_jsonl(evidence_path)
        for row in rows:
            if row['target'] != row.get('meta', {}).get('caption_evidence_label'):
                raise ValueError('scene training target is not the caption evidence label')
        for row in baseline['train'] + baseline['dev']:
            if row['source'] == 'engineering_fixture':
                rows.append({**row, 'meta': {**row.get('meta', {}), 'caption_evidence_label': row['target'], 'label_source': 'handwritten_engineering_fixture'}})
        test_rows = R.read_jsonl(args.test_plan)
        frozen_rows = list(csv.DictReader(Path(args.frozen_split).open()))
        frozen_test = [r['prompt_id'] for r in frozen_rows if r['dimension'] == 'scene' and r['split'] == 'test']
        data, excluded, source_audit = scene_partition(rows, test_rows, frozen_test)
        for p in [args.evidence, args.test_plan, args.frozen_split]:
            inputs[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if {r['group_id'] for r in data['train']} & {r['group_id'] for r in data['dev']}:
        raise ValueError('train/dev group overlap')
    manifest = {'task': args.task, 'inputs_sha256': inputs, 'splits': {}, 'source_audit': source_audit,
                'excluded': dict(Counter(r['exclusion_reason'] for r in excluded)),
                'policy': 'frozen label rules and source partition; no model outputs or metrics used; final fixed-step checkpoint'}
    for split, rows in data.items():
        if not rows:
            raise ValueError(f'empty {split}')
        for row in rows:
            checked = R.validate_record(row)
            if not checked.ok:
                raise ValueError(f"{row['sample_id']}: {checked.errors}")
        path = out / f'{split}.jsonl'
        R.write_jsonl(path, rows)
        T.load_records(path, args.task)
        manifest['splits'][split] = {'rows': len(rows), 'families': len({r['group_id'] for r in rows}),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'labels': dict(Counter(str(r['target']) for r in rows)) if args.task == 'scene' else dict(Counter(t['relation'] for r in rows for t in r['target']['relationships']))}
    R.write_jsonl(out / 'excluded.jsonl', excluded)
    J.atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
