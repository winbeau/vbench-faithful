#!/usr/bin/env python3
"""Scene scorecard on caption-evidence labels, with a separate visual-label audit.

Exact official auxiliary keys/case-sensitive substring rule, common binary support
metric, separate three-way classification, source-component bootstrap of the same
estimator, and correct-and-consistent synonyms. Missing outputs and labels remain
in the frozen denominator; unknown-label accuracy is explicitly a lower bound.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.experiments import (SCENE_SYNONYMS, bare_scene, cluster_summary,
    metadata_index, official_target, scene_rule_key, scene_rule_caption, text_key)
from vbench_prompts_compile.official_replay import scene_scores, UPSTREAM_SHA
import audit_scene_sources as audit

SCENE_LABELS = ('supported', 'contradicted', 'insufficient')
SCHEMES = ('Origin', 'Repair-rule', 'Repair-model')


def official_label(scene_key, caption):
    if scene_key is None:
        return None
    return 'supported' if scene_scores(scene_key, [caption])[0] else 'contradicted'


def normalize_record(row):
    return {**row, 'input': row.get('input', {'prompt': row.get('prompt'), 'caption': row.get('caption')}),
        'meta': {**row.get('meta', {}), 'image': row.get('meta', {}).get('image', row.get('image'))}}


def build_conditions(gold_rows, synonym_rows=None, *, plan_rows=None):
    gold_rows = [normalize_record(r) for r in gold_rows]
    planned = [normalize_record(r) for r in (plan_rows if plan_rows is not None else gold_rows)]
    by_id = defaultdict(list)
    for row in gold_rows:
        by_id[J.observation_id(row)].append(row)
    components = audit.components(planned)
    conditions = []
    for row, family in zip(planned, components):
        labels = by_id.get(J.observation_id(row), [])
        expected = {r.get('meta', {}).get('caption_evidence_label') for r in labels} - {None}
        visual = {r.get('meta', {}).get('visual_truth') for r in labels} - {None}
        conditions.append({'condition': row.get('meta', {}).get('pair_type', 'unknown'),
            'expected': next(iter(expected)) if len(expected) == 1 else None,
            'visual_expected': next(iter(visual)) if len(visual) == 1 else None,
            'prompt': row['input']['prompt'], 'caption': row['input']['caption'],
            'family': family, 'source_id': J.observation_id(row),
            'image': row['meta'].get('image'), 'video_path': row['meta'].get('frame_video'),
            'frame_index': row['meta'].get('frame_index')})
    originals = list(conditions)
    unmatched = 0
    if synonym_rows is None:
        for row in originals:
            key = bare_scene(row['prompt'])
            if key in SCENE_SYNONYMS:
                synonym = SCENE_SYNONYMS[key]
                article = 'an ' if synonym[0].lower() in 'aeiou' else 'a '
                conditions.append({**row, 'condition': 'synonym', 'original_prompt': row['prompt'],
                    'original_expected': row['expected'], 'prompt': article + synonym,
                    'source_id': row['source_id'] + ':synonym'})
    else:
        for synonym in synonym_rows:
            matched = [r for r in originals if r['prompt'] == synonym['original_prompt'] and r['caption'] == synonym['caption']]
            if not matched:
                unmatched += 1
            for row in matched:
                conditions.append({**row, 'condition': 'synonym', 'original_prompt': row['prompt'],
                    'original_expected': row['expected'], 'prompt': synonym['variant_prompt'],
                    'source_id': row['source_id'] + ':synonym'})
    return conditions, unmatched


def binary_ok(label, expected):
    return label in SCENE_LABELS and expected in SCENE_LABELS and ((label == 'supported') == (expected == 'supported'))


def score_rows(conditions, index, predictions):
    scored = []
    for row in conditions:
        base_prompt = row.get('original_prompt', row['prompt'])
        base_key = official_target(index, 'scene', base_prompt)
        scene_key = bare_scene(row['prompt']) if row['condition'] == 'synonym' and base_key else base_key
        # Dictionary normalisation is deliberately disclosed as a closed-lexicon control.
        labels = {'Origin': official_label(scene_key, row['caption']),
                  'Repair-rule': official_label(scene_rule_key(scene_key), scene_rule_caption(row['caption'])) if scene_key else None,
                  'Repair-model': predictions.get(text_key('scene', row['prompt'], row['caption']))}
        base_labels = {'Origin': official_label(base_key, row['caption']),
                      'Repair-rule': official_label(scene_rule_key(base_key), scene_rule_caption(row['caption'])) if base_key else None,
                      'Repair-model': predictions.get(text_key('scene', base_prompt, row['caption']))}
        values = {}
        for scheme, label in labels.items():
            base = base_labels[scheme]
            values[scheme] = {'label': label, 'base_label': base, 'binary_ok': binary_ok(label, row['expected']),
                'three_way_ok': label in SCENE_LABELS and label == row['expected'],
                'visual_binary_ok': binary_ok(label, row['visual_expected']),
                'correct_and_consistent': binary_ok(label, row['expected']) and binary_ok(base, row['expected'])
                                          and (label == 'supported') == (base == 'supported'),
                'raw_consistent': label in SCENE_LABELS and base in SCENE_LABELS and label == base,
                'abstained': label == 'insufficient', 'missing': label is None,
                'support_score': int(label == 'supported'),
                'paired_support_delta': int(label == 'supported') - int(base == 'supported')}
        scored.append({**row, 'official_key': scene_key, 'official_applicable': scene_key is not None, 'schemes': values})
    return scored


def metrics(rows, scheme):
    flat = [{**r, **r['schemes'][scheme]} for r in rows]
    binary = [{**r, 'expected': 'supported' if r['expected'] == 'supported' else 'not_supported'}
              for r in flat if r['expected'] in SCENE_LABELS]
    three = [r for r in flat if r['expected'] in SCENE_LABELS]
    return {'binary_accuracy_full_denominator': cluster_summary(flat, 'binary_ok'),
        'balanced_binary_labelled_only': cluster_summary(binary, 'binary_ok', strata=['supported', 'not_supported']),
        'three_way_accuracy_full_denominator': cluster_summary(flat, 'three_way_ok') if scheme == 'Repair-model' else None,
        'balanced_three_way_labelled_only': cluster_summary(three, 'three_way_ok', strata=SCENE_LABELS) if scheme == 'Repair-model' else None,
        'label_counts': dict(Counter(r['label'] for r in flat)),
        'prediction_coverage': sum(not r['missing'] for r in flat) / len(flat) if flat else None,
        'abstentions': sum(r['abstained'] for r in flat),
        'label_coverage': len(three) / len(flat) if flat else None}


def baselines(rows):
    out = {}
    for label in SCENE_LABELS:
        flat = [{**r, 'correct': binary_ok(label, r['expected']), 'three_way': label == r['expected']} for r in rows]
        out['always_' + label] = {'binary': cluster_summary(flat, 'correct'), 'three_way': cluster_summary(flat, 'three_way')}
    return out


def summarize(rows, unmatched):
    originals = [r for r in rows if r['condition'] != 'synonym']
    comparable = [r for r in originals if r['official_applicable']]
    synonyms = [r for r in rows if r['condition'] == 'synonym']
    synonym_metrics = {}
    for scheme in SCHEMES:
        flat = [{**r, **r['schemes'][scheme]} for r in synonyms]
        synonym_metrics[scheme] = {k: cluster_summary(flat, k) for k in ['correct_and_consistent', 'raw_consistent', 'paired_support_delta']}
        # Copying a base verdict guarantees consistency but cannot fix its error.
        copies = [{**r, 'correct': binary_ok(r['schemes'][scheme]['base_label'], r['expected'])} for r in synonyms]
        synonym_metrics[scheme]['copy_original_correct_and_consistent'] = cluster_summary(copies, 'correct')
    visual = [r for r in originals if r['visual_expected'] in SCENE_LABELS]
    paired = {}
    for scheme in ['Repair-rule', 'Repair-model']:
        differences = [{**r, 'difference':int(r['schemes'][scheme]['binary_ok']) - int(r['schemes']['Origin']['binary_ok'])}
                       for r in comparable]
        balanced = [{**r, 'expected':'supported' if r['expected']=='supported' else 'not_supported'}
                    for r in differences if r['expected'] in SCENE_LABELS]
        paired[scheme] = {'binary_accuracy_difference':cluster_summary(differences, 'difference'),
                         'balanced_binary_difference_labelled_only':cluster_summary(balanced,'difference',strata=['supported','not_supported'])}
    return {'planned_observations': len(originals), 'source_components': len({r['family'] for r in originals}),
        'label_missing': sum(r['expected'] not in SCENE_LABELS for r in originals),
        'unmatched_synonym_pairs': unmatched, 'official_defined_observations': len(comparable),
        'primary_common_binary': {s: metrics(comparable, s) for s in SCHEMES},
        'paired_against_origin':paired,
        'model_confusion':{label:dict(Counter(r['schemes']['Repair-model']['label'] for r in originals if r['expected']==label)) for label in SCENE_LABELS},
        'full_plan': {s: metrics(originals, s) for s in SCHEMES},
        'conditions': {c: {s: metrics([r for r in rows if r['condition'] == c], s) for s in SCHEMES}
                       for c in sorted({r['condition'] for r in rows})},
        'synonyms': synonym_metrics, 'constant_baselines_common_binary': baselines(comparable),
        'constant_baselines_full_plan': baselines(originals),
        'visual_label_diagnostic': {s: cluster_summary([{**r, **r['schemes'][s]} for r in visual], 'visual_binary_ok') for s in SCHEMES},
        'upstream_commit': UPSTREAM_SHA,
        'caveats': ['LLM-annotated evidence and visual labels are separate domains, neither is human gold.',
            'Missing labels count as failure in full-denominator accuracy, making it a lower bound; balanced metrics report labelled coverage.',
            'Origin/Repair-rule cannot emit insufficient; only binary support comparison shares their output domain.',
            'Scene-less prompts have no official auxiliary key; shown separately and as missing in the full plan.',
            'Connected source components, including both cross-pair endpoints, are bootstrap units; fewer than two gives no CI.',
            'Synonym labels inherit the base evidence annotation and are not independently human-verified.',
            'Repair-rule uses a declared closed synonym dictionary; copy-original and constant controls remain visible.']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold', required=True, help='caption-evidence annotation JSONL')
    p.add_argument('--plan', help='full frozen visual item plan; retain missing annotations')
    p.add_argument('--synonyms', help='optional explicit pairs; default is strict frozen dictionary')
    p.add_argument('--metadata', required=True, help='pinned official VBench_full_info.json')
    p.add_argument('--predictions', help='batch_predict JSONL; omission records all model outputs as missing')
    p.add_argument('--emit-inputs', help='save all original/synonym prompt-caption pairs for inference')
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    gold = J.read_jsonl(Path(args.gold))
    plan = J.read_jsonl(Path(args.plan)) if args.plan else None
    synonyms = J.read_jsonl(Path(args.synonyms)) if args.synonyms else None
    conditions, unmatched = build_conditions(gold, synonyms, plan_rows=plan)
    if args.emit_inputs:
        from vbench_prompts_compile.records import write_jsonl
        write_jsonl(Path(args.emit_inputs), [{k: r[k] for k in ['prompt', 'caption', 'source_id']} for r in conditions])
    index = metadata_index(json.loads(Path(args.metadata).read_text()))
    predictions = {r.get('request_id', text_key('scene', r['prompt'], r['caption'])): r.get('target')
                   for r in J.read_jsonl(Path(args.predictions))} if args.predictions else {}
    rows = score_rows(conditions, index, predictions)
    summary = summarize(rows, unmatched)
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for name, path in vars(args).items() if name in {'gold', 'plan', 'synonyms', 'metadata', 'predictions'} and path}
    J.atomic_json(Path(args.output), {'summary': summary, 'input_sha256': hashes, 'rows': rows})
    print(json.dumps({k: summary[k] for k in ['planned_observations', 'source_components', 'label_missing', 'official_defined_observations']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
