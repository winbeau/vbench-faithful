#!/usr/bin/env python3
"""Audit Objects temporal confirmation without treating abstention as absence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J, records as R
from vbench_prompts_compile.experiments import cluster_summary, text_key
from vbench_prompts_compile.objects_repair import video_scores
from score_matrix import score_one, paired_rows, native_entity_codec
from cache_matrix_transforms import evidence_id
from analyze_object_controls import matched_contrasts
from render_matrix_tables import table, ci_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError('duplicate ' + key + ': ' + str(identity))
        result[identity] = row
    return result


def regress(previous_path, current_path):
    unchanged, before, after = Counter(), [], []
    total = 0
    with Path(previous_path).open() as old, Path(current_path).open() as new:
        for a, b in itertools.zip_longest(old, new):
            if a is None or b is None:
                raise ValueError('changed matrix denominator')
            a, b = json.loads(a), json.loads(b)
            if (a['sample_id'], a['scheme']) != (b['sample_id'], b['scheme']):
                raise ValueError('changed observation identity/order')
            total += 1
            if a['task'] != 'objects' or a['scheme'] == 'Origin':
                if a != b:
                    raise ValueError('untargeted record changed: ' + a['sample_id'])
                unchanged[a['task'] + '/' + a['scheme']] += 1
            if a['task'] == 'objects':
                before.append(a)
                after.append(b)
    return {'total_rows': total, 'untouched_rows_exact': dict(unchanged)}, before, after


def frame_truth(target, other):
    """Three-state conjunction of independently stored visibility judgments."""
    if target not in {'visible', 'not_visible', 'uncertain'} or other not in {'visible', 'not_visible', 'uncertain'}:
        raise ValueError('invalid visibility label')
    if target == other == 'visible':
        return 'both_visible'
    if 'not_visible' in (target, other):
        return 'at_least_one_not_visible'
    return 'uncertain'


def legacy_result(entities, evidence):
    frames = evidence.get('frame_labels', [])
    states = ['missing' if i >= len(frames) or frames[i] is None else
              'supported' if all(name in frames[i] for name in entities) else 'absent' for i in range(16)]
    scores = [float(v == 'supported') for v in states]
    return {'score': sum(scores)/16, 'coverage': 1-states.count('missing')/16,
            'abstention': 0., 'frame_scores': scores,
            'objects_resolution': {'frame_states': states}}


def summarize_metrics(rows, keys):
    return {key: cluster_summary(rows, key) for key in keys}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'cache', 'transforms', 'visibility', 'predictions', 'metadata',
                 'previous-paired', 'paired', 'score-report', 'out'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args(argv)
    regression, before, after = regress(args.previous_paired, args.paired)
    matrix = indexed([r for r in J.read_jsonl(Path(args.matrix)) if r['task'] == 'objects'], 'sample_id')
    caches = indexed(J.read_jsonl(Path(args.cache)), 'relative_path')
    transforms = indexed(J.read_jsonl(Path(args.transforms)), 'evidence_id')
    reviews = indexed(J.read_jsonl(Path(args.visibility)), 'evidence_id')
    predictions = {text_key('objects', r['prompt']): r['target'] for r in
                   indexed(J.read_jsonl(Path(args.predictions)), 'request_id').values()}
    codec = native_entity_codec(json.loads(Path(args.metadata).read_text()))
    report = json.loads(Path(args.score_report).read_text())
    if not report.get('execution_complete') or report.get('objects_backend') != 'repair-v2':
        raise ValueError('complete Objects v2 execution required')
    # This audit must be attached to exactly the inputs/code of the scored run.
    for path, digest in report['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('score input changed: ' + path)
    for path, digest in report['scorer_source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('scorer source changed: ' + path)
    for key, review in reviews.items():
        digest = hashlib.sha256(json.dumps(transforms.get(key), sort_keys=True).encode()).hexdigest()
        if digest != review.get('transform_record_sha256'):
            raise ValueError('visibility review does not match transformed evidence')
    expected_ids = set(matrix)
    old_by_scheme = {s: indexed([r for r in before if r['scheme'] == s], 'sample_id')
                     for s in ('Origin', 'Repair-rule', 'Repair-model')}
    new_by_scheme = {s: indexed([r for r in after if r['scheme'] == s], 'sample_id') for s in old_by_scheme}
    if any(set(rows) != expected_ids for rows in [*old_by_scheme.values(), *new_by_scheme.values()]):
        raise ValueError('Objects matrix not fully paired')
    legacy_rows = paired_rows(list(matrix.values()), {'objects': caches}, transforms,
                             predictions, None, codec, reviews, objects_backend='legacy-official')
    if legacy_rows != before:
        raise ValueError('full legacy Objects rows no longer reproduce previous run')
    regression['legacy_full_rows_exact'] = len(legacy_rows)
    del legacy_rows
    checked = Counter()
    for sample_id, row in matrix.items():
        evidence = caches[row['relative_path']] if row['evidence_kind'] == 'original' else transforms[evidence_id(row)]
        for scheme in ('Repair-rule', 'Repair-model'):
            old, new = old_by_scheme[scheme][sample_id], new_by_scheme[scheme][sample_id]
            replay = score_one(row, scheme, evidence, predictions, None, codec, objects_backend='legacy-official')
            for field, dest in [('score', 'transformed_score'), ('coverage', 'transformed_coverage'),
                                ('abstention', 'transformed_abstention'), ('frame_scores', 'transformed_frame_scores'),
                                ('target', 'transformed_target'), ('backend_target', 'transformed_backend_target')]:
                if replay.get(field) != old[dest]:
                    raise ValueError('legacy Objects replay mismatch: ' + sample_id)
            if old['original_backend_target'] != new['original_backend_target'] or old['transformed_backend_target'] != new['transformed_backend_target']:
                raise ValueError('object requirement changed')
            if any(a > b for a, b in zip(new['transformed_frame_scores'], old['transformed_frame_scores'])):
                raise ValueError('temporal confirmation invented a positive')
            if old['original_coverage'] != new['original_coverage'] or old['transformed_coverage'] != new['transformed_coverage']:
                raise ValueError('changed backend evidence coverage')
            checked['legacy_replayed_rows'] += 1
            if row['transform'] == 'identity' or row.get('level') == 0:
                if new['original_score'] != new['transformed_score'] or new['original_objects_resolution'] != new['transformed_objects_resolution']:
                    raise ValueError('zero/identity is not exact')
                checked['identity_or_zero_exact_rows'] += 1
        rule, model = new_by_scheme['Repair-rule'][sample_id], new_by_scheme['Repair-model'][sample_id]
        for field in ('original_score', 'transformed_score', 'original_abstention', 'transformed_abstention',
                      'original_backend_target', 'transformed_backend_target', 'original_objects_resolution', 'transformed_objects_resolution'):
            if rule[field] != model[field]:
                raise ValueError('rule/model differ; separate ablation runs required')
    regression.update(checked)
    variants = ('legacy', 'adjacent_labels', 'adjacent_boxes')
    scored = {v: {} for v in variants}
    for sample_id, row in matrix.items():
        evidence = caches[row['relative_path']] if row['evidence_kind'] == 'original' else transforms[evidence_id(row)]
        entities = new_by_scheme['Repair-model'][sample_id]['transformed_backend_target']['entities']
        for variant in variants:
            value = legacy_result(entities, evidence) if variant == 'legacy' else video_scores(
                entities, evidence.get('frame_detections', []), confirmation=variant.removeprefix('adjacent_'))
            scored[variant][sample_id] = value
        actual = new_by_scheme['Repair-model'][sample_id]
        if scored['adjacent_boxes'][sample_id]['objects_resolution'] != actual['transformed_objects_resolution']:
            raise ValueError('live scorer audit replay mismatch: ' + sample_id)
    originals = {r['relative_path']: r for r in matrix.values() if r['transform'] == 'identity'}
    endpoints = [r for r in matrix.values() if r.get('occlusion_control') == 'target' and r.get('level') == 1]
    if len(originals) != len(caches) or len(endpoints) != len(caches) or len(reviews) != len(caches):
        raise ValueError('incomplete source or endpoint denominator')
    impact, endpoint_stats, visibility_stats, residuals = [], [], [], []
    for variant in variants:
        base_values, retained_frames = [], []
        for row in originals.values():
            a, b = scored['legacy'][row['sample_id']], scored[variant][row['sample_id']]
            base_values.append({'family': row['family'], 'score': b['score'], 'delta_from_legacy': b['score']-a['score'],
                'abstention': b['abstention'], 'positive_video_retained': b['score'] > 0, 'legacy_positive_video': a['score'] > 0})
            retained_frames.extend({'family': row['family'], 'retained': new > 0}
                for old, new in zip(a['frame_scores'], b['frame_scores']) if old > 0)
        impact.append({'variant': variant, **summarize_metrics(base_values, ('score', 'delta_from_legacy', 'abstention')),
            'original_positive_frame_retention': cluster_summary(retained_frames, 'retained'),
            'original_positive_video_retention': cluster_summary([r for r in base_values if r['legacy_positive_video']], 'positive_video_retained')})
        invisible, full_plan, isolated = [], [], []
        visibility_rows = defaultdict(list)
        for row in endpoints:
            key = evidence_id(row)
            review = reviews[key]
            a = scored[variant][originals[row['relative_path']]['sample_id']]
            b = scored[variant][row['sample_id']]
            states = b['objects_resolution']['frame_states']
            old_endpoint = scored['legacy'][row['sample_id']]
            item = {'family': row['family'], 'positive_video': b['score'] > 0,
                    'no_positive': b['coverage'] == 1 and b['score'] == 0,
                    'all_decisive_negative': all(s == 'absent' for s in states),
                    'positive_frame_fraction': b['score'], 'unconfirmed_frame_fraction': b['abstention'],
                    'positive_video_delta': int(b['score'] > 0)-int(old_endpoint['score'] > 0),
                    'positive_frame_fraction_delta': b['score']-old_endpoint['score']}
            known = bool(review.get('endpoint_invisible_verified'))
            full_plan.append({'family': row['family'], 'verified_no_positive': known and item['no_positive'],
                              'verified_all_decisive_negative': known and item['all_decisive_negative']})
            if known:
                invisible.append(item)
                if variant == 'adjacent_boxes' and (b['score'] > 0 or b['abstention'] > 0):
                    residuals.append({'relative_path': row['relative_path'], 'family': row['family'],
                        'legacy': scored['legacy'][row['sample_id']]['score'], 'repair': b['score'],
                        'abstention': b['abstention'], 'frame_states': states})
            old_base = scored['legacy'][originals[row['relative_path']]['sample_id']]
            if review.get('removal_verified') and old_base['score'] > 0:
                isolated.append({'family': row['family'], 'drop': a['score']-b['score'] >= .03,
                    'decisive_drop': a['score']-b['score'] >= .03 and a['abstention'] == b['abstention'] == 0,
                    'baseline_retained': a['score'] > 0})
            labels = review.get('labels')
            if labels:
                for side, value in [('before', a), ('after', b)]:
                    truths = [frame_truth(x, y) for x, y in zip(labels[side+'_target'], labels[side+'_other'])]
                    if len(truths) != 16:
                        raise ValueError('visibility frame slots changed')
                    for truth, state in zip(truths, value['objects_resolution']['frame_states']):
                        visibility_rows[side, truth].append({'family': row['family'], 'positive': state == 'supported',
                            'decisive_negative': state == 'absent', 'unconfirmed': state == 'unconfirmed', 'missing': state == 'missing'})
        endpoint_stats.append({'variant': variant, 'remaining_positive_videos': sum(r['positive_video'] for r in invisible),
            'no_positive_videos': sum(r['no_positive'] for r in invisible),
            'all_decisive_negative_videos': sum(r['all_decisive_negative'] for r in invisible),
            'invisible': summarize_metrics(invisible, ('positive_video', 'no_positive', 'all_decisive_negative', 'positive_frame_fraction', 'unconfirmed_frame_fraction', 'positive_video_delta', 'positive_frame_fraction_delta')),
            'full_plan': summarize_metrics(full_plan, ('verified_no_positive', 'verified_all_decisive_negative')),
            'same_legacy_positive_isolated_cohort': summarize_metrics(isolated, ('drop', 'decisive_drop', 'baseline_retained'))})
        visibility_stats.extend({'variant': variant, 'side': key[0], 'truth': key[1],
            **summarize_metrics(rows, ('positive', 'decisive_negative', 'unconfirmed', 'missing'))}
            for key, rows in sorted(visibility_rows.items()))
    controls = [{'scheme': scheme, 'level': level,
        **summarize_metrics(rows, ('original', 'target', 'background', 'target_minus_background'))}
        for (scheme, level), rows in sorted(matched_contrasts(after).items())]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(out/'residual-endpoints.jsonl', residuals)
    summary = {'version': 'objects-repair-v2', 'regression': regression, 'original_impact': impact,
        'endpoints': endpoint_stats, 'visibility': visibility_stats, 'matched_controls': controls,
        'planned_videos': len(caches), 'planned_conditions': len(matrix),
        'visibility_status': dict(Counter(r['status'] for r in reviews.values())),
        'summary': [r for r in report['summary'] if r['task'] == 'objects'],
        'input_sha256': {v: sha(v) for k, v in vars(args).items() if k != 'out'},
        'source_sha256': {n: sha(ROOT/n) for n in ['scripts/evaluate_objects_repair.py', 'scripts/score_matrix.py',
            'scripts/render_matrix_tables.py', 'scripts/analyze_object_controls.py',
            'src/vbench_prompts_compile/objects_repair.py', 'src/vbench_prompts_compile/experiments.py']},
        'protocol_sha256': sha(ROOT/'docs/plans/15-objects-temporal-repair.md'),
        'notes': ['Post-hoc engineering repair, not an independent blind evaluation.',
            'Same 0.5 IoU rule on originals and every control; no threshold sweep or new training/inference.',
            'Visibility model consensus is used for evaluation only, never by the scorer; not human gold.',
            'Unconfirmed predictions are abstentions scored zero, not verified absence; report both no-positive and decisive-negative endpoints.',
            'Original positive frame retention measures agreement with the old detector, not recall; visibility-stratified rates are separate.',
            'The same legacy-positive isolated-removal cohort is retained for comparing variants.',
            '2000 source-family bootstrap draws, seed 20260919; frames are not independent observations.',
            'Sustained detector hallucinations and brief/fast-moving true objects remain limitations.']}
    rows = [{'Variant': r['variant'], 'Videos': r['score']['n'], 'Mean score (CI)': ci_text(r['score']),
             'Delta from legacy (CI)': ci_text(r['delta_from_legacy']), 'Abstain (CI)': ci_text(r['abstention']),
             'Old positive frames retained (CI)': ci_text(r['original_positive_frame_retention']),
             'Old positive videos retained (CI)': ci_text(r['original_positive_video_retention'])} for r in impact]
    table(out, 'original-impact', rows, list(rows[0]))
    rows = [{'Variant': r['variant'], 'Invisible videos': r['invisible']['positive_video']['n'],
             'Still positive': r['remaining_positive_videos'], 'No positive': r['no_positive_videos'],
             'All decisive negative': r['all_decisive_negative_videos'],
             'Positive video delta (CI)': ci_text(r['invisible']['positive_video_delta']),
             'Positive frames (CI)': ci_text(r['invisible']['positive_frame_fraction']),
             'Unconfirmed frames (CI)': ci_text(r['invisible']['unconfirmed_frame_fraction']),
             'Same isolated cohort drop (CI)': ci_text(r['same_legacy_positive_isolated_cohort']['drop']),
             'Same isolated cohort decisive drop (CI)': ci_text(r['same_legacy_positive_isolated_cohort']['decisive_drop'])}
            for r in endpoint_stats]
    table(out, 'endpoints', rows, list(rows[0]))
    rows = [{'Variant': r['variant'], 'Side': r['side'], 'Model visibility': r['truth'],
             'Frames': r['positive']['n'], 'Families': r['positive']['families'],
             'Positive rate (CI)': ci_text(r['positive']), 'Decisive negative rate (CI)': ci_text(r['decisive_negative']),
             'Unconfirmed rate (CI)': ci_text(r['unconfirmed'])} for r in visibility_stats]
    table(out, 'visibility', rows, list(rows[0]))
    rows = [{'Scheme': r['scheme'], 'Level': r['level'], 'Complete videos': r['target']['n'],
             'Target (CI)': ci_text(r['target']), 'Background (CI)': ci_text(r['background']),
             'Target minus background (CI)': ci_text(r['target_minus_background'])} for r in controls]
    table(out, 'controls', rows, list(rows[0]))
    rows = [{'Scheme': r['scheme'], 'Transform': r['transform'], 'N': r['eligible'],
             'Original': r['original_score']['estimate'], 'Transformed': r['transformed_score']['estimate'],
             'Delta (CI)': ci_text(r['paired_delta']), 'Coverage': r['transformed_coverage']['estimate'],
             'Abstain': r['transformed_abstention']['estimate']} for r in summary['summary']]
    table(out, 'scores', rows, list(rows[0]))
    summary['output_sha256'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.suffix in {'.md', '.csv', '.tex', '.jsonl'}}
    J.atomic_json(out/'report.json', summary)
    print(json.dumps({'regression': regression, 'remaining_positive_endpoints':
        {r['variant']: r['remaining_positive_videos'] for r in endpoint_stats}, 'out': str(out)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
