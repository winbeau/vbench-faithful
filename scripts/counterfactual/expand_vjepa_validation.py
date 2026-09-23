"""Expand frozen VBench validation from 90 to all 450 existing TEST MP4s.

No new training, score mapping, model download, calibration media, or unseen-test
claim. Reuse the exact scoring implementation; new manifests never overwrite v1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from . import validate_vjepa_motion as frozen
from .static_jitter import ROOT, digest
from .vjepa_motion_probe import fresh_output, read_rows, write_json, write_rows


def load_config(path):
    extension = json.loads(Path(path).read_text())
    if extension['protocol'] != 'dynamic-vjepa-frozen-expansion450-v1':
        raise ValueError('unexpected expansion protocol')
    parent = ROOT / extension['parent_config']
    frozen.check_sha(parent, extension['parent_config_sha256'])
    frozen.check_sha(Path(frozen.__file__), extension['frozen_validation_script_sha256'])
    config = json.loads(parent.read_text())
    config['extension'] = extension
    config['construction']['selection_rule'] = extension['selection_rule']
    config['construction']['analysis_intent'] = extension['construction_intent']
    return config


def expand_sources(natural, previous, counts):
    natural = sorted(natural, key=lambda r: r['video_uid'])
    uids = {r['video_uid'] for r in natural}
    old = {r['video_uid'] for r in previous}
    if len(natural) != len(uids) or len(uids) != counts['sources']:
        raise ValueError('natural source coverage or duplicate UID')
    if len(previous) != len(old) or len(old) != counts['previous_cf_sources'] or not old <= uids:
        raise ValueError('previous source coverage')
    if any(r['split'] != 'test' or not r['relative_video_path'].endswith('.mp4') for r in natural):
        raise ValueError('only the existing official TEST MP4 cohort is allowed')
    prompts = Counter(r['prompt_id'] for r in natural)
    if len(prompts) != counts['prompts'] or set(prompts.values()) != {15}:
        raise ValueError('prompt coverage changed')
    additional = [r for r in natural if r['video_uid'] not in old]
    if len(additional) != counts['additional_cf_sources']:
        raise ValueError('additional source coverage')
    return natural, additional


def select(args, config):
    previous = Path(args.previous_root)
    ext = config['extension']
    for relative, key in [('inputs/inputs.jsonl', 'previous_inputs_sha256'),
                          ('selection/natural_sources.jsonl', 'previous_natural_sources_sha256'),
                          ('selection/counterfactual_sources.jsonl', 'previous_cf_sources_sha256')]:
        frozen.check_sha(previous / relative, ext[key])
    natural, additional = expand_sources(read_rows(previous / 'selection/natural_sources.jsonl'),
                                        read_rows(previous / 'selection/counterfactual_sources.jsonl'), ext['cohorts'])
    out = fresh_output(args.output)
    write_rows(out / 'all_sources.jsonl', natural)
    write_rows(out / 'additional_sources.jsonl', additional)
    write_json(out / 'construction.json', config['construction'])
    write_json(out / 'selection.json', {
        'config_sha256': digest(Path(args.config)), 'driver_sha256': digest(Path(__file__)),
        'files': {p.name: digest(p) for p in sorted(out.iterdir())},
        'sources': len(natural), 'additional_sources': len(additional),
        'scores_used_for_selection': False, 'label_values_used_for_selection': False,
        'previous_test_exposure': True, 'new_independent_prompts': 0,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'selection.json').read_text(), flush=True)


def validate_combinations(rows, expected_uids):
    expected = {(uid, family, seed) for uid in expected_uids for family, seed in
                [('original', 0), ('encoding_control', 0), ('local_texture_alternating', 1701),
                 ('local_texture_alternating', 2904)]}
    actual = [(r['base_id'], r['family'], r['seed']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('missing, extra or duplicate original/control/two-seed input')


def prepare(args, config):
    selection = Path(args.selection)
    frozen.verify_selection(selection, args.config)
    ext = config['extension']
    previous = Path(args.previous_root).resolve()
    frozen.check_sha(previous / 'inputs/inputs.jsonl', ext['previous_inputs_sha256'])
    old_inputs = read_rows(previous / 'inputs/inputs.jsonl')
    originals = {r['base_id']: r for r in old_inputs if r['cohort'] == 'holdout' and r['family'] == 'original'}
    natural = read_rows(selection / 'all_sources.jsonl')
    additional = {r['video_uid'] for r in read_rows(selection / 'additional_sources.jsonl')}
    old = {r['video_uid'] for r in natural} - additional
    rows = []
    for r in old_inputs:
        if r['cohort'] == 'holdout' and r['counterfactual_primary']:
            video = Path(r['video'])
            rows.append(dict(r, video=str((previous / 'code' / video).resolve()) if not video.is_absolute() else str(video),
                             construction_group='initial90'))
    validate_combinations(rows, old)
    manifests = sorted(Path(args.construction).glob('shard-*/candidates.jsonl'))
    if len(manifests) != 4:
        raise ValueError('four complete construction shards required')
    new = []
    for manifest in manifests:
        completion = json.loads((manifest.parent / 'completion.json').read_text())
        provenance = json.loads((manifest.parent / 'construction.json').read_text())
        frozen.check_sha(manifest, completion['manifest_sha256'])
        frozen.check_sha(selection / 'construction.json', provenance['config_sha256'])
        if completion['status'] != 'finished' or completion['counts'].get('construction_failed', 0):
            raise ValueError('construction incomplete or failed; preserve all failures')
        new.extend(dict(r, video=str(Path(r['video']).resolve()), construction_group='additional360')
                   for r in read_rows(manifest))
    validate_combinations(new, additional)
    rows.extend(new)
    for r in rows:
        r.update(cohort='holdout', counterfactual_primary=True,
                 evaluation_id=f"{r['base_id']}:{r['family']}:{r['seed']}")
        frozen.check_sha(r['video'], r['sha256'])
        if not r['pixel_exact_to_intended'] or not r['native_timeline_preserved'] or r['decoded_shape'][0] != 16 or abs(r['fps'] - 8) > 1e-6:
            raise ValueError('native timeline or pixel identity failed')
        if r['family'] == 'original' and r['sha256'] != originals[r['base_id']]['sha256']:
            raise ValueError('official original changed since first validation')
    validate_combinations(rows, {r['video_uid'] for r in natural})
    out = fresh_output(args.output)
    rows.sort(key=lambda r: r['evaluation_id'])
    write_rows(out / 'inputs.jsonl', rows)
    write_json(out / 'completion.json', {
        'status': 'finished', 'inputs': len(rows), 'originals': len(natural),
        'manifest_sha256': digest(out / 'inputs.jsonl'), 'config_sha256': digest(Path(args.config)),
        'driver_sha256': digest(Path(__file__)), 'selection_sha256': digest(selection / 'selection.json'),
        'construction_manifests': {str(p): digest(p) for p in manifests},
        'previous_inputs_sha256': ext['previous_inputs_sha256'],
        'reused_constructed_mp4': 270, 'new_constructed_mp4': 1080,
        'construction_status_counts': dict(Counter(r['status'] for r in rows)),
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'completion.json').read_text(), flush=True)


def score(args, config):
    receipt = json.loads((Path(args.inputs).parent / 'completion.json').read_text())
    if receipt['inputs'] != config['extension']['cohorts']['inputs_per_backend']:
        raise ValueError('unexpected expansion input count')
    frozen.check_sha(Path(__file__), receipt['driver_sha256'])
    frozen.score(args, config)
    write_json(Path(args.output) / 'expansion_driver.json', {
        'driver_sha256': digest(Path(__file__)), 'parent_config_sha256': config['extension']['parent_config_sha256'],
        'frozen_scoring_script_sha256': digest(Path(frozen.__file__)), 'training_updates': 0})


def load_predictions(inputs, score_root, config_path):
    ledger = {r['evaluation_id']: r for r in inputs}
    output, receipts = {}, {}
    for backend in ('origin', 'vjepa'):
        merged = {}; receipts[backend] = []
        shards = sorted((Path(score_root) / backend).glob('shard-*'))
        if len(shards) != 4:
            raise ValueError('four completed shards required per backend')
        for shard in shards:
            c = json.loads((shard / 'completion.json').read_text())
            p = json.loads((shard / 'provenance.json').read_text())
            if c['status'] != 'finished' or c['failed'] or c['completed'] != c['expected']:
                raise ValueError('incomplete scoring')
            frozen.check_sha(shard / 'scores.jsonl', c['scores_sha256'])
            frozen.check_sha(config_path, p['config_sha256'])
            records = read_rows(shard / 'scores.jsonl')
            if len(records) != c['completed']:
                raise ValueError('scoring count mismatch')
            for r in records:
                key = r['evaluation_id']
                if key in merged or key not in ledger or r['status'] != 'ok' or r['input_sha256'] != ledger[key]['sha256']:
                    raise ValueError('duplicate/missing/changed prediction')
                for name in ('base_id', 'cohort', 'family', 'seed', 'prompt_id', 'generator'):
                    if r[name] != ledger[key][name]:
                        raise ValueError('prediction metadata mismatch')
                merged[key] = r
            receipts[backend].append({'shard': str(shard), 'completion': c,
                                      'provenance_sha256': digest(shard / 'provenance.json')})
        if set(merged) != set(ledger):
            raise ValueError('incomplete predictions')
        output[backend] = merged
    return output, receipts


def summarize(args, config):
    inputs = read_rows(args.inputs)
    if len(inputs) != 1800:
        raise ValueError('all 1800 inputs required')
    validate_combinations(inputs, {r['base_id'] for r in inputs})
    results, receipts = load_predictions(inputs, args.scores, args.config)
    for backend in ('origin', 'vjepa'):
        for shard in receipts[backend]:
            provenance = json.loads((Path(shard['shard']) / 'provenance.json').read_text())
            frozen.check_sha(args.inputs, provenance['input_sha256'])
    def value(row):
        key = row['evaluation_id']; v = results['vjepa'][key]
        return {'origin': results['origin'][key]['score'], 'joint': v['joint']['score'],
                'natural_only': v['natural_only']['score'], 'joint_latent': v['joint']['latent']}
    indexed = {(r['base_id'], r['family'], r['seed']): r for r in inputs}
    bases = sorted({r['base_id'] for r in inputs})
    pairs = []
    for uid in bases:
        base, control = (indexed[uid, f, 0] for f in ('original', 'encoding_control'))
        cf = [indexed[uid, 'local_texture_alternating', s] for s in (1701, 2904)]
        b, c = value(base), value(control)
        if b != c or results['vjepa'][base['evaluation_id']]['feature_sha256'] != results['vjepa'][control['evaluation_id']]['feature_sha256']:
            raise ValueError('encoding control differs')
        pairs.append({'base_id': uid, 'prompt_id': base['prompt_id'], 'generator': base['generator'],
                      'construction_group': base['construction_group'], 'quality_flags': [r.get('reason') for r in cf],
                      **{key: [b[key], *[value(r)[key] for r in cf]] for key in b}})
    frozen.check_sha(args.human_pairs, config['human_pairs_sha256'])
    human = []
    with Path(args.human_pairs).open() as handle:
        for r in csv.DictReader(handle):
            if r['dimension'] != 'dynamics_degree' or r['split'] != 'test':
                continue
            if r['video_a_uid'] in bases and r['video_b_uid'] in bases:
                label = float(r['human_label'])
                if label not in (0, .5, 1):
                    raise ValueError('unexpected human label')
                human.append(dict(a=r['video_a_uid'], b=r['video_b_uid'], prompt_id=r['prompt_id'], label=label))
    if len(human) != 450:
        raise ValueError('all 450 human pairs required')
    human_results = {}; credit = {}
    ordered = [r for r in human if r['label'] != .5]
    for view, family, seed in [('base', 'original', 0), ('1701', 'local_texture_alternating', 1701),
                               ('2904', 'local_texture_alternating', 2904)]:
        lookup = {uid: value(indexed[uid, family, seed]) for uid in bases}
        human_results[view] = frozen.human_statistics(human, lookup, config)
        credit[view] = {}
        for backend in ('origin', 'joint', 'natural_only'):
            margins = np.array([(lookup[r['a']][backend] - lookup[r['b']][backend]) * (2 * r['label'] - 1) for r in ordered])
            credit[view][backend] = (margins > 0).astype(float) + .5 * (margins == 0)
    response = {}
    for seed in ('1701', '2904'):
        response[seed] = {}
        for backend in ('origin', 'joint', 'natural_only'):
            change = credit[seed][backend] - credit['base'][backend]
            a = config['analysis']
            response[seed][backend] = {'concordance_change': float(change.mean()),
                'prompt_ci95': frozen.cluster_interval(change, [r['prompt_id'] for r in ordered], a['bootstrap_seed'], a['bootstrap_replicates']),
                'better_pairs': int((change > 0).sum()), 'unchanged_pairs': int((change == 0).sum()), 'worse_pairs': int((change < 0).sum())}
    groups = {'all450': pairs, **{g: [r for r in pairs if r['construction_group'] == g] for g in ('initial90', 'additional360')}}
    if [len(groups[k]) for k in ('all450', 'initial90', 'additional360')] != [450, 90, 360]:
        raise ValueError('analysis subgroup coverage changed')
    out = fresh_output(args.output)
    write_rows(out / 'pairs.jsonl', pairs); write_rows(out / 'human_pairs.jsonl', human)
    write_json(out / 'summary.json', {
        'status': 'completed_exposed_test_coverage_expansion_not_absolute_strength_certification',
        'config_sha256': digest(Path(args.config)), 'input_sha256': digest(Path(args.inputs)), 'driver_sha256': digest(Path(__file__)),
        'coverage': {'sources': 450, 'prompts': 30, 'inputs_per_backend': 1800, 'counterfactual_videos': 900,
                     'encoding_controls_exact': 450, 'runtime_failures': 0, 'human_pairs_by_view': 450, 'new_independent_prompts': 0},
        'counterfactual': {name: frozen.paired_stats(rows, config) for name, rows in groups.items()},
        'human_preferences_by_view': human_results, 'human_response_change': response,
        'shards': receipts, 'absolute_strength_calibration': 'NOT RUN', 'motion_type_human_review': 'NOT RUN',
        'no_training_or_mapping_changes': True, 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'summary.json').read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select'); s.add_argument('--previous-root', required=True)
    s = sub.add_parser('prepare')
    for name in ('previous-root', 'selection', 'construction'):
        s.add_argument('--' + name, required=True)
    s = sub.add_parser('score')
    for name in ('inputs', 'probe-root', 'video-root', 'raft-weight', 'upstream'):
        s.add_argument('--' + name, required=True)
    s.add_argument('--backend', choices=['origin', 'vjepa'], required=True)
    s.add_argument('--shard', type=int, required=True); s.add_argument('--shards', type=int, default=4)
    s = sub.add_parser('summarize')
    for name in ('inputs', 'scores', 'human-pairs'):
        s.add_argument('--' + name, required=True)
    for s in sub.choices.values():
        s.add_argument('--output', required=True)
    args = p.parse_args()
    globals()[args.command](args, load_config(args.config))


if __name__ == '__main__':
    main()
