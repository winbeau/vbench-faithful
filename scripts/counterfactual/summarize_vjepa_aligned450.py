"""Summarize the frozen aligned-head TEST450 run without selecting a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from .score_vjepa_aligned450 import previous_data
from .static_jitter import digest
from .validate_vjepa_motion import check_sha, cluster_interval
from .vjepa_motion_probe import fresh_output, read_rows, write_json, write_rows


ARMS = ('origin', 'old_joint', 'aligned')
VIEWS = (('base', 'original', 0), ('1701', 'local_texture_alternating', 1701),
         ('2904', 'local_texture_alternating', 2904))


def interval(values, prompts, config):
    a = config['analysis']
    return cluster_interval(values, prompts, a['bootstrap_seed'], a['bootstrap_replicates'])


def paired_statistics(pairs, config):
    prompts = [r['prompt_id'] for r in pairs]
    out = {}; deltas = {}
    for arm in ARMS + ('aligned_latent',):
        values = np.array([r[arm] for r in pairs], float)
        diff = values[:, 1:] - values[:, :1]; delta = diff.mean(1); deltas[arm] = delta
        out[arm] = {'sources': len(pairs), 'base_mean': float(values[:, 0].mean()),
            'cf_mean': float(values[:, 1:].mean()), 'delta': float(delta.mean()), 'mae': float(abs(diff).mean()),
            'delta_prompt_ci95': interval(delta, prompts, config),
            'mae_prompt_ci95': interval(abs(diff).mean(1), prompts, config),
            'base_std': float(values[:, 0].std()), 'base_range': [float(values[:, 0].min()), float(values[:, 0].max())],
            'base_quantiles_05_25_50_75_95': np.quantile(values[:, 0], [.05, .25, .5, .75, .95]).tolist(),
            'largest_drop': float(diff.min()), 'largest_increase': float(diff.max()),
            'maximum_absolute_change': float(abs(diff).max()),
            'cf_by_seed': dict(zip(('1701', '2904'), values[:, 1:].mean(0).tolist()))}
    od = out['origin']['delta']; nd = out['aligned']['delta']
    out['relative_scale_invariance_check'] = {'allowed_delta': .1 * od if od > 0 else None,
        'origin_increased': od > 0, 'new_delta_over_origin': nd / od if od > 0 else None,
        'point_estimate_pass': bool(nd <= .1 * od) if od > 0 else None,
        'margin_prompt_ci95': interval(deltas['aligned'] - .1 * deltas['origin'], prompts, config),
        'absolute_physical_units_aligned': False, 'overall_goal_complete': False}
    return out


def preference_statistics(pairs, lookup, config):
    ordered = [r for r in pairs if r['label'] != .5]
    ties = [r for r in pairs if r['label'] == .5]
    if not ordered or not ties:
        raise ValueError('expected both ordered preferences and human ties')
    prompts = [r['prompt_id'] for r in ordered]
    out = {}; credits = {}
    for arm in ARMS:
        margin = np.array([(lookup[r['a']][arm] - lookup[r['b']][arm]) * (2 * r['label'] - 1) for r in ordered])
        credit = (margin > 0).astype(float) + .5 * (margin == 0); credits[arm] = credit
        gaps = [abs(lookup[r['a']][arm] - lookup[r['b']][arm]) for r in ties]
        out[arm] = {'ordered': len(ordered), 'strict_correct': int((margin > 0).sum()),
            'predicted_ties': int((margin == 0).sum()), 'incorrect': int((margin < 0).sum()),
            'concordance': float(credit.mean()), 'prompt_ci95': interval(credit, prompts, config),
            'human_ties': len(ties), 'human_tie_mean_absolute_gap': float(np.mean(gaps))}
    for reference in ('origin', 'old_joint'):
        change = credits['aligned'] - credits[reference]
        out['aligned_minus_' + reference] = {'difference': float(change.mean()),
            'prompt_ci95': interval(change, prompts, config), 'better_pairs': int((change > 0).sum()),
            'unchanged_pairs': int((change == 0).sum()), 'worse_pairs': int((change < 0).sum())}
    return out, credits


def collect(root, inputs, config, config_path):
    ledger = {r['evaluation_id']: r for r in inputs}; records = {}; receipts = []
    directories = sorted((Path(root) / 'scores').glob('shard-*'))
    if len(directories) != 4:
        raise ValueError('all four shards must finish before analysis')
    for i, path in enumerate(directories):
        c = json.loads((path / 'completion.json').read_text()); p = json.loads((path / 'provenance.json').read_text())
        if c['status'] != 'finished' or c['failed'] or c['expected'] != c['completed']:
            raise ValueError('incomplete run; preserve failures, do not omit')
        check_sha(path / 'scores.jsonl', c['scores_sha256']); check_sha(config_path, p['config_sha256'])
        if (p['head_sha256'] != config['head_sha256'] or p['old_head_sha256'] != config['old_head_sha256']
                or p['input_sha256'] != config['inputs_sha256'] or p['training_updates'] != 0 or p['score_mapping_changed']):
            raise ValueError('model/input/protocol identity changed')
        for name, checksum in p['code'].items():
            check_sha(Path(root) / 'code' / name, checksum)
        part = read_rows(path / 'scores.jsonl')
        expected_bases = set(sorted({r['base_id'] for r in inputs})[i::4])
        if (len(part) != c['completed'] or {r['evaluation_id'] for r in part} != set(p['expected_ids'])
                or {r['base_id'] for r in part} != expected_bases):
            raise ValueError('shard coverage mismatch')
        for r in part:
            key = r['evaluation_id']
            if key not in ledger or key in records or r['status'] != 'ok':
                raise ValueError('duplicate, extra or failed input')
            original = ledger[key]
            if r['input_sha256'] != original['sha256'] or r['pixels_sha256'] != original['decoded_pixels_sha256']:
                raise ValueError('input pixel/hash mismatch')
            for field in ('base_id', 'family', 'seed', 'prompt_id', 'generator', 'construction_group'):
                if r[field] != original[field]:
                    raise ValueError('input metadata changed')
            records[key] = r
        receipts.append({'shard': i, 'completion': c, 'gpu': p['gpu'], 'started_utc': p['started_utc'],
                         'provenance_sha256': digest(path / 'provenance.json')})
    if set(records) != set(ledger):
        raise ValueError('full 1800 predictions required')
    return records, receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'root', 'previous-root', 'human-pairs', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    inputs, old = previous_data(args.previous_root, config)
    scored, receipts = collect(args.root, inputs, config, args.config)
    index = {(r['base_id'], r['family'], r['seed']): r for r in inputs}
    bases = sorted({r['base_id'] for r in inputs})
    def values(row):
        key = row['evaluation_id']; now = scored[key]; before = old['vjepa'][key]
        if (now['feature_sha256'] != before['feature_sha256']
                or abs(now['old_joint']['latent'] - before['joint']['latent']) > config['inference']['head_tolerance']):
            raise ValueError('new run disagrees with old feature/head identity')
        for arm in ('aligned', 'old_joint'):
            if not np.isfinite(now[arm]['latent']) or abs(1 / (1 + np.exp(-now[arm]['latent'])) - now[arm]['score']) > 1e-12:
                raise ValueError('invalid head score')
        return dict(origin=old['origin'][key]['score'], old_joint=before['joint']['score'],
                    aligned=now['aligned']['score'], aligned_latent=now['aligned']['latent'])
    paired = []
    for uid in bases:
        base = index[uid, 'original', 0]; control = index[uid, 'encoding_control', 0]
        b, c = values(base), values(control)
        if b != c or scored[base['evaluation_id']]['feature_sha256'] != scored[control['evaluation_id']]['feature_sha256']:
            raise ValueError('encoding control mismatch')
        cf = [values(index[uid, 'local_texture_alternating', seed]) for seed in (1701, 2904)]
        paired.append(dict(base_id=uid, prompt_id=base['prompt_id'], generator=base['generator'],
            construction_group=base['construction_group'],
            **{k: [b[k], cf[0][k], cf[1][k]] for k in b}))
    check_sha(args.human_pairs, config['human_pairs_sha256'])
    human = []
    with Path(args.human_pairs).open() as handle:
        for r in csv.DictReader(handle):
            if r['dimension'] != 'dynamics_degree' or r['split'] != 'test':
                continue
            if r['video_a_uid'] in bases and r['video_b_uid'] in bases:
                human.append(dict(a=r['video_a_uid'], b=r['video_b_uid'], prompt_id=r['prompt_id'], label=float(r['human_label'])))
    if len(human) != 450 or Counter(r['label'] for r in human)[.5] != 279:
        raise ValueError('human preference coverage changed')
    human_results = {}; credits = {}
    for view, family, seed in VIEWS:
        lookup = {uid: values(index[uid, family, seed]) for uid in bases}
        human_results[view], credits[view] = preference_statistics(human, lookup, config)
    prompts = [r['prompt_id'] for r in human if r['label'] != .5]
    response = {}
    for seed in ('1701', '2904'):
        response[seed] = {}
        for arm in ARMS:
            change = credits[seed][arm] - credits['base'][arm]
            response[seed][arm] = {'difference': float(change.mean()), 'prompt_ci95': interval(change, prompts, config),
                'better_pairs': int((change > 0).sum()), 'unchanged_pairs': int((change == 0).sum()), 'worse_pairs': int((change < 0).sum())}
    groups = {'all450': paired}
    for group in ('initial90', 'additional360'):
        groups[group] = [r for r in paired if r['construction_group'] == group]
    for generator in sorted({r['generator'] for r in paired}):
        groups['generator:' + generator] = [r for r in paired if r['generator'] == generator]
    out = fresh_output(args.output)
    write_rows(out / 'pairs.jsonl', paired); write_rows(out / 'human_pairs.jsonl', human)
    summary = {'status': 'completed_frozen_aligned450_validation', 'config_sha256': digest(Path(args.config)),
        'head_sha256': config['head_sha256'], 'inputs_sha256': config['inputs_sha256'], 'summarizer_sha256': digest(Path(__file__)),
        'coverage': dict(config['coverage'], new_head_scored=len(scored), runtime_failures=0,
            encoding_controls_exact=450, all_new_features_match_previous=True,
            construction_status_counts=dict(Counter(r['status'] for r in inputs))),
        'counterfactual': {name: paired_statistics(group, config) for name, group in groups.items()},
        'human_preferences_by_view': human_results, 'human_response_change': response, 'shards': receipts,
        'training_updates': 0, 'mapping_changes': False, 'origin_recomputed': False,
        'previous_development_acceptance': 'natural_ordering_17_of_23_required_19_scale_gap_0.142674_user_accepts_0.6_range',
        'test_prompts_already_exposed': True, 'new_independent_holdout': False, 'calibration45_opened': False,
        'absolute_strength_calibration': 'NOT RUN', 'motion_type_human_review': 'NOT RUN',
        'formal_goal_complete': False, 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(out / 'summary.json', summary)
    print(json.dumps({'coverage': summary['coverage'], 'all450': summary['counterfactual']['all450'],
                      'human': human_results}), flush=True)


if __name__ == '__main__':
    main()
