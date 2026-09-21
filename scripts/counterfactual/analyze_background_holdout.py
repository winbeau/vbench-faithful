"""Evaluate the locked background primary once, retaining every test failure."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from .analyze_background_development import checked_rows
from .analyze_background_native import summarize_native
from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json, write_jsonl
from .run_background_holdout import frozen_protocol


def evaluate_natural(pairs, videos, protocol):
    # evaluate_population uses its first method as the paired reference.
    # Protocol method names are sorted, so passing them directly would compare
    # against aggregation while labelling the difference "vs official".
    methods = ['official', *[m for m in protocol['methods'] if m != 'official']]
    return evaluate_population(pairs, videos, methods,
        {m: protocol['analysis']['primary_tie_margin'] for m in methods},
        resamples=protocol['analysis']['bootstrap_resamples'],
        seed=protocol['analysis']['bootstrap_seed'])


def verify_construction(index, verification, protocol, expected):
    if (len(index) != len(expected) or {r['video_uid'] for r in index} != set(expected)
            or verification['dataset_index_sha256'] != protocol['test_dataset_index_sha256']
            or verification['protocol_sha256'] != protocol['construction_protocol_sha256']):
        raise ValueError('pixel verification or construction population mismatch')
    results = verification['results']
    verified = {r['video_uid']: r for r in results}
    if len(results) != len(expected) or set(verified) != set(expected):
        raise ValueError('pixel verification coverage mismatch')
    allowed = {'accepted': 'accepted_replay_verified', 'construction_rejected': 'rejection_verified'}
    for entry in index:
        result = verified[entry['video_uid']]
        if result['status'] != allowed.get(entry['status']):
            raise ValueError('unverified construction candidate')
        if entry['status'] == 'accepted' and (
                result['subject_blur_background_changed_pixels'] != 0
                or result['background_switch_subject_changed_pixels'] != 0
                or result['variants'] != 8):
            raise ValueError('pixel intervention invariant failed')


def localization_summary(rows):
    groups = {}
    for row in rows:
        if row['status'] != 'completed':
            continue
        for name, variant in row.get('variants', {'natural': row}).items():
            diag = variant['localizer_diagnostics']
            fractions = variant['background_fraction']['frame']
            groups.setdefault(name, []).append((
                1-diag['num_empty_foreground_frames']/diag['num_frames'],
                sum(x < .05 for x in fractions), len(fractions)))
    return {name: {'scored_videos': len(values),
        'mean_foreground_frame_presence': float(np.mean([v[0] for v in values])),
        'videos_without_foreground': sum(v[0] == 0 for v in values),
        'frames_with_insufficient_background': sum(v[1] for v in values),
        'total_frames': sum(v[2] for v in values)} for name, values in groups.items()}


def conditional_full_blur(rows, cases, primary):
    """Descriptive localization strata, never used to select the test cohort."""
    groups = {'foreground_on_at_least_half_frames_in_both': [], 'remaining_scored_bases': []}
    lookup = {(r['base_id'], r['method']): r for r in cases if r['position'] == 'full'}
    for row in rows:
        if row['status'] != 'completed':
            continue
        variants = [row['variants'][k] for k in ('clean', 'full/subject_blur')]
        presence = [1-v['localizer_diagnostics']['num_empty_foreground_frames']/v['num_frames'] for v in variants]
        name = ('foreground_on_at_least_half_frames_in_both' if min(presence) >= .5
                else 'remaining_scored_bases')
        groups[name].append(row['video_uid'])
    return {name: {'n_bases': len(uids), 'descriptive_only': True,
        'mean_absolute_change': {method: float(np.mean([
            lookup[uid, method]['foreground_absolute_change'] for uid in uids])) if uids else None
            for method in ('official', primary)}} for name, uids in groups.items()}


def load_scores(path, mode, protocol, protocol_hash, expected):
    rows, runs = [], []
    for directory in sorted(path.glob('shard*')):
        if not directory.is_dir():
            continue
        chunk, run = checked_rows(directory)
        if (run['protocol_sha256'] != protocol_hash or run['mode'] != mode
                or run['source_sha256'] != protocol['source_sha256']):
            raise ValueError('incomplete, mixed or changed heldout method')
        if mode == 'interventions' and run['dataset_index_sha256'] != protocol['test_dataset_index_sha256']:
            raise ValueError('heldout construction was changed')
        rows.extend(chunk); runs.append(run)
    if len(rows) != len(expected) or {r['video_uid'] for r in rows} != set(expected):
        raise ValueError('missing, duplicate or unexpected heldout identities')
    for row in rows:
        base = row if mode == 'natural' else row.get('base', {})
        if any(base.get(k) != v for k, v in expected[row['video_uid']].items()):
            raise ValueError('heldout input identity mismatch')
        if row['status'] == 'completed':
            variants = row.get('variants', {'natural': row})
            if mode == 'interventions' and set(variants) != {'clean', 'full/subject_blur'} | {p+'/'+k for p in ('start', 'middle', 'end') for k in ('subject_blur', 'background_switch')}:
                raise ValueError('completed intervention has missing variants')
            if any(set(v['scores']) != set(protocol['methods']) for v in variants.values()):
                raise ValueError('completed result has missing methods')
            if any(s['status'] != 'succeeded' or not np.isfinite(s['score'])
                   for v in variants.values() for s in v['scores'].values()):
                raise ValueError('invalid score marked completed')
    return rows, runs


def assess_gates(natural, native, protocol, *, max_parity_error, clean_replay_error):
    primary, spec = protocol['primary_method'], protocol['gates']
    def lower(value):
        return value[0] if value is not None else -float('inf')
    gates = {
        'natural_noninferiority': lower(natural['methods'][primary]['paired_delta_ci95']) >= spec['natural_accuracy_delta_ci95_lower_min'],
        'natural_coverage': natural['methods'][primary]['coverage'] >= spec['natural_pair_coverage_min'],
        'origin_parity': max_parity_error is not None and max_parity_error <= spec['origin_parity_max_error'],
        'clean_replay_parity': clean_replay_error is not None and clean_replay_error <= spec['origin_parity_max_error'],
    }
    accepted = native['construction_status_counts'].get('accepted', 0)
    gates['counterfactual_scoring_coverage'] = bool(accepted and native['status_counts'].get('completed', 0)/accepted >= spec['counterfactual_scoring_coverage_min'])
    for position in spec['foreground_positions_for_absolute_delta_gate']:
        stats = native['by_position'][position][primary]['foreground_absolute_change']
        gates['foreground_absolute_'+position] = stats['mean'] is not None and stats['mean'] <= spec['foreground_mean_absolute_delta_max'] and stats['n_bases'] == accepted
    robust = native['by_position'][spec['paired_robustness_improvement_position']][primary]['paired_improvement_vs_official']
    gates['paired_robustness_improvement'] = lower(robust['mean_ci95']) > spec['paired_robustness_improvement_ci95_lower_gt']
    for position in spec['background_positions']:
        stats = native['by_position'][position][primary]
        gates['background_positive_'+position] = lower(stats['background_temporal_drop']['mean_ci95']) > spec['background_temporal_drop_ci95_lower_gt']
        value = stats['background_drop_gt_foreground_abs']['mean']
        gates['background_distinction_'+position] = value is not None and value >= spec['background_drop_gt_subject_abs_change_fraction_min']
    scale = native['scale_adjusted_response'][primary]
    value = scale['background_response_retention']
    gates['background_response_retention'] = value is not None and value >= spec['background_response_retention_mean_min']
    gates['scale_adjusted_selectivity'] = lower(scale['paired_ratio_improvement_ci95']) > spec['scale_adjusted_paired_improvement_ci95_lower_gt']
    return gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--natural-run', type=Path, required=True)
    parser.add_argument('--intervention-run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--freeze-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = frozen_protocol(args.protocol)
    protocol_hash = sha256_file(args.protocol)
    expected = {r['video_uid']: r for r in read_jsonl(ROOT/protocol['manifest']) if r['split'] == 'test'}
    if len(expected) != protocol['natural_video_count']:
        raise ValueError('heldout natural denominator changed')
    natural, nruns = load_scores(args.natural_run, 'natural', protocol, protocol_hash, expected)
    native, iruns = load_scores(args.intervention_run, 'interventions', protocol, protocol_hash, expected)
    receipt = json.loads(args.freeze_receipt.read_text())
    if (receipt['protocol_sha256'] != protocol_hash
            or receipt['test_dataset_index_sha256'] != protocol['test_dataset_index_sha256']
            or receipt['pixel_verification_sha256'] != sha256_file(args.verification)
            or datetime.fromisoformat(receipt['witnessed_utc_before_launch']).timestamp()
                >= min(r['started_unix'] for r in nruns+iruns)):
        raise ValueError('freeze receipt does not precede these test runs')
    if sha256_file(args.dataset/'index.jsonl') != protocol['test_dataset_index_sha256']:
        raise ValueError('frozen intervention index changed')
    verification = json.loads(args.verification.read_text())
    index = read_jsonl(args.dataset/'index.jsonl')
    verify_construction(index, verification, protocol, expected)
    manifests = {}
    for entry in index:
        path = artifact_path(args.dataset, entry['manifest'])
        if sha256_file(path) != entry['manifest_sha256']:
            raise ValueError('test construction manifest changed')
        manifest = json.loads(path.read_text())
        if (manifest['base'] != expected[entry['video_uid']]
                or manifest['status'] != entry['status']):
            raise ValueError('test construction source identity mismatch')
        manifests[entry['video_uid']] = manifest
    for row in native:
        if row['construction_status'] != manifests[row['video_uid']]['status']:
            raise ValueError('test construction status changed')
    if sum(m['status'] == 'accepted' for m in manifests.values()) != protocol['test_construction_accepted']:
        raise ValueError('test construction acceptance denominator changed')
    labels = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(labels) != protocol['pair_labels_sha256']:
        raise ValueError('background preference labels changed')
    pairs = [r for r in csv.DictReader(labels.open()) if r['dimension'] == 'background_consistency' and r['split'] == 'test']
    if len(pairs) != protocol['natural_pair_count']:
        raise ValueError('test preference denominator changed')
    for pair in pairs:
        if any(expected[pair[k]]['prompt_id'] != pair['prompt_id'] for k in ('video_a_uid', 'video_b_uid')):
            raise ValueError('preference pair prompt mismatch')
    videos = {r['video_uid']: r for r in natural}
    errors = []
    for row in native:
        if row['construction_status'] != 'accepted' or row['status'] != 'completed':
            continue
        source = videos[row['video_uid']]
        if source['status'] != 'completed':
            continue
        clean = row['variants']['clean']
        if clean['frame_array_sha256'] != source['frame_array_sha256']:
            raise ValueError('clean intervention PNGs differ from natural decoding')
        errors.extend(abs(clean['scores'][m]['score']-source['scores'][m]['score']) for m in protocol['methods'])
    # Do not count incomplete rows as successes just because a partial score
    # was computed before a later failure. The raw rows remain archived.
    analysis_natural, analysis_native = copy.deepcopy(natural), copy.deepcopy(native)
    for row in analysis_natural:
        if row['status'] != 'completed':
            row['scores'] = {}
    for row in analysis_native:
        row['rejection_reasons'] = manifests[row['video_uid']].get('rejection_reasons', [])
        if row['status'] != 'completed':
            for variant in row.get('variants', {}).values():
                variant['scores'] = {}
    methods = protocol['methods']
    nstats = evaluate_natural(pairs, {r['video_uid']: r for r in analysis_natural}, protocol)
    nstats.update(video_count=len(natural), status_counts=dict(Counter(r['status'] for r in natural)),
                  media_duration_seconds=sum(r.get('media_duration_seconds', 0) for r in natural))
    istats, cases = summarize_native(analysis_native, manifests, methods=methods,
                                    resamples=protocol['analysis']['bootstrap_resamples'])
    istats['uncertainty']['limitation'] = (
        'Held-out prompt groups; scene switches resample connected source/donor '
        'pairs. Precision depends on the number of accepted prompt pairs.')
    parity = [r['parity_absolute_error'] for r in natural if r.get('status') == 'completed']
    parity_error = max(parity) if len(parity) == len(natural) else None
    replay_error = max(errors) if len(errors) == protocol['test_construction_accepted']*len(methods) else None
    gates = assess_gates(nstats, istats, protocol, max_parity_error=parity_error, clean_replay_error=replay_error)
    report = {'protocol_sha256': protocol_hash, 'primary_method': protocol['primary_method'],
        'analysis_source_sha256': sha256_file(Path(__file__)), 'natural': nstats, 'interventions': istats,
        'freeze_receipt_sha256': sha256_file(args.freeze_receipt),
        'localization': {'natural': localization_summary(natural),
                         'interventions': localization_summary(native)},
        'conditional_full_blur': conditional_full_blur(native, cases, protocol['primary_method']),
        'origin_parity_max_error': parity_error, 'clean_replay_max_error': replay_error,
        'primary_gates': gates, 'all_primary_gates_pass': all(gates.values()),
        'source_runs': {'natural': nruns, 'interventions': iruns},
        'timing': {name: {'started_unix': min(r['started_unix'] for r in runs),
                         'finished_unix': max(r['finished_unix'] for r in runs),
                         'wall_seconds': max(r['finished_unix'] for r in runs)-min(r['started_unix'] for r in runs)}
                   for name, runs in (('natural', nruns), ('interventions', iruns))},
        'zero_primary_scores': {name: sum(v.get('scores', {}).get(protocol['primary_method'], {}).get('score') == 0
            for row in rows for v in row.get('variants', {'natural': row}).values())
            for name, rows in (('natural', natural), ('interventions', native))}}
    output = new_output(args.output)
    analysis_files = [Path(__file__), *[ROOT/'scripts/counterfactual'/name for name in (
        'analyze_background_native.py', 'analyze_background_development.py',
        'analyze_subject_background_auto.py', 'analyze_subject_natural.py')]]
    report['analysis_sources_sha256'] = {}
    for path in analysis_files:
        relative = path.resolve().relative_to(ROOT)
        target = output/'analysis_source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        report['analysis_sources_sha256'][str(relative)] = sha256_file(path)
    write_json(output/'statistics.json', report)
    write_jsonl(output/'natural_scores.jsonl', natural)
    write_jsonl(output/'intervention_scores.jsonl', native)
    with (output/'intervention_per_case.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader(); writer.writerows(cases)
    print(json.dumps({'output': str(output), 'all_primary_gates_pass': all(gates.values()), 'primary_gates': gates}))


if __name__ == '__main__':
    main()
