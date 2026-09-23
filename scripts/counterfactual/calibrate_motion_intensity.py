"""Audit/fit the separate natural scale, then apply it to DEV32 without refit.

Two explicit stages: `fit` cannot read DEV32 scores; `apply` cannot fit a scale.
Both write fresh outputs and preserve raw intensity and all original scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.intensity_scale import calibrated_intensity, fit_scale
from .select_intensity_calibration import validate_calibration
from .static_jitter import digest
from .summarize_mp4_dev32 import audit_evidence


def bootstrap_scales(cohort, replicates=20000):
    prompts = sorted({r['prompt_id'] for r in cohort})
    grouped = [[r for r in cohort if r['prompt_id'] == p] for p in prompts]
    if len(prompts) != 21 or any(len(g) != 3 for g in grouped):
        raise ValueError('21 balanced calibration prompt clusters required')
    x = np.array([[r['intensity'] for r in group] for group in grouped])
    y = np.array([[r['origin'] for r in group] for group in grouped])
    index = np.random.default_rng(20260923).integers(0, 21, (replicates, 21))
    values = x[index].reshape(replicates, -1)
    target = y[index].mean(axis=(1, 2))
    valid = (target > 0) & (target < (values > 0).mean(axis=1))
    if not valid.all():
        # Endpoint fits have no finite positive scale; never silently drop them.
        return {'status': 'NOT ESTIMABLE', 'replicates': replicates,
                'degenerate_resamples': int((~valid).sum()), 'ci95': None}
    low, high = np.zeros(replicates), values.mean(axis=1)/target
    for _ in range(80):
        mid = (low+high)/2
        actual = (values/(values+mid[:, None])).mean(axis=1)
        above = actual > target
        low = np.where(above, mid, low)
        high = np.where(above, high, mid)
    return {'status': 'estimated', 'unit': 'short_side_lengths_per_second',
            'replicates': replicates, 'seed': 20260923, 'clusters': 21,
            'ci95': np.quantile((low+high)/2, [.025, .975]).tolist()}


def fit_audited(args):
    run, execution, manifest, sources_path, config_path = map(Path,
        (args.run_root, args.execution_root, args.manifest, args.sources, args.config))
    config = json.loads(config_path.read_text())
    if (config['protocol'] != 'natural63-cotracker3-intensity-calibration-v1'
            or config['sources_sha256'] != digest(sources_path)):
        raise ValueError('frozen natural calibration protocol differs')
    sources = list(map(json.loads, sources_path.read_text().splitlines()))
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    validate_calibration(rows, sources)
    for name, sha in config['excluded_sources_sha256'].items():
        path = execution / name
        if digest(path) != sha:
            raise ValueError('excluded cohort identity changed')
        excluded = list(map(json.loads, path.read_text().splitlines()))
        if ({r['prompt_id'] for r in excluded} & {r['prompt_id'] for r in sources}
                or {r['video_uid'] for r in excluded} & {r['video_uid'] for r in sources}):
            raise ValueError('calibration is not prompt/UID disjoint')
    ledger = {r['candidate_id']: r for r in rows}
    records, identities, runtimes, evidence_hashes = {}, [], [], {}
    for shard in range(4):
        directory = run / f'shard-{shard}'
        identity = json.loads((directory/'provenance.json').read_text())
        runtime = json.loads((directory/'runtime.json').read_text())
        expected_uids = {r['video_uid'] for r in sources[shard::4]}
        expected_ids = {r['candidate_id'] for r in rows if r['base_id'] in expected_uids}
        if (identity['sources_sha256'] != digest(sources_path) or identity['manifest_sha256'] != digest(manifest)
                or identity['config_sha256'] != digest(config_path) or not identity['calibration_natural_only']
                or set(identity['expected_ids']) != expected_ids or runtime['status'] != 'finished'
                or runtime['failed'] != 0 or runtime['completed'] != len(expected_ids)):
            raise ValueError('incomplete/failed or identity-mismatched calibration shard')
        for name, sha in identity['code_files'].items():
            if digest(execution/name) != sha:
                raise ValueError('execution snapshot differs: '+name)
        shard_records = list(map(json.loads, (directory/'scores.jsonl').read_text().splitlines()))
        if {r['candidate_id'] for r in shard_records} != expected_ids:
            raise ValueError('calibration shard coverage differs')
        if not shard_records[0]['origin'].get('reference_infer_parity'):
            raise ValueError('missing real original-infer parity')
        for record in shard_records:
            key = record['candidate_id']
            if (key in records or record['status'] != 'finished' or record['input_sha256'] != ledger[key]['sha256']
                    or record['repair']['score_kind'] != 'continuous_intensity'
                    or record['base_id'] != ledger[key]['base_id']
                    or record['prompt_id'] != ledger[key]['prompt_id'] or record['family'] != 'original'):
                raise ValueError('changed, failed or duplicate calibration record')
            cache = directory/'evidence'/(key+'.npz')
            if digest(cache) != identity['evidence_sha256'][key]:
                raise ValueError('calibration model evidence differs')
            audit_evidence(cache, ledger[key], record)
            records[key] = record
            evidence_hashes[key] = digest(cache)
        identities.append({'shard': shard, 'provenance_sha256': digest(directory/'provenance.json'),
                           'scores_sha256': digest(directory/'scores.jsonl')})
        runtimes.append(runtime)
    if set(records) != set(ledger):
        raise ValueError('all 63 natural calibration inputs must be retained')
    cohort = [{'candidate_id': key, 'video_uid': row['base_id'], 'prompt_id': row['prompt_id'],
               'generator': row['generator'], 'origin': row['origin']['score'],
               'intensity': row['repair']['score'], 'raw_ablation_intensity': row['repair']['raw_ablation']['score']}
              for key, row in sorted(records.items())]
    fitted = fit_scale([r['intensity'] for r in cohort], [r['origin'] for r in cohort])
    calibration = {'status': 'fitted_natural_scale_not_validated_repair', **fitted,
        'fit_inputs': '63 natural DEV MP4s only; current DEV32 bases and CFs excluded by prompt and UID',
        'cohort': cohort, 'scale_bootstrap': bootstrap_scales(cohort),
        'manifest_sha256': digest(manifest), 'sources_sha256': digest(sources_path),
        'config_sha256': digest(config_path), 'config': config,
        'execution_root': str(execution), 'shards': identities, 'evidence_sha256': evidence_hashes,
        'script_sha256': digest(Path(__file__)),
        'scale_implementation_sha256': digest(Path(__file__).resolve().parents[2]/'metrics/dynamic-degree/src/dynamic_degree/intensity_scale.py'),
        'python': platform.python_version(), 'numpy': np.__version__, 'runtime': runtimes,
        'gpu_worker_window_seconds': (max(datetime.fromisoformat(r['finished_utc']) for r in runtimes)
                                    -min(datetime.fromisoformat(r['started_utc']) for r in runtimes)).total_seconds(),
        'new_cf_created': 0, 'physical_correspondence': 'NOT VERIFIED',
        'human_motion_noninferiority': 'NOT RUN', 'holdout_media_opened': False,
        'fitted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False))
    print(json.dumps({k: calibration[k] for k in ('status', 'scale', 'origin_mean', 'intensity_mean',
                                                  'calibrated_mean', 'scale_bootstrap', 'gpu_worker_window_seconds')}, indent=2))


def align_pairs(pairs, calibration):
    fit_ids = {r['video_uid'] for r in calibration['cohort']}
    fit_prompts = {r['prompt_id'] for r in calibration['cohort']}
    if fit_ids & {r['base_id'] for r in pairs} or fit_prompts & {r['prompt_id'] for r in pairs}:
        raise ValueError('fit/evaluation leakage by UID or prompt')
    if (len(pairs) != 32 or len({r['base_id'] for r in pairs}) != 32
            or len({r['prompt_id'] for r in pairs}) != 8
            or set(Counter(r['prompt_id'] for r in pairs).values()) != {4}):
        raise ValueError('complete DEV32 design required')
    scale = calibration['scale']
    mapped = deepcopy(pairs)
    for row in mapped:
        if not np.isin([row['origin']['base'], *row['origin']['cf']], (0, 1)).all():
            raise ValueError('Origin must remain its original binary score')
        for backend in ('raw_ablation', 'repair'):
            values = [row[backend]['base'], *row[backend]['cf']]
            if len(values) != 3:
                raise ValueError('both intervention seeds must remain present')
            row[backend+'_intensity'] = deepcopy(row[backend])
            scores = calibrated_intensity(values, scale)
            row[backend] = {'base': float(scores[0]), 'cf': scores[1:].tolist()}
    return mapped


def aligned_statistics(pairs):
    prompts = sorted({r['prompt_id'] for r in pairs})
    rng = np.random.default_rng(20260923)
    prompt_index = rng.integers(0, 8, (20000, 8))
    source_index = rng.integers(0, 32, (20000, 32))
    stats, deltas = {}, {}
    for backend in ('origin', 'raw_ablation', 'repair'):
        values = np.array([[r[backend]['base'], *r[backend]['cf']] for r in pairs])
        base, cf = values[:, 0], values[:, 1:].mean(axis=1)
        delta = cf-base
        clusters = np.array([[np.mean(v[[i for i,r in enumerate(pairs) if r['prompt_id'] == p]])
                              for v in (base, cf, delta)] for p in prompts])
        ci = np.quantile(clusters[prompt_index].mean(axis=1), [.025, .975], axis=0)
        stats[backend] = {'base_mean': float(base.mean()), 'cf_mean': float(cf.mean()), 'delta': float(delta.mean()),
            'units': 'official_dynamic_video_fraction' if backend == 'origin' else 'dimensionless_calibrated_motion_intensity',
            'cf_mean_by_seed': {str(seed): float(values[:, i+1].mean()) for i,seed in enumerate((1701,2904))},
            'base_ci95_prompt_cluster': ci[:, 0].tolist(), 'cf_ci95_prompt_cluster': ci[:, 1].tolist(),
            'delta_ci95_prompt_cluster': ci[:, 2].tolist(),
            'delta_ci95_source': np.quantile(delta[source_index].mean(axis=1), [.025,.975]).tolist()}
        deltas[backend] = delta
    origin_delta, repair_delta = stats['origin']['delta'], stats['repair']['delta']
    margin = deltas['repair']-.1*deltas['origin']
    margin_clusters = np.array([np.mean([margin[i] for i,r in enumerate(pairs) if r['prompt_id']==p]) for p in prompts])
    gap = np.array([r['repair']['base']-r['origin']['base'] for r in pairs])
    gap_clusters = np.array([np.mean([gap[i] for i,r in enumerate(pairs) if r['prompt_id']==p]) for p in prompts])
    stats['criterion'] = {'status': 'evaluated_on_frozen_calibrated_score_scale',
        'repair_delta_limit': .1*origin_delta,
        'repair_over_origin_delta': repair_delta/origin_delta if origin_delta>0 else None,
        'passes_batch_point_estimate': bool(repair_delta <= .1*origin_delta) if origin_delta>0 else None,
        'margin_ci95_prompt_cluster': np.quantile(margin_clusters[prompt_index].mean(axis=1), [.025,.975]).tolist(),
        'scope': 'DEV32 numerical point estimate only, not physical-motion validity or held-out success'}
    stats['baseline_alignment'] = {'repair_minus_origin': float(gap.mean()),
        'gap_ci95_prompt_cluster': np.quantile(gap_clusters[prompt_index].mean(axis=1), [.025,.975]).tolist(),
        'target_offset': 0., 'refitted_on_dev32': False}
    stats['ci_scope'] = '20,000 fixed-scale paired prompt-cluster/source bootstrap replicates; calibration uncertainty is separate'
    return stats


def apply_frozen(args):
    fit_path, evaluation, config_path = Path(args.calibration), Path(args.evaluation), Path(args.config)
    fit = json.loads(fit_path.read_text())
    config = json.loads(config_path.read_text())
    if (config['protocol'] != 'dev32-cotracker3-calibrated-intensity-v1'
            or digest(evaluation/'summary.json') != config['evaluation_summary_sha256']
            or fit['config_sha256'] != config['fit_config_sha256']
            or fit['status'] != 'fitted_natural_scale_not_validated_repair'
            or fit['formula'] != 'I / (I + scale)' or fit['offset'] != 0 or fit['exponent'] != 1):
        raise ValueError('fixed calibration/evaluation protocol differs')
    previous = json.loads((evaluation/'summary.json').read_text())
    records = list(map(json.loads, (evaluation/'scores.jsonl').read_text().splitlines()))
    if len(records) != 128 or len({r['candidate_id'] for r in records}) != 128:
        raise ValueError('all 128 evaluated records required')
    pairs = align_pairs(previous['pairs'], fit)
    mapped = deepcopy(records)
    lookup = {r['candidate_id']: r for r in records}
    for pair in previous['pairs']:
        for j,key in enumerate(pair['candidate_ids']):
            for backend in ('origin', 'repair', 'raw_ablation'):
                score = (lookup[key]['repair']['raw_ablation']['score'] if backend=='raw_ablation' else lookup[key][backend]['score'])
                expected = pair[backend]['base'] if j==0 else pair[backend]['cf'][j-1]
                if score != expected:
                    raise ValueError('evaluation scores and summary disagree')
    for row in mapped:
        row['repair_intensity'] = deepcopy(row['repair'])
        for head in ('guarded', 'raw_ablation'):
            record = row['repair'][head]
            record['intensity'] = record['score']
            record['intensity_units'] = record['units']
            record['score'] = float(calibrated_intensity(record['intensity'], fit['scale']))
            record['units'] = 'dimensionless_calibrated_motion_intensity'
        row['repair'].update(score=row['repair']['guarded']['score'],
            score_kind='calibrated_continuous_intensity', units='dimensionless_calibrated_motion_intensity',
            calibration_sha256=digest(fit_path))
    for row in mapped:
        if row['origin'] != lookup[row['candidate_id']]['origin']:
            raise ValueError('Origin was changed')
        if row['family'] == 'original':
            control = next(r for r in mapped if r['base_id']==row['base_id'] and r['family']=='encoding_control')
            if row['repair'] != control['repair']:
                raise ValueError('calibration broke exact encoding control')
    stats = aligned_statistics(pairs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'complete_calibrated_development_comparison_not_validated_repair',
               'groups': 32, 'videos_scored': 128, 'counterfactuals': 64,
               'origin_records_unchanged': 128, 'encoding_controls_exact': 32,
               'statistics': stats, 'pairs': pairs, 'calibration': str(fit_path),
               'calibration_sha256': digest(fit_path), 'scale': fit['scale'],
               'source_summary_sha256': digest(evaluation/'summary.json'),
               'source_scores_sha256': digest(evaluation/'scores.jsonl'),
               'config': config, 'config_sha256': digest(config_path), 'script_sha256': digest(Path(__file__)),
               'holdout': 'NOT RUN', 'natural_motion_noninferiority': 'NOT RUN',
               'new_cf_human_review': 'NOT RUN', 'new_model_calls_for_dev32': 0,
               'raw_intensity_statistics': previous['statistics']}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output/'scores.jsonl').open('x') as handle:
        for row in mapped:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    lines = ['# DEV32: frozen natural-scale calibration', '',
             f'63 independent natural DEV videos; I/(I+tau), tau={fit["scale"]:.12g}; no fitted offset.', '',
             '| Method | Base | CF | Signed change | Prompt-cluster 95% CI of change |',
             '|---|---:|---:|---:|---|']
    for backend in ('origin', 'raw_ablation', 'repair'):
        r = stats[backend]
        ci = r['delta_ci95_prompt_cluster']
        lines.append(f'| {backend} | {r["base_mean"]:.6f} | {r["cf_mean"]:.6f} | {r["delta"]:+.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |')
    lines.extend(['', 'This aligns numerical scales, not the meaning of dynamic fraction and motion intensity.',
                  f'10%-drift criterion: {stats["criterion"]}', '',
                  'All 128 Origin records preserved. Continuous raw intensity is also retained.',
                  'Natural-motion noninferiority, new CF human review and formal holdout: NOT RUN.', '',
                  '| Source | Prompt | Origin base→CF | Repair base→CF |', '|---|---|---:|---:|'])
    for r in pairs:
        lines.append(f'| {r["base_id"]} | {r["prompt_id"]} | {r["origin"]["base"]:.6f}→{np.mean(r["origin"]["cf"]):.6f} | '
                     f'{r["repair"]["base"]:.6f}→{np.mean(r["repair"]["cf"]):.6f} |')
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='audit and fit only the independent natural cohort')
    for name in ('run-root', 'execution-root', 'manifest', 'sources', 'config', 'output'):
        fit.add_argument('--'+name, required=True)
    apply = commands.add_parser('apply', help='map every DEV32 record using the already-fitted scale')
    for name in ('calibration', 'evaluation', 'config', 'output'):
        apply.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        fit_audited(args)
    else:
        apply_frozen(args)


if __name__ == '__main__':
    main()
