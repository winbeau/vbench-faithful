"""Audit DEV32 byte identities and report differences of batch mean scores.

No complete-case deletion, per-pair pass threshold, MAE substitution, or
positive-part averaging. Both CF seeds get half weight within each source.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from dynamic_degree.candidate_boolean import fixed_grid_decision
from dynamic_degree.image_plane_modes import region_partition, regional_reversal_ablation
from .score_mp4_dev32 import grid_queries, validate_cohort
from .static_jitter import digest


def batch_statistics(pairs, seed=20260922):
    if len(pairs) != 32 or len({r['base_id'] for r in pairs}) != 32:
        raise ValueError('32 unique source groups required; never substitute surviving cases')
    rng = np.random.default_rng(seed)
    source_boot = rng.integers(0, 32, (20000, 32))
    prompts = sorted({r['prompt_id'] for r in pairs})
    prompt_boot = rng.integers(0, len(prompts), (20000, len(prompts)))
    result, deltas = {}, {}
    for backend in ('origin', 'raw_ablation', 'repair'):
        values = np.asarray([[r[backend]['base'], *r[backend]['cf']] for r in pairs], float)
        if values.shape != (32, 3) or not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError('all base and both CF seed predictions required, binary scale only')
        delta = values[:, 1:].mean(axis=1) - values[:, 0]
        deltas[backend] = delta
        prompt_delta = np.array([np.mean([delta[i] for i, r in enumerate(pairs) if r['prompt_id'] == p]) for p in prompts])
        result[backend] = {'base_mean': float(values[:, 0].mean()), 'cf_mean': float(values[:, 1:].mean()),
                          'delta': float(delta.mean()),
                          'cf_mean_by_seed': {str(s): float(values[:, j+1].mean()) for j, s in enumerate((1701, 2904))},
                          'delta_ci95_source_bootstrap': np.quantile(delta[source_boot].mean(axis=1), [.025, .975]).tolist(),
                          'delta_ci95_prompt_cluster_bootstrap': np.quantile(prompt_delta[prompt_boot].mean(axis=1), [.025, .975]).tolist(),
                          'increased_groups': int((delta > 0).sum()), 'decreased_groups': int((delta < 0).sum()),
                          'unchanged_groups': int((delta == 0).sum())}
    origin, repair = result['origin']['delta'], result['repair']['delta']
    result['numerical_criterion'] = {'origin_increased': origin > 0, 'repair_limit': .1 * origin,
            'repair_over_origin_delta': repair / origin if origin > 0 else None,
            'passes_batch_point_estimate': bool(repair <= .1 * origin) if origin > 0 else None,
            'interpretation': 'DEV point estimate only, not overall Repair validity or natural-motion noninferiority',
            'margin_ci95_source_bootstrap': np.quantile(
                (deltas['repair'] - .1*deltas['origin'])[source_boot].mean(axis=1), [.025, .975]).tolist()}
    return result


def audit_evidence(path, row, scored):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    times, shape = arrays['timestamps'], arrays['shape']
    if not np.array_equal(times, row['pts']) or not np.array_equal(shape, row['decoded_shape']):
        raise ValueError('evidence/input native geometry or time mismatch')
    count, height, width, _ = shape
    xy = grid_queries(height, width, 0)[:, 1:]
    from vbench_audit_models.sam_regions import unpack_masks
    masks = unpack_masks(arrays['sam_masks_packed'], arrays['sam_image_shape'])
    np.testing.assert_array_equal(arrays['owners'], region_partition(masks, xy))
    deltas, observed = [], []
    for t in range(count):
        tracks, visible = arrays[f'tracks_{t:03d}'], arrays[f'visible_{t:03d}']
        if tracks.shape != (count, 144, 2) or visible.shape != (count, 144) or not np.isfinite(tracks).all():
            raise ValueError('incomplete tracker cache')
        np.testing.assert_allclose(tracks[t], xy, atol=1e-4, rtol=0)
        if t < count - 1:
            inside = np.all((tracks >= 0) & (tracks <= [width-1, height-1]), axis=-1)
            deltas.append(tracks[t+1].astype(float) - tracks[t].astype(float))
            observed.append(visible[t] & visible[t+1] & inside[t] & inside[t+1])
    dt = np.diff(times)[:, None, None]
    np.testing.assert_allclose(arrays['raw_velocity'], np.stack(deltas) / dt, rtol=0, atol=0)
    np.testing.assert_array_equal(arrays['reliable_pair'], observed)
    diagnostic, replay = regional_reversal_ablation(arrays['raw_velocity'], times, arrays['xy'], arrays['owners'],
                                                   arrays['reliable_pair'], (height, width))
    for name in ('removal', 'corrected_velocity', 'proposed_component'):
        np.testing.assert_allclose(arrays[name], replay[name], rtol=1e-9, atol=1e-7)
    if scored['repair'].get('score_kind') == 'continuous_intensity':
        from dynamic_degree.motion_intensity import motion_intensity
        raw = motion_intensity(np.stack(deltas), times, (height, width))
        corrected = motion_intensity(arrays['corrected_velocity'] * dt, times, (height, width))
    else:
        raw = fixed_grid_decision(np.stack(deltas), (height, width))
        corrected = fixed_grid_decision(arrays['corrected_velocity'] * dt, (height, width))
    if raw != scored['repair']['raw_ablation'] or corrected != scored['repair']['guarded'] or corrected['score'] != scored['repair']['score']:
        raise ValueError('saved decision differs from all-phase array replay')
    from dynamic_degree.backends.vbench import official_check_move, official_parameters
    official = scored['origin']['diagnostics']['official']
    threshold, count_num = official_parameters((height, width), count)
    if (official['official_threshold'] != threshold or official['official_count_num'] != count_num
            or float(official_check_move(official['raw_flow_top5_mean'], threshold, count_num)) != scored['origin']['score']):
        raise ValueError('Origin decisions do not follow the pinned official formula')
    return {'tracker_calls': int(count), 'subtracted': diagnostic['subtracted'],
            'proxy_coverage': diagnostic['reliable_pair_fraction']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'manifest', 'sources', 'execution-root', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    run, manifest, sources_path, execution, output = map(Path,
            (args.run_root, args.manifest, args.sources, args.execution_root, args.output))
    sources = [json.loads(s) for s in sources_path.read_text().splitlines()]
    manifest_rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    validate_cohort(manifest_rows, sources)
    ledger = {r['candidate_id']: r for r in manifest_rows}
    scores, paths, audits, runtimes, identities = {}, {}, {}, [], []
    for shard in range(4):
        directory = run / f'shard-{shard}'
        identity = json.loads((directory / 'provenance.json').read_text())
        runtime = json.loads((directory / 'runtime.json').read_text())
        if (identity['manifest_sha256'] != digest(manifest) or identity['sources_sha256'] != digest(sources_path)
                or runtime['status'] != 'finished' or runtime['failed'] or runtime['completed'] != 32):
            raise ValueError('missing/failed shard retained; no complete 32-group claim')
        for name, sha in identity['code_files'].items():
            if digest(execution / name) != sha:
                raise ValueError(f'execution snapshot differs: {name}')
        for item in map(json.loads, (directory / 'scores.jsonl').read_text().splitlines()):
            key = item['candidate_id']
            if key in scores or key not in ledger or item['status'] != 'finished' or item['input_sha256'] != ledger[key]['sha256']:
                raise ValueError('duplicate, unknown, failed or changed candidate')
            path = directory / 'evidence' / (key + '.npz')
            if digest(path) != identity['evidence_sha256'][key]:
                raise ValueError('evidence bytes changed')
            scores[key], paths[key] = item, path
            audits[key] = audit_evidence(path, ledger[key], item)
        runtimes.append(runtime)
        identities.append({'shard': shard, 'provenance_sha256': digest(directory / 'provenance.json'),
                           'scores_sha256': digest(directory / 'scores.jsonl')})
    if set(scores) != set(ledger):
        raise ValueError('not all 128 inputs scored')
    pairs, controls = [], []
    for source in sources:
        rows = [r for r in manifest_rows if r['base_id'] == source['video_uid']]
        def get(family, seed):
            return scores[next(r['candidate_id'] for r in rows if r['family'] == family and r['seed'] == seed)]
        base, control = get('original', 0), get('encoding_control', 0)
        cf = [get('local_texture_alternating', s) for s in (1701, 2904)]
        with np.load(paths[base['candidate_id']], allow_pickle=False) as a, np.load(paths[control['candidate_id']], allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a.files):
                raise ValueError('encoding control predictions differ')
        if base['origin']['diagnostics'] != control['origin']['diagnostics'] or base['repair'] != control['repair']:
            raise ValueError('encoding control scores/diagnostics differ')
        controls.append(source['video_uid'])
        pair = {'base_id': source['video_uid'], 'prompt_id': source['prompt_id'], 'generator': source['generator'],
                'candidate_ids': [base['candidate_id'], *[r['candidate_id'] for r in cf]],
                'construction_flags': [ledger[r['candidate_id']].get('reason') for r in cf]}
        for backend in ('origin', 'repair', 'raw_ablation'):
            def value(r):
                return r['repair']['raw_ablation']['score'] if backend == 'raw_ablation' else r[backend]['score']
            pair[backend] = {'base': value(base), 'cf': [value(r) for r in cf]}
        pairs.append(pair)
    statistics = batch_statistics(pairs)
    start = min(datetime.fromisoformat(r['started_utc'].replace('Z', '+00:00')) for r in runtimes)
    stop = max(datetime.fromisoformat(r['finished_utc'].replace('Z', '+00:00')) for r in runtimes)
    summary = {'status': 'complete_development_comparison_not_validated_repair', 'groups': 32,
               'videos_scored': 128, 'counterfactuals': 64, 'encoding_controls_exact': len(controls),
               'tracker_calls': sum(a['tracker_calls'] for a in audits.values()), 'sam_calls': 128,
               'sources_sha256': digest(sources_path), 'manifest_sha256': digest(manifest),
               'analysis_script_sha256': digest(Path(__file__)), 'shards': identities,
               'statistics': statistics, 'pairs': pairs, 'evidence_audit': audits,
               'construction_status_counts': dict(Counter(r['status'] for r in manifest_rows)),
               'construction_reasons': dict(Counter(r.get('reason') for r in manifest_rows if r.get('reason'))),
               'four_gpu_wall_seconds': (stop-start).total_seconds(), 'runtimes': runtimes,
               'human_review_new_counterfactuals': 'NOT RUN', 'natural_motion_noninferiority': 'NOT RUN',
               'independent_physical_correspondence': 'NOT VERIFIED', 'formal_holdout': 'NOT RUN'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# DEV32 official-MP4 8px batch comparison', '',
             '32 sources; CF averages the two fixed seeds within source. Development candidate, not validated Repair.', '',
             '| Backend | Base mean | CF mean | Δ | Source bootstrap 95% CI |', '|---|---:|---:|---:|---|']
    for backend in ('origin', 'raw_ablation', 'repair'):
        s = statistics[backend]
        lines.append(f'| {backend} | {s["base_mean"]:.6f} | {s["cf_mean"]:.6f} | {s["delta"]:+.6f} | {s["delta_ci95_source_bootstrap"]} |')
    lines += ['', json.dumps(statistics['numerical_criterion']), '',
              '| Source | Generator | Origin base→CF1701/2904 | Raw tracker base→CF | Guarded candidate base→CF |',
              '|---|---|---|---|---|']
    for row in pairs:
        cells = [f'{row[b]["base"]:g} → {row[b]["cf"][0]:g}/{row[b]["cf"][1]:g}' for b in ('origin', 'raw_ablation', 'repair')]
        lines.append('| ' + ' | '.join([row['base_id'], row['generator'], *cells]) + ' |')
    lines += ['', 'Natural-motion noninferiority, independent physical correspondence and new CF human review remain unverified.']
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: summary[k] for k in ('groups', 'videos_scored', 'tracker_calls', 'encoding_controls_exact', 'statistics')}, indent=2))


if __name__ == '__main__':
    main()
