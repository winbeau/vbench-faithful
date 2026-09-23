"""Keep original VBench scores; replay ONLY Repair as continuous intensity.

Reuse the fully audited same-video model arrays without new inference. Never
overwrite the superseded binary run. Distinct units are explicit: no direct
10%-of-Origin drift verdict without an independently justified scale mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.motion_intensity import motion_intensity
from .static_jitter import digest
from .summarize_mp4_dev32 import audit_evidence


def summarize_pairs(pairs):
    if len(pairs) != 32 or len({r['base_id'] for r in pairs}) != 32:
        raise ValueError('all 32 unique source groups required')
    prompts = sorted({r['prompt_id'] for r in pairs})
    if len(prompts) != 8 or set(Counter(r['prompt_id'] for r in pairs).values()) != {4}:
        raise ValueError('fixed eight-prompt/four-source-per-prompt design required')
    rng = np.random.default_rng(20260922)
    samples = rng.integers(0, 32, (20000, 32))
    clusters = rng.integers(0, 8, (20000, 8))
    result = {}
    for backend in ('origin', 'raw_ablation', 'repair'):
        values = np.asarray([[r[backend]['base'], *r[backend]['cf']] for r in pairs], float)
        if values.shape != (32, 3) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('complete nonnegative finite base and both CF values required')
        if backend == 'origin' and not np.isin(values, (0, 1)).all():
            raise ValueError('original official VBench scores must remain binary')
        delta = values[:, 1:].mean(axis=1) - values[:, 0]
        per_prompt = np.array([np.mean([delta[i] for i, r in enumerate(pairs) if r['prompt_id'] == p]) for p in prompts])
        result[backend] = {'base_mean': float(values[:, 0].mean()), 'cf_mean': float(values[:, 1:].mean()),
                          'delta': float(delta.mean()),
                          'units': 'official_dynamic_video_fraction' if backend == 'origin' else 'short_side_lengths_per_second',
                          'cf_mean_by_seed': {str(s): float(values[:, j+1].mean()) for j, s in enumerate((1701, 2904))},
                          'delta_ci95_source_bootstrap': np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist(),
                          'delta_ci95_prompt_cluster_bootstrap': np.quantile(per_prompt[clusters].mean(axis=1), [.025, .975]).tolist(),
                          'increased_groups': int((delta > 0).sum()), 'decreased_groups': int((delta < 0).sum()),
                          'unchanged_groups': int((delta == 0).sum())}
    result['cross_metric_10_percent_criterion'] = {'status': 'NOT EVALUATED', 'passes': None,
        'reason': 'Original VBench dynamic fraction and continuous Repair motion intensity have different units; no independent scale mapping is available.'}
    raw_delta, repaired_delta = result['raw_ablation']['delta'], result['repair']['delta']
    result['same_unit_decomposition_ablation'] = {
        'raw_to_guarded_drift_reduction_fraction': 1-repaired_delta/raw_delta if raw_delta > 0 else None,
        'scope': 'same CoTracker3 estimator and continuous score head only; not an Origin ratio or a success claim'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'audit', 'manifest', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run, audit_path, manifest, config_path, output = map(Path,
            (args.run_root, args.audit, args.manifest, args.config, args.output))
    audit = json.loads(audit_path.read_text())
    config = json.loads(config_path.read_text())
    if (audit['status'] != 'complete_development_comparison_not_validated_repair'
            or audit['groups'] != 32 or audit['videos_scored'] != 128 or audit['encoding_controls_exact'] != 32
            or digest(manifest) != audit['manifest_sha256'] or config['sources_sha256'] != audit['sources_sha256']
            or config['protocol'] != 'dev32-cotracker3-guarded-intensity-v1'):
        raise ValueError('fully audited immutable source run and explicit intensity protocol required')
    ledger = {r['candidate_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
    records, cache_hashes = {}, {}
    started = time.monotonic()
    for shard in audit['shards']:
        folder = run / f'shard-{shard["shard"]}'
        if digest(folder / 'provenance.json') != shard['provenance_sha256'] or digest(folder / 'scores.jsonl') != shard['scores_sha256']:
            raise ValueError('source score/provenance changed')
        identity = json.loads((folder / 'provenance.json').read_text())
        for row in map(json.loads, (folder / 'scores.jsonl').read_text().splitlines()):
            key = row['candidate_id']
            if key in records or key not in ledger or row['input_sha256'] != ledger[key]['sha256']:
                raise ValueError('duplicate/missing/changed video identity')
            cache = folder / 'evidence' / (key + '.npz')
            sha = digest(cache)
            if sha != identity['evidence_sha256'][key]:
                raise ValueError('cached model evidence changed')
            audit_evidence(cache, ledger[key], row)
            with np.load(cache, allow_pickle=False) as z:
                times, shape = z['timestamps'], z['shape'][1:3]
                dt = np.diff(times)[:, None, None]
                raw = motion_intensity(z['raw_velocity'] * dt, times, shape)
                corrected = motion_intensity(z['corrected_velocity'] * dt, times, shape)
            # The original serialized official backend is copied unchanged.
            result = {k: row[k] for k in ('candidate_id', 'base_id', 'prompt_id', 'generator', 'family', 'seed',
                                         'input_sha256', 'construction_status', 'construction_reason', 'origin')}
            result['repair'] = {'status': 'experimental_continuous_score', 'score': corrected['score'],
                                'score_kind': 'continuous_intensity', 'units': corrected['units'],
                                'raw_ablation': raw, 'guarded': corrected,
                                'model_visibility_proxy': row['repair']['diagnostic']['reliable_pair_fraction'],
                                'subtracted': row['repair']['diagnostic']['subtracted'],
                                'evidence_sha256': sha, 'physical_correspondence': 'NOT VERIFIED'}
            records[key], cache_hashes[key] = result, sha
    if set(records) != set(ledger) or len(records) != 128:
        raise ValueError('all 128 inputs must be retained')
    pairs = []
    for previous in audit['pairs']:
        base, *cfs = [records[key] for key in previous['candidate_ids']]
        control = next(r for r in records.values() if r['base_id'] == base['base_id'] and r['family'] == 'encoding_control')
        if base['repair'] != control['repair'] or base['origin']['diagnostics'] != control['origin']['diagnostics']:
            raise ValueError('encoding control parity changed')
        pair = {k: previous[k] for k in ('base_id', 'prompt_id', 'generator', 'candidate_ids', 'construction_flags')}
        for backend in ('origin', 'raw_ablation', 'repair'):
            def score(r):
                return r['repair']['raw_ablation']['score'] if backend == 'raw_ablation' else r[backend]['score']
            pair[backend] = {'base': score(base), 'cf': [score(r) for r in cfs]}
        if pair['origin'] != previous['origin']:
            raise ValueError('official Origin was modified')
        pairs.append(pair)
    statistics = summarize_pairs(pairs)
    root = Path(__file__).resolve().parents[2]
    summary = {'status': 'complete_continuous_repair_replay_not_validated_repair', 'groups': 32, 'videos_scored': 128,
        'counterfactuals': 64, 'encoding_controls_exact': 32, 'origin_records_unchanged': 128,
        'new_model_inference_calls': 0, 'source_audit_sha256': digest(audit_path),
        'source_audit': str(audit_path), 'config': config, 'config_sha256': digest(config_path),
        'code_sha256': {str(p.relative_to(root)): digest(p) for p in
                       (Path(__file__), root/'metrics/dynamic-degree/src/dynamic_degree/motion_intensity.py',
                        root/'scripts/counterfactual/summarize_mp4_dev32.py')},
        'evidence_sha256': cache_hashes, 'statistics': statistics, 'pairs': pairs,
        'python': platform.python_version(), 'numpy': np.__version__, 'elapsed_seconds': time.monotonic()-started,
        'human_review_new_counterfactuals': 'NOT RUN', 'natural_motion_noninferiority': 'NOT RUN', 'holdout': 'NOT RUN'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output/'scores.jsonl').open('x') as f:
        for key in sorted(records):
            f.write(json.dumps(records[key], allow_nan=False) + '\n')
    lines = ['# DEV32: original Origin + continuous Repair', '',
        'Origin is unchanged official VBench. Only Repair is changed to continuous intensity. The rows have different units.', '',
        '| Backend | Unit | Base | CF (two seeds) | Signed change | Prompt-cluster 95% CI |',
        '|---|---|---:|---:|---:|---|']
    for backend in ('origin', 'raw_ablation', 'repair'):
        s = statistics[backend]
        lines.append(f'| {backend} | {s["units"]} | {s["base_mean"]:.6f} | {s["cf_mean"]:.6f} | {s["delta"]:+.6f} | {s["delta_ci95_prompt_cluster_bootstrap"]} |')
    lines += ['', 'No binary Repair decisions, scale fitting, probability mapping or clipping. No cross-unit 10% success claim.', '',
              '| Source | Generator | Origin base→CF1701/2904 | Repair intensity base→CF1701/2904 |', '|---|---|---|---|']
    for row in pairs:
        cells = [f'{row[b]["base"]:.6f} → {row[b]["cf"][0]:.6f}/{row[b]["cf"][1]:.6f}' for b in ('origin', 'repair')]
        lines.append('| '+' | '.join([row['base_id'], row['generator'], *cells])+' |')
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'groups': 32, 'origin_unchanged': 128, 'statistics': statistics}, indent=2))


if __name__ == '__main__':
    main()
