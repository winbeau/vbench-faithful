"""Replay the fixed residual-first candidate on all DEV32 + 63 natural inputs.

No inference, calibration fitting, CF construction, exclusion or threshold
search. Per-video motion decomposition receives no paired or construction data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree.intensity_scale import calibrated_intensity
from dynamic_degree.motion_intensity import motion_intensity
from dynamic_degree.residual_reversal import residual_first_reversal
from dynamic_degree.outer_support import outer_support_reversal
from .calibrate_motion_intensity import aligned_statistics
from .static_jitter import digest
from .summarize_mp4_dev32 import audit_evidence


def natural_change(rows):
    if len(rows) != 63 or len({r['base_id'] for r in rows}) != 63:
        raise ValueError('all 63 distinct natural controls required')
    prompts = sorted({r['prompt_id'] for r in rows})
    if len(prompts) != 21 or set(Counter(r['prompt_id'] for r in rows).values()) != {3}:
        raise ValueError('balanced 21-prompt natural cohort required')
    before = np.array([r['previous_repair']['score'] for r in rows])
    after = np.array([r['repair']['score'] for r in rows])
    delta = after-before
    clusters = np.array([np.mean([delta[i] for i,r in enumerate(rows) if r['prompt_id']==p]) for p in prompts])
    samples = np.random.default_rng(20260923).integers(0, 21, (20000, 21))
    return {'videos': 63, 'previous_mean': float(before.mean()), 'new_mean': float(after.mean()),
            'mean_change': float(delta.mean()), 'mean_absolute_change': float(abs(delta).mean()),
            'largest_drop': float(min(0., delta.min())), 'largest_increase': float(max(0., delta.max())),
            'unchanged_within_1e12': int((abs(delta)<=1e-12).sum()),
            'changed_videos': [r['base_id'] for i,r in enumerate(rows) if abs(delta[i])>1e-12],
            'mean_change_ci95_prompt_cluster': np.quantile(clusters[samples].mean(axis=1), [.025,.975]).tolist(),
            'interpretation': 'score retention on reused natural DEV controls, not human-motion noninferiority'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', default='output/dynamic-static-jitter')
    parser.add_argument('--config', default='configs/dynamic-static-jitter/scoring.residual-reversal-dev32-v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, config_path, output = Path(args.task_root), Path(args.config), Path(args.output)
    config = json.loads(config_path.read_text())
    protocols = {'protected-residual-reversal-dev32-v1', 'outer-support-residual-reversal-dev32-v1'}
    if config['protocol'] not in protocols:
        raise ValueError('explicit fixed residual-first protocol required')
    outer_support = config['protocol'] == 'outer-support-residual-reversal-dev32-v1'
    fit_path = root/'natural63-calibration-fit-v1/calibration.json'
    previous_path = root/'dev32-cotracker3-calibrated-intensity-v1/summary.json'
    if digest(fit_path) != config['calibration_artifact_sha256'] or digest(previous_path) != config['evaluation_summary_sha256']:
        raise ValueError('calibration or prior evaluation identity differs')
    fitted, previous = json.loads(fit_path.read_text()), json.loads(previous_path.read_text())
    if fitted['scale'] != config['tau']:
        raise ValueError('scale must remain frozen')
    old_records = {r['candidate_id']: r for r in map(json.loads,
        (previous_path.parent/'scores.jsonl').read_text().splitlines())}
    development_audit = json.loads((root/'dev32-cotracker3-boolean-v1-audit/summary.json').read_text())
    datasets = [('dev32', root/'dev32-cotracker3-boolean-v1',
                 root/'local-texture-dev32-mp4-8px-v1/candidates.jsonl', development_audit['shards'], 128),
                ('natural63', root/'natural63-calibration-scoring-v1',
                 root/'natural63-calibration-inputs-v1/candidates.jsonl', fitted['shards'], 63)]
    output.mkdir(parents=True, exist_ok=False)
    (output/'evidence').mkdir()
    started = time.monotonic()
    records, input_hashes, output_hashes = {}, {}, {}
    runtime = {'status': 'running', 'expected': 191, 'completed': 0, 'failed': 0,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'python': platform.python_version(), 'numpy': np.__version__}

    def checkpoint():
        runtime['elapsed_seconds'] = time.monotonic()-started
        (output/'runtime.json').write_text(json.dumps(runtime, indent=2))

    checkpoint()
    try:
        with (output/'scores.jsonl').open('x') as handle:
            for dataset, run, manifest, shard_receipts, expected in datasets:
                ledger = {r['candidate_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
                seen = set()
                for shard in shard_receipts:
                    folder = run/f'shard-{shard["shard"]}'
                    for name in ('scores', 'provenance'):
                        if digest(folder/(name+'.jsonl' if name=='scores' else name+'.json')) != shard[name+'_sha256']:
                            raise ValueError('source score/provenance bytes differ')
                    identity = json.loads((folder/'provenance.json').read_text())
                    if identity['manifest_sha256'] != digest(manifest):
                        raise ValueError('manifest identity differs')
                    for row in map(json.loads, (folder/'scores.jsonl').read_text().splitlines()):
                        key = row['candidate_id']
                        if key in seen or key not in ledger or row['status'] != 'finished':
                            raise ValueError('duplicate, unknown or failed source record')
                        seen.add(key)
                        cache = folder/'evidence'/(key+'.npz')
                        if digest(cache) != identity['evidence_sha256'][key]:
                            raise ValueError('source model cache differs')
                        audit_evidence(cache, ledger[key], row)
                        with np.load(cache, allow_pickle=False) as z:
                            times, shape = z['timestamps'], z['shape'][1:3]
                            if outer_support:
                                masks = np.unpackbits(z['sam_masks_packed'], axis=-1, count=int(shape[1])).astype(bool)
                                diagnostic, arrays = outer_support_reversal(z['raw_velocity'], times, z['xy'], masks, z['reliable_pair'], shape)
                            else:
                                diagnostic, arrays = residual_first_reversal(z['raw_velocity'], times, z['xy'], z['owners'], z['reliable_pair'], shape)
                                np.testing.assert_allclose(arrays['baseline_corrected_velocity'], z['corrected_velocity'], rtol=0, atol=1e-7)
                            old_intensity = motion_intensity(z['corrected_velocity']*np.diff(times)[:, None, None], times, shape)
                            intensity = motion_intensity(arrays['corrected_velocity']*np.diff(times)[:, None, None], times, shape)
                        old_score = float(calibrated_intensity(old_intensity['score'], config['tau']))
                        if dataset == 'dev32':
                            if row['origin'] != old_records[key]['origin'] or abs(old_score-old_records[key]['repair']['score'])>1e-12:
                                raise ValueError('previous Origin or calibrated Repair failed parity')
                        else:
                            reference = next(r for r in fitted['cohort'] if r['candidate_id']==key)
                            if reference['origin'] != row['origin']['score'] or abs(reference['intensity']-old_intensity['score'])>1e-12:
                                raise ValueError('natural calibration score differs')
                        record = {k: row[k] for k in ('candidate_id', 'base_id', 'prompt_id', 'generator', 'family', 'seed', 'origin', 'input_sha256')}
                        record.update(dataset=dataset, status='finished',
                            previous_repair={'score': old_score, 'intensity': old_intensity},
                            repair={'score': float(calibrated_intensity(intensity['score'], config['tau'])),
                                    'units': 'dimensionless_calibrated_motion_intensity', 'score_kind': 'continuous',
                                    'intensity': intensity, 'diagnostic': diagnostic})
                        identity_key = dataset+'/'+key
                        evidence = output/'evidence'/(dataset+'_'+key+'.npz')
                        np.savez_compressed(evidence, **arrays)
                        input_hashes[identity_key], output_hashes[identity_key] = digest(cache), digest(evidence)
                        records[identity_key] = record
                        handle.write(json.dumps(record, allow_nan=False)+'\n')
                        handle.flush()
                        runtime['completed'] += 1
                        checkpoint()
                        if runtime['completed'] % 16 == 0:
                            print(json.dumps({'completed': runtime['completed'], 'expected': 191}), flush=True)
                if len(seen) != expected or seen != set(ledger):
                    raise ValueError('all declared inputs must be retained')
        pairs = deepcopy(previous['pairs'])
        for pair in pairs:
            base, *cf = [records['dev32/'+key] for key in pair['candidate_ids']]
            control = next(r for r in records.values() if r['dataset']=='dev32' and r['base_id']==pair['base_id'] and r['family']=='encoding_control')
            if base['repair'] != control['repair']:
                raise ValueError('encoding control score or diagnostic mismatch')
            with np.load(output/'evidence'/('dev32_'+base['candidate_id']+'.npz')) as a, np.load(output/'evidence'/('dev32_'+control['candidate_id']+'.npz')) as b:
                if set(a.files)!=set(b.files) or any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a.files):
                    raise ValueError('encoding control arrays differ')
            pair['previous_repair'] = deepcopy(pair['repair'])
            pair['repair'] = {'base': base['repair']['score'], 'cf': [r['repair']['score'] for r in cf]}
            pair['repair_intensity'] = {'base': base['repair']['intensity']['score'], 'cf': [r['repair']['intensity']['score'] for r in cf]}
        natural = [r for r in records.values() if r['dataset']=='natural63']
        if {r['prompt_id'] for r in natural} & {r['prompt_id'] for r in pairs}:
            raise ValueError('natural controls overlap DEV32 prompts')
        stats = aligned_statistics(pairs)
        project = Path(__file__).resolve().parents[2]
        code = [Path(__file__), project/'metrics/dynamic-degree/src/dynamic_degree/residual_reversal.py',
                project/'metrics/dynamic-degree/src/dynamic_degree/image_plane_modes.py',
                project/'metrics/dynamic-degree/src/dynamic_degree/motion_intensity.py',
                project/'metrics/dynamic-degree/src/dynamic_degree/intensity_scale.py']
        if outer_support:
            code.append(project/'metrics/dynamic-degree/src/dynamic_degree/outer_support.py')
        summary = {'status': 'complete_residual_first_development_candidate_not_validated_repair',
            'protocol': config['protocol'],
            'groups': 32, 'counterfactuals': 64, 'dev_inputs': 128, 'natural_controls': 63,
            'origin_records_unchanged': 191, 'encoding_controls_exact': 32, 'new_model_calls': 0,
            'calibration_refitted': False, 'scale': config['tau'],
            'statistics': stats, 'previous_statistics': previous['statistics'],
            'natural_score_retention': natural_change(natural), 'pairs': pairs,
            'extra_subtraction_counts': dict(Counter(r['dataset']+'/'+r['family'] for r in records.values()
                                                   if r['repair']['diagnostic']['extra_subtracted'])),
            'config': config, 'config_sha256': digest(config_path),
            'code_sha256': {str(p.relative_to(project)): digest(p) for p in code},
            'input_evidence_sha256': input_hashes, 'new_evidence_sha256': output_hashes,
            'natural_motion_noninferiority': 'NOT RUN', 'holdout': 'NOT RUN', 'public_default_changed': False}
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        lines = ['# DEV32 '+config['protocol']+': fixed score scale', '',
                 '| Method | Base | CF | Signed change |', '|---|---:|---:|---:|']
        for name, values in [('Origin', stats['origin']), ('Previous Repair', previous['statistics']['repair']), ('Residual-first Repair', stats['repair'])]:
            lines.append(f'| {name} | {values["base_mean"]:.6f} | {values["cf_mean"]:.6f} | {values["delta"]:+.6f} |')
        lines.extend(['', 'All 128 DEV inputs and 63 natural controls retained; no new inference, scale refit or CF edits.',
                      '', 'Natural score retention: '+json.dumps(summary['natural_score_retention']),
                      '', '10%-drift numerical criterion: '+json.dumps(stats['criterion']),
                      '', 'Reused development evidence; real-motion noninferiority and formal holdout NOT RUN.'])
        (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
        runtime['status'] = 'finished'
        print(json.dumps({k: summary[k] for k in ('statistics', 'natural_score_retention', 'extra_subtraction_counts')}, indent=2), flush=True)
    except BaseException as exc:
        runtime.update(status='failed', failed=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        runtime['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        checkpoint()


if __name__ == '__main__':
    main()
