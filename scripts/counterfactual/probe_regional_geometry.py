"""Replay all regional sparse matches as geometric proposals, never scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from dynamic_degree import regional_geometry as geometry
from dynamic_degree import native_region_motion as native
from vbench_audit_models.sam_regions import unpack_masks
from .static_jitter import digest


def validate_source(source):
    source = Path(source)
    identity = json.loads((source / 'provenance.json').read_text())
    if (identity['status'] != 'diagnostic_only' or identity['score'] is not None
            or digest(source / 'diagnostics.jsonl') != identity['diagnostics_sha256']
            or identity['completed'] != len(identity['common_identity']['full_cohort'])):
        raise ValueError('complete hash-bound merged diagnostics required')
    return identity


def replay_pair(pair, target_masks=None):
    rows = []
    for region in pair['regions']:
        if target_masks is None:
            result=geometry.regional_geometry(pair['sift_matches'], region['sift_match_source_keys'])
        else:
            result=native.rank_with_common_sift(region,target_masks,pair['sift_matches'])
            index=result['best_compatible_hypothesis']
            result.update(photometric_displacement_pixels=region['displacement_pixels'],
                          compatible_displacement_pixels=region['hypotheses'][index]['displacement_pixels'] if index is not None else None)
        rows.append({'region': region['region'], 'area_pixels': region['area_pixels'],
                     'whole_frame_control': region['whole_frame_control'], **result})
    return {'start': pair['start'], 'lag': pair['lag'], 'seconds': pair['seconds'], 'regions': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--analysis', choices=['geometry','common-sift'], default='geometry')
    parser.add_argument('--region-run', help='bound original SAM cache, required for common-sift')
    args = parser.parse_args(argv)
    if args.analysis=='common-sift' and not args.region_run:
        parser.error('common-sift requires --region-run')
    source, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise FileExistsError('fresh output required; do not overwrite another probe')
    identity = validate_source(source)
    expected = identity['common_identity']['full_cohort']
    region_root=Path(args.region_run) if args.analysis=='common-sift' else None
    region_id=None
    if region_root is not None:
        region_id=json.loads((region_root/'provenance.json').read_text())
        if (digest(region_root/'provenance.json')!=identity['common_identity']['region_provenance_sha256']
                or region_id['input_sha256']!=identity['common_identity']['input_sha256']):
            raise ValueError('native input and SAM identity differ')
    output.mkdir(parents=True)
    provenance = {'status': 'diagnostic_only', 'score': None, 'formal_acceptance': 'NOT EVALUATED',
                  'source_run': str(source), 'source_provenance_sha256': digest(source / 'provenance.json'),
                  'source_diagnostics_sha256': identity['diagnostics_sha256'],
                  'source_identity': identity['common_identity'],
                  'analysis': args.analysis, 'script_sha256': digest(Path(__file__)),
                  'algorithm_sha256': digest(Path(geometry.__file__ if args.analysis=='geometry' else native.__file__)),
                  'scope': 'all source phases/regions; no motion gate, score, or automatic reliability certification',
                  'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'}}
    if args.analysis=='geometry':
        provenance.update(models=list(geometry.MODELS),estimators=['least_squares','huber_irls'],irls_iterations=20,
                          huber_multiplier_of_fitted_residual_scale=1.345,
                          folds='sorted unique source coordinates modulo 2; duplicate orientations stay together')
    else:
        provenance.update(region_run=str(region_root),region_cache_sha256=region_id['cache_sha256'],
                          selection='rank by median residual on source locations common to ALL target mask candidates; not a calibrated confidence gate')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    runtime = {'status': 'running', 'pid': os.getpid(), 'completed': 0, 'expected': len(expected),
               'failed': 0, 'pairs_completed': 0, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    start = time.monotonic()
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    with (source / 'diagnostics.jsonl').open() as source_file, (output / 'diagnostics.jsonl').open('x') as out:
        for line in source_file:
            row = json.loads(line)
            if runtime['completed'] >= len(expected) or row['candidate_id'] != expected[runtime['completed']]:
                raise ValueError('source order/assignment differs from bound cohort')
            record = {k: row[k] for k in ('candidate_id', 'base_id', 'prompt_id', 'family', 'seed', 'status', 'score')}
            record['pairs'] = []
            try:
                if row['status'] != 'diagnostic_only' or row['score'] is not None:
                    raise ValueError('source failed/missing; do not fill missing motion')
                record['sampling'] = row['sampling']
                masks=None
                if region_root is not None:
                    cache=region_root/'evidence'/f'{row["candidate_id"]}.npz'
                    if digest(cache)!=region_id['cache_sha256'][row['candidate_id']]:raise ValueError('SAM cache changed')
                    with np.load(cache,allow_pickle=False) as z:
                        if not np.array_equal(z['timestamps'],row['sampling']['timestamps']):raise ValueError('SAM/native timeline differs')
                        masks=[unpack_masks(z[f'frame_{i}_masks_packed'],z[f'frame_{i}_image_shape']) for i in range(len(z['timestamps']))]
                for pair in row['pairs']:
                    target_masks=masks[pair['start']+pair['lag']] if masks is not None else None
                    record['pairs'].append(replay_pair(pair,target_masks))
                    runtime.update(pairs_completed=runtime['pairs_completed'] + 1,
                                   current_candidate=row['candidate_id'], current_start=pair['start'],
                                   current_lag=pair['lag'], elapsed_seconds=time.monotonic() - start)
                    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
            except Exception as exc:
                record.update(status='failed', score=None, error=f'{type(exc).__name__}: {exc}')
                runtime['failed'] += 1
            out.write(json.dumps(record, allow_nan=False) + '\n'); out.flush()
            runtime['completed'] += 1
            (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
            print(json.dumps({k: record[k] for k in ('candidate_id', 'status', 'score')}), flush=True)
    if runtime['completed'] != len(expected):
        raise ValueError('missing source videos; incomplete run retained')
    runtime.update(status='finished', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   elapsed_seconds=time.monotonic() - start)
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    provenance['diagnostics_sha256'] = digest(output / 'diagnostics.jsonl')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return 1 if runtime['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
