"""Fixed DEV32 run: original official RAFT plus continuous guarded CT3 Repair.

The legacy binary protocol remains explicitly reproducible, not the current
Repair head. Neither candidate is promoted to public/default Repair. The video-only
candidate never receives a partner, prompt, seed or construction field.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.candidate_boolean import fixed_grid_decision
from dynamic_degree.motion_intensity import motion_intensity
from dynamic_degree.image_plane_modes import region_partition, regional_reversal_ablation, spatial_design
from .official_video_jitter import native_video
from .static_jitter import digest


def grid_queries(height, width, phase, size=12):
    x = (np.arange(size) + .5) * width / size - .5
    y = (np.arange(size) + .5) * height / size - .5
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((np.full(xx.size, phase), xx.ravel(), yy.ravel())).astype(np.float32)


def evaluate_candidate(frames, times, tracker, sam, *, score_kind='continuous_intensity'):
    from vbench_audit_models.sam_regions import unpack_masks

    times = np.asarray(times, float)
    if len(frames) < 5 or not np.allclose(np.diff(times), .125, atol=1e-6, rtol=0):
        raise ValueError('predeclared experiment supports native 8 FPS CFR only; no implicit sampling')
    count, height, width, _ = frames.shape
    evidence = {'timestamps': times, 'shape': np.array(frames.shape)}
    deltas, observed = [], []
    xy = grid_queries(height, width, 0)[:, 1:].astype(float)
    for t in range(count):
        query = grid_queries(height, width, t)
        prediction = tracker.track_queries(frames, query)
        tracks, visible = prediction['tracks'], prediction['visible']
        if (tracks.shape != (count, len(xy), 2) or visible.shape != (count, len(xy))
                or visible.dtype != np.bool_ or not np.isfinite(tracks).all()
                or not np.allclose(tracks[t], xy, atol=1e-4, rtol=0)):
            raise ValueError('invalid full-phase tracker output')
        evidence[f'tracks_{t:03d}'] = tracks
        evidence[f'visible_{t:03d}'] = visible
        if t < count - 1:
            inside = np.all((tracks >= 0) & (tracks <= [width - 1, height - 1]), axis=-1)
            deltas.append(tracks[t + 1].astype(float) - tracks[t].astype(float))
            observed.append(visible[t] & visible[t + 1] & inside[t] & inside[t + 1])
    regions = sam.propose(frames[count // 2])
    evidence.update({'sam_' + k: v for k, v in regions.items()})
    masks = unpack_masks(regions['masks_packed'], regions['image_shape'])
    owners = region_partition(masks, xy)
    velocity = np.stack(deltas) / np.diff(times)[:, None, None]
    known = np.stack(observed)
    diagnostic, decomposition = regional_reversal_ablation(velocity, times, xy, owners, known, (height, width))
    removal = decomposition['removal']
    design = spatial_design(xy, (height, width), 0)
    affine_error = 0.
    for t in range(count - 1):
        for owner in np.unique(owners):
            selected = known[t] & (owners == owner)
            affine_error = max(affine_error, float(np.max(np.abs(design[selected].T @ removal[t, selected]), initial=0.)))
    mean_error = float(np.max(np.abs(np.sum(removal * np.diff(times)[:, None, None], axis=0))))
    if np.any(removal[~known]) or max(affine_error, mean_error) > 1e-5:
        raise ValueError('decomposition changed protected regional-affine/mean/unknown motion')
    corrected_delta = decomposition['corrected_velocity'] * np.diff(times)[:, None, None]
    if score_kind == 'continuous_intensity':
        raw = motion_intensity(np.stack(deltas), times, (height, width))
        corrected = motion_intensity(corrected_delta, times, (height, width))
    elif score_kind == 'legacy_boolean':
        raw = fixed_grid_decision(np.stack(deltas), (height, width))
        corrected = fixed_grid_decision(corrected_delta, (height, width))
    else:
        raise ValueError('explicit continuous-intensity or legacy-boolean head required')
    evidence.update(decomposition)
    return {'status': 'experimental_score', 'score': corrected['score'], 'score_kind': score_kind,
            'units': corrected['units'], 'raw_ablation': raw,
            'guarded': corrected, 'diagnostic': diagnostic,
            'native_shape': list(frames.shape), 'duration_seconds': float(count / 8),
            'tracker_calls': count, 'sam_calls': 1, 'sam_frame': count // 2,
            'independent_affine_error': affine_error, 'independent_mean_error': mean_error,
            'physical_correspondence_validation': 'NOT VERIFIED',
            'natural_motion_noninferiority': 'NOT RUN'}, evidence


def validate_cohort(rows, sources):
    if len(sources) != 32 or len({r['video_uid'] for r in sources}) != 32 or len(rows) != 128:
        raise ValueError('exactly 32 distinct sources and 128 candidate records required')
    expected = {(s['video_uid'], family, seed) for s in sources
                for family, seed in [('original', 0), ('encoding_control', 0),
                                     ('local_texture_alternating', 1701), ('local_texture_alternating', 2904)]}
    actual = {(r['base_id'], r['family'], r['seed']) for r in rows}
    if actual != expected or len({r['candidate_id'] for r in rows}) != 128:
        raise ValueError('incomplete/duplicate or unexpected candidate cohort')
    if any(r['split'] != 'dev' or r['protocol'] != 'official-video-local-texture-jitter-v1' for r in rows):
        raise ValueError('only declared DEV local texture construction allowed')
    if any(r['amplitude'] != (8 if r['seed'] else 0) for r in rows):
        raise ValueError('intervention dose changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'sources', 'config', 'tracker-source-manifest', 'tracker-root', 'tracker-weight',
                 'sam-root', 'sam-weight', 'sam-reference', 'sam-config', 'raft-weight', 'upstream', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=4)
    args = p.parse_args()
    if not 0 <= args.shard < args.shards:
        p.error('invalid shard')
    import torch
    from dynamic_degree.backends.vbench import OfficialDynamicEvaluator, official_result_payload
    from vbench_audit_models.cotracker3 import CoTracker3OfflineModel
    from vbench_audit_models.sam_regions import SamRegionModel

    config = json.loads(Path(args.config).read_text())
    protocols = {'dev32-cotracker3-guarded-intensity-v1': 'continuous_intensity',
                 'natural63-cotracker3-intensity-calibration-v1': 'continuous_intensity',
                 'dev32-cotracker3-guarded-official-decision-v1': 'legacy_boolean'}
    if config['protocol'] not in protocols or digest(Path(args.sources)) != config['sources_sha256']:
        raise ValueError('frozen configuration/source identity differs')
    sources = [json.loads(s) for s in Path(args.sources).read_text().splitlines()]
    rows = [json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
    calibration_run = config['protocol'] == 'natural63-cotracker3-intensity-calibration-v1'
    if calibration_run:
        from .select_intensity_calibration import validate_calibration
        validate_calibration(rows, sources)
        forbidden = [r for path, sha in config['excluded_sources_sha256'].items()
                     for r in _verified_sources(Path(path), sha)]
        if ({r['prompt_id'] for r in forbidden} & {r['prompt_id'] for r in sources}
                or {r['video_uid'] for r in forbidden} & {r['video_uid'] for r in sources}):
            raise ValueError('calibration overlaps evaluated batch or reserved holdout')
    else:
        validate_cohort(rows, sources)
    selected = {s['video_uid'] for s in sources[args.shard::args.shards]}
    rows = [r for r in rows if r['base_id'] in selected]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'evidence').mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cv2.setNumThreads(1)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError('at least 40 GiB free memory required; no models started')
    torch.cuda.set_per_process_memory_fraction(.20)
    root = Path(__file__).resolve().parents[2]
    code_paths = sorted(set(root.glob('metrics/dynamic-degree/src/**/*.py')) |
                        set(root.glob('packages/audit-core/src/**/*.py')) |
                        set(root.glob('packages/audit-models/src/**/*.py')) |
                        set(root.glob('scripts/counterfactual/*.py')))
    identity = {'config': config, 'config_sha256': digest(Path(args.config)),
                'manifest_sha256': digest(Path(args.manifest)), 'sources_sha256': digest(Path(args.sources)),
                'script_sha256': digest(Path(__file__)),
                'code_files': {str(f.relative_to(root)): digest(f) for f in code_paths},
                'expected_ids': [r['candidate_id'] for r in rows],
                'shard': args.shard, 'shards': args.shards, 'evidence_sha256': {},
                'arguments': vars(args), 'public_default_changed': False,
                'construction_quality_exclusions_applied': False,
                'human_review': 'NOT RUN', 'calibration_natural_only': calibration_run}
    runtime = {'status': 'running', 'pid': os.getpid(), 'expected': len(rows), 'completed': 0, 'failed': 0,
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'numpy': np.__version__, 'opencv': cv2.__version__, 'python_executable': os.sys.executable,
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': torch.cuda.get_device_name(0)}
    def checkpoint():
        (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        (output / 'provenance.json').write_text(json.dumps(identity, indent=2))
    checkpoint()
    start = time.monotonic()
    try:
        manifest = json.loads(Path(args.tracker_source_manifest).read_text())
        tracker = CoTracker3OfflineModel(Path(args.tracker_root), Path(args.tracker_weight), 'cuda:0',
                    expected_source=manifest['source_files'],
                    expected_sha256='2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834',
                    expected_size_bytes=101890938)
        sam_reference = json.loads(Path(args.sam_reference).read_text())['model_identity']
        if digest(Path(args.sam_weight)) != sam_reference['checkpoint_sha256']:
            raise ValueError('SAM weight identity differs')
        sam = SamRegionModel(args.sam_root, args.sam_weight, 'cuda:0',
                             json.loads(Path(args.sam_config).read_text())['generator'])
        if sam.identity != sam_reference:
            raise ValueError('SAM model/source/config identity differs')
        origin = OfficialDynamicEvaluator(torch.device('cuda:0'), Path(args.raft_weight), Path(args.upstream))
        if digest(Path(args.raft_weight)) != 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1':
            raise ValueError('official RAFT weight differs')
        identity.update(tracker=tracker.identity, sam=sam.identity, official_upstream=asdict(origin.upstream_state),
                        raft_weight_sha256=digest(Path(args.raft_weight)))
        checkpoint()
        with (output / 'scores.jsonl').open('x') as handle:
            for ordinal, row in enumerate(rows):
                began = time.monotonic()
                result = {k: row[k] for k in ('candidate_id', 'base_id', 'prompt_id', 'generator', 'family', 'seed')}
                result.update(input_sha256=row.get('sha256'), construction_status=row['status'],
                              construction_reason=row.get('reason'), origin={'score': None}, repair={'score': None})
                try:
                    video = Path(row['video'])
                    if not row.get('sha256') or digest(video) != row['sha256']:
                        raise ValueError('missing or changed input')
                    frames, times, fps = native_video(video, 1e-6)
                    if (abs(fps - 8) > 1e-6 or hashlib.sha256(frames.tobytes()).hexdigest() != row['decoded_pixels_sha256']
                            or not row['pixel_exact_to_intended'] or not row['native_timeline_preserved']):
                        raise ValueError('native video/timeline/pixel verification differs')
                    for backend in ('origin', 'repair'):
                        try:
                            if backend == 'origin':
                                measured = origin.evaluate_video(video)
                                result['origin'] = {'status': 'succeeded', 'score': float(measured.official_video_boolean),
                                                    'diagnostics': official_result_payload(measured)}
                                if ordinal == 0:
                                    parity = bool(origin.dynamic.infer(str(video)))
                                    if parity != measured.official_video_boolean:
                                        raise ValueError('official infer parity failed')
                                    result['origin']['reference_infer_parity'] = True
                            else:
                                result['repair'], arrays = evaluate_candidate(frames, times, tracker, sam,
                                                        score_kind=protocols[config['protocol']])
                                cache = output / 'evidence' / (row['candidate_id'] + '.npz')
                                np.savez_compressed(cache, **arrays)
                                identity['evidence_sha256'][row['candidate_id']] = digest(cache)
                        except Exception as exc:
                            result[backend] = {'status': 'failed', 'score': None, 'error': f'{type(exc).__name__}: {exc}'}
                    result['status'] = 'finished' if all(result[b]['score'] is not None for b in ('origin', 'repair')) else 'failed'
                except Exception as exc:
                    result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                result['seconds'] = time.monotonic() - began
                handle.write(json.dumps(result, allow_nan=False) + '\n')
                handle.flush()
                runtime['completed'] += 1
                runtime['failed'] += result['status'] != 'finished'
                runtime.update(current_candidate=row['candidate_id'], elapsed_seconds=time.monotonic() - start)
                checkpoint()
                print(json.dumps({'candidate_id': row['candidate_id'], 'status': result['status'],
                                  'origin': result['origin']['score'], 'repair': result['repair']['score'],
                                  'seconds': result['seconds'], 'error': result.get('error')}), flush=True)
        runtime['status'] = 'finished'
    except BaseException as exc:
        runtime.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        runtime.update(finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       elapsed_seconds=time.monotonic() - start,
                       gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
        checkpoint()
    return 1 if runtime['failed'] else 0


def _verified_sources(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError('excluded cohort identity differs')
    return [json.loads(s) for s in path.read_text().splitlines()]


if __name__ == '__main__':
    raise SystemExit(main())
