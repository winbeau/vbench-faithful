"""Frozen aligned-v1 evaluation on score-blind external 128-prompt cohorts.

New output directories only. Select and hash inputs before construction/scoring.
Three fixed 16-frame/8fps windows; Origin remains its original per-clip boolean.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import unicodedata

import cv2
import numpy as np

from .local_texture_jitter import local_texture_jitter
from .official_video_jitter import encode_lossless, native_video
from .static_jitter import ROOT, digest
from .validate_vjepa_motion import check_sha, cluster_interval
from .vjepa_motion_probe import fresh_output, gpu_precheck, read_rows, write_json, write_rows


def normalize_prompt(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split()).strip(' .')


def window_indices(count, fps):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid source FPS')
    interval = max(1, round(fps / 8))
    if abs(fps / interval - 8) > 1e-6:
        raise ValueError('source cannot produce an exact 8fps official sampling grid')
    grid = np.arange(0, count, interval)
    if len(grid) < 16:
        raise ValueError('source shorter than one 16-frame window')
    starts = [0, (len(grid) - 16) // 2, len(grid) - 16]
    if len(set(starts)) != 3:
        raise ValueError('three distinct fixed windows required')
    return {name: grid[start:start + 16].tolist()
            for name, start in zip(('start', 'middle', 'end'), starts)}


def source_manifest(directory, prompt_file, cohort, config):
    directory = Path(directory).resolve()
    with Path(prompt_file).open(newline='') as handle:
        metadata = list(csv.DictReader(handle))
    count = config['source_count_per_cohort']
    if len(metadata) != count or {int(r['index']) for r in metadata} != set(range(count)):
        raise ValueError('metadata must have all 128 distinct indices')
    expected = {f'video_{i:03d}.mp4' for i in range(count)}
    if {p.name for p in directory.glob('*.mp4')} != expected:
        raise ValueError('source directory must contain exactly the fixed 128 videos')
    rows = []
    for meta in sorted(metadata, key=lambda r: int(r['index'])):
        index = int(meta['index']); path = directory / f'video_{index:03d}.mp4'
        prompt = meta['prompt'].strip()
        if not prompt:
            raise ValueError('empty prompt')
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        windows = window_indices(n, fps)
        rows.append({'source_id': f'{cohort}:{index:03d}', 'cohort': cohort, 'index': index,
                     'prompt': prompt, 'prompt_id': normalize_prompt(prompt),
                     'source_video': str(path), 'source_sha256': digest(path),
                     'frames': n, 'fps': fps, 'height': height, 'width': width,
                     'duration_seconds': n / fps, 'windows': windows})
    if len({r['prompt_id'] for r in rows}) != count:
        raise ValueError('duplicated normalized prompts')
    return rows


def select(args, config):
    check_sha(args.probe_sources, config['probe_sources_sha256'])
    sources = source_manifest(args.video_dir, args.prompts, args.cohort, config)
    development = read_rows(args.probe_sources)
    roles = ('train', 'validation', 'calibration_reserved')
    overlap = {role: sorted({r['prompt_id'] for r in sources} &
                           {normalize_prompt(r['prompt_id']) for r in development if r['role'] == role})
               for role in roles}
    training_hashes = {r['video_sha256'] for p in (Path(args.probe_root) / 'features').glob('shard-*/features.jsonl')
                       for r in read_rows(p) if r.get('view') == 'base'}
    if len(training_hashes) != 270:
        raise ValueError('all 270 TRAIN/DEV source hashes required for media-overlap audit')
    byte_overlap = sorted({r['source_sha256'] for r in sources} & training_hashes)
    if any(overlap.values()) or byte_overlap:
        raise ValueError('training/development prompt or media overlap; do not silently replace samples')
    output = fresh_output(args.output)
    write_rows(output / 'sources.jsonl', sources)
    write_json(output / 'selection.json', {'status': 'frozen', 'cohort': args.cohort,
        'config_sha256': digest(Path(args.config)), 'sources_sha256': digest(output / 'sources.jsonl'),
        'source_directory': str(Path(args.video_dir).resolve()), 'prompts_csv': str(Path(args.prompts).resolve()),
        'prompts_sha256': digest(Path(args.prompts)), 'probe_sources_sha256': config['probe_sources_sha256'],
        'source_count': len(sources), 'windows': len(sources) * 3, 'expected_scoring_inputs': len(sources) * 12,
        'prompt_overlap': overlap, 'source_byte_overlap': byte_overlap,
        'model_scores_read': False, 'semantic_overlap_manually_audited': False,
        'source_generator_checkpoint': 'NOT VERIFIED: pre-existing user generation directory',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(json.dumps(json.loads((output / 'selection.json').read_text())), flush=True)


def selected_sources(args):
    directory = Path(args.selection)
    receipt = json.loads((directory / 'selection.json').read_text())
    check_sha(args.config, receipt['config_sha256'])
    check_sha(directory / 'sources.jsonl', receipt['sources_sha256'])
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard')
    return read_rows(directory / 'sources.jsonl')[args.shard::args.shards]


def read_windows(source):
    needed = {i for indices in source['windows'].values() for i in indices}
    capture = cv2.VideoCapture(source['source_video']); frames = {}; index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if abs(pts - index / source['fps']) > 1e-5:
                raise ValueError('nonuniform source timeline, no invented frame timestamps')
            if index in needed:
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    if index != source['frames'] or set(frames) != needed:
        raise ValueError('incomplete source decode')
    return {name: np.stack([frames[i] for i in indices]) for name, indices in source['windows'].items()}


def build(args, config):
    import imageio_ffmpeg
    start = time.monotonic(); cv2.setNumThreads(1)
    sources = selected_sources(args); output = fresh_output(args.output)
    videos = output / 'videos'; videos.mkdir()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    failures = 0
    with (output / 'inputs.jsonl').open('x') as handle:
        for source in sources:
            source_error = None
            try:
                check_sha(source['source_video'], source['source_sha256'])
                windows = read_windows(source)
            except Exception as exc:
                source_error = f'{type(exc).__name__}: {exc}'
            for name in config['sampling']['windows']:
                for view in config['views']:
                    uid = f"{source['source_id']}:{name}:{view}"
                    row = {**source, 'evaluation_id': uid, 'window': name, 'view': view,
                           'frame_indices': source['windows'][name], 'seed': 0, 'amplitude': 0}
                    try:
                        if source_error:
                            raise RuntimeError(source_error)
                        original = windows[name]
                        if view.startswith('jitter'):
                            seed = int(view[6:]); row.update(seed=seed, amplitude=8)
                            intended, info, _ = local_texture_jitter(original, 8, seed, config['intervention'])
                            row['intervention'] = info
                        else:
                            intended = original
                        path = videos / (uid.replace(':', '_') + '.mp4')
                        encode_lossless(path, intended, 8, ffmpeg)
                        decoded, pts, fps = native_video(path, 1e-6)
                        if fps != 8 or not np.array_equal(decoded, intended):
                            raise ValueError('encoded clip not pixel/time exact')
                        row.update(video=str(path), sha256=digest(path), decoded_shape=list(decoded.shape),
                                   decoded_pixels_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                                   source_window_pixels_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                                   pts=pts, clip_fps=fps, status='ok')
                    except Exception as exc:
                        failures += 1; row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    handle.write(json.dumps(row, allow_nan=False) + '\n'); handle.flush()
            print(json.dumps({'stage': 'build', 'shard': args.shard, 'source': source['source_id'], 'failed': failures}), flush=True)
    write_json(output / 'completion.json', {'status': 'finished' if not failures else 'failed',
        'sources': len(sources), 'inputs': len(sources) * 12, 'failed': failures,
        'config_sha256': digest(Path(args.config)), 'manifest_sha256': digest(output / 'inputs.jsonl'),
        'selection_sha256': digest(Path(args.selection) / 'selection.json'),
        'script_sha256': digest(Path(__file__)), 'jitter_sha256': digest(Path(__file__).with_name('local_texture_jitter.py')),
        'ffmpeg_sha256': digest(Path(ffmpeg)), 'wall_seconds': time.monotonic() - start})
    if failures:
        raise RuntimeError('construction failures retained; cohort incomplete')


def load_inputs(path, selection, config_path):
    receipt = json.loads((Path(selection) / 'selection.json').read_text())
    check_sha(config_path, receipt['config_sha256'])
    sources = read_rows(Path(selection) / 'sources.jsonl')
    check_sha(Path(selection) / 'sources.jsonl', receipt['sources_sha256'])
    rows = []
    for manifest in sorted(Path(path).glob('shard-*/inputs.jsonl')):
        c = json.loads((manifest.parent / 'completion.json').read_text())
        check_sha(manifest, c['manifest_sha256']); check_sha(config_path, c['config_sha256'])
        check_sha(Path(selection) / 'selection.json', c['selection_sha256'])
        if c['status'] != 'finished' or c['failed']:
            raise ValueError('incomplete construction')
        rows.extend(read_rows(manifest))
    expected = {f"{r['source_id']}:{w}:{v}" for r in sources for w in ('start', 'middle', 'end')
                for v in ('original', 'encoding_control', 'jitter1701', 'jitter2904')}
    if len(rows) != len(expected) or {r['evaluation_id'] for r in rows} != expected or any(r['status'] != 'ok' for r in rows):
        raise ValueError('input coverage failure')
    return sorted(rows, key=lambda r: r['evaluation_id'])


def rectangular_preprocess(frames, config, device):
    """Same arithmetic as FrozenVJEPA.preprocess; relax only the square guard."""
    import torch
    import torch.nn.functional as F
    c = config['input']
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != c['frames'] or frames.shape[-1] != 3:
        raise ValueError('all 16 RGB uint8 frames required')
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(device, torch.float32).permute(0, 3, 1, 2) / 255
    x = F.interpolate(x, size=(c['size'], c['size']), mode='bilinear', align_corners=False, antialias=True)
    mean, std = (torch.tensor(c[key], device=device)[None, :, None, None] for key in ('mean', 'std'))
    return ((x - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)


def score(args, config, *, supplied_rows=None):
    import io
    import torch
    from dynamic_degree.backends.vbench import OfficialDynamicEvaluator, official_result_payload
    from dynamic_degree.learned_probe import MotionProbe
    from vbench_audit_models.vjepa import FrozenVJEPA

    start = time.monotonic(); os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    gpu = gpu_precheck(); torch.set_num_threads(3); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    all_rows = load_inputs(args.construction, args.selection, args.config) if supplied_rows is None else supplied_rows
    sources = selected_sources(args); chosen = {r['source_id'] for r in sources}
    rows = [r for r in all_rows if r['source_id'] in chosen]
    probe = Path(args.probe_root); trained = Path(args.trained_root)
    pc_path = ROOT / 'configs/dynamic-static-jitter/vjepa-probe-v1.json'
    check_sha(pc_path, config['probe_config_sha256']); pc = json.loads(pc_path.read_text())
    head_path = trained / 'training/aligned.pt'; check_sha(head_path, config['head_sha256'])
    check_sha(ROOT / 'metrics/dynamic-degree/src/dynamic_degree/learned_probe.py', config['head_implementation_sha256'])
    check_sha(args.raft_weight, config['raft_sha256'])
    output = fresh_output(args.output)
    class ExternalEncoder(FrozenVJEPA):
        def preprocess(self, frames):
            return rectangular_preprocess(frames, self.config, self.device)
    encoder = ExternalEncoder(probe / 'assets/vjepa2', probe / 'assets/vjepa2_1_vitb_dist_vitG_384.pt', pc)
    head = MotionProbe().to('cuda:0')
    head.load_state_dict(torch.load(head_path, map_location='cuda:0', weights_only=True), strict=True)
    head.eval().requires_grad_(False)
    reference = next(r for r in read_rows(probe / 'features/shard-0/features.jsonl') if r['view'] == 'base')
    reference_path = Path(args.official_video_root) / reference['relative_video_path']
    check_sha(reference_path, reference['video_sha256'])
    frames, _, _ = native_video(reference_path, 1e-6)
    if not torch.equal(encoder.preprocess(frames), FrozenVJEPA.preprocess(encoder, frames)):
        raise ValueError('square preprocessing parity failure')
    feature = encoder.encode(frames); buffer = io.BytesIO(); np.save(buffer, feature, allow_pickle=False)
    if hashlib.sha256(buffer.getbuffer()).hexdigest() != reference['feature_sha256']:
        raise ValueError('encoder prior training feature mismatch')
    saved = next(r for r in read_rows(trained / 'training/scores.jsonl')
                 if r['video_uid'] == reference['video_uid'] and r['arm'] == 'aligned')
    with torch.inference_mode():
        error = abs(float(head(torch.from_numpy(feature).to('cuda:0', torch.float32)[None])[0]) - saved['latent'][0])
    if error > 1e-6:
        raise ValueError('frozen head loading parity failure')
    origin = OfficialDynamicEvaluator(torch.device('cuda:0'), Path(args.raft_weight), Path(args.upstream))
    if origin.upstream_state.sha != config['origin_upstream_commit'] or origin.upstream_state.dirty:
        raise ValueError('official source mismatch')
    write_json(output / 'provenance.json', {'config_sha256': digest(Path(args.config)),
        'script_sha256': digest(Path(__file__)), 'gpu': gpu, 'upstream': asdict(origin.upstream_state),
        'encoder': encoder.identity, 'rectangular_adapter': config['repair_preprocessing'],
        'head_sha256': config['head_sha256'], 'head_reload_error': error,
        'square_preprocessing_exact': True, 'training_feature_exact': True,
        'training_updates': 0, 'posthoc_mapping_changed': False,
        'expected_ids': [r['evaluation_id'] for r in rows], 'torch': torch.__version__,
        'opencv': cv2.__version__, 'numpy': np.__version__,
        'selection_sha256': digest(Path(args.selection) / 'selection.json')})
    failures = 0
    with (output / 'scores.jsonl').open('x') as handle:
        for i, row in enumerate(rows):
            result = {k: row[k] for k in ('evaluation_id', 'source_id', 'cohort', 'index', 'prompt_id', 'window', 'view', 'seed')}
            item_start = time.monotonic()
            try:
                path = Path(row['video']); check_sha(path, row['sha256'])
                frames, pts, fps = native_video(path, 1e-6)
                pixel_sha = hashlib.sha256(frames.tobytes()).hexdigest()
                if pixel_sha != row['decoded_pixels_sha256'] or fps != 8 or len(frames) != 16:
                    raise ValueError('input decode changed')
                features = encoder.encode(frames)
                with torch.inference_mode():
                    latent = float(head(torch.from_numpy(features).to('cuda:0', torch.float32)[None])[0])
                if not np.isfinite(latent):
                    raise ValueError('nonfinite Repair')
                measured = origin.evaluate_video(path)
                if i == 0:
                    if bool(origin.dynamic.infer(str(path))) != measured.official_video_boolean:
                        raise ValueError('official infer parity failure')
                    result['official_infer_parity'] = True
                result.update(status='ok', repair=float(1 / (1 + np.exp(-latent))), repair_latent=latent,
                              origin=float(measured.official_video_boolean), diagnostics=official_result_payload(measured),
                              input_sha256=row['sha256'], decoded_pixels_sha256=pixel_sha,
                              feature_pixels_sha256=hashlib.sha256(features.tobytes()).hexdigest())
            except Exception as exc:
                failures += 1; result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            result['wall_seconds'] = time.monotonic() - item_start
            handle.write(json.dumps(result, allow_nan=False) + '\n'); handle.flush()
            if (i + 1) % 12 == 0 or i + 1 == len(rows):
                print(json.dumps({'stage': 'score', 'shard': args.shard, 'completed': i + 1,
                                  'expected': len(rows), 'failed': failures, 'wall_seconds': time.monotonic() - start}), flush=True)
            if i == 0 and failures:
                raise RuntimeError(result['error'])
    write_json(output / 'completion.json', {'status': 'finished' if not failures else 'failed',
        'completed': len(rows), 'expected': len(rows), 'failed': failures,
        'config_sha256': digest(Path(args.config)), 'scores_sha256': digest(output / 'scores.jsonl'),
        'wall_seconds': time.monotonic() - start})
    if failures:
        raise RuntimeError('scoring failures retained')


def compute_statistics(pairs, config):
    prompts = [p['prompt_id'] for p in pairs]; a = config['analysis']; output = {}
    def ci(values):
        return cluster_interval(values, prompts, a['bootstrap_seed'], a['bootstrap_replicates'])
    for backend in ('origin', 'repair'):
        values = np.array([p[backend] for p in pairs])  # source x three windows x (base, cf1, cf2)
        base = values[:, :, 0].mean(axis=1); cf = values[:, :, 1:].mean(axis=(1, 2))
        difference = values[:, :, 1:] - values[:, :, :1]
        delta = cf - base; mae = abs(difference).mean(axis=(1, 2))
        output[backend] = {'base_mean': float(base.mean()), 'cf_mean': float(cf.mean()),
            'delta': float(delta.mean()), 'delta_ci95': ci(delta),
            'mae': float(mae.mean()), 'mae_ci95': ci(mae),
            'source_base_std': float(base.std()), 'source_base_range': [float(base.min()), float(base.max())],
            'window_seed_count': int(difference.size), 'drops_over_0_1': int((difference < -.1).sum()),
            'increases_over_0_1': int((difference > .1).sum()),
            'largest_drop': float(difference.min()), 'largest_increase': float(difference.max()),
            'base_window_zero_fraction': float((values[:, :, 0] == 0).mean()),
            'base_window_one_fraction': float((values[:, :, 0] == 1).mean()),
            'per_window': {name: {'base': float(values[:, j, 0].mean()),
                                  'cf': float(values[:, j, 1:].mean()),
                                  'delta': float(difference[:, j].mean()),
                                  'mae': float(abs(difference[:, j]).mean())}
                           for j, name in enumerate(('start', 'middle', 'end'))}}
    inflation = output['origin']['delta']
    output['relative_increase_check'] = {
        'applicable': inflation > 0,
        'pass': output['repair']['delta'] <= .1 * inflation if inflation > 0 else None,
        'note': 'descriptive relative-scale diagnostic; not motion-accuracy validation'}
    return output


def summarize(args, config):
    inputs = load_inputs(args.construction, args.selection, args.config); scores = []
    for path in sorted(Path(args.scores).glob('shard-*/scores.jsonl')):
        c = json.loads((path.parent / 'completion.json').read_text())
        check_sha(path, c['scores_sha256']); check_sha(args.config, c['config_sha256'])
        if c['status'] != 'finished' or c['failed']:
            raise ValueError('incomplete scoring; do not drop failed items')
        provenance = json.loads((path.parent / 'provenance.json').read_text())
        check_sha(Path(args.selection) / 'selection.json', provenance['selection_sha256'])
        if provenance['head_sha256'] != config['head_sha256'] or provenance['training_updates'] != 0:
            raise ValueError('model identity changed')
        scores.extend(read_rows(path))
    by_id = {r['evaluation_id']: r for r in scores}
    if len(by_id) != len(scores) or set(by_id) != {r['evaluation_id'] for r in inputs}:
        raise ValueError('scoring coverage mismatch')
    for row in inputs:
        score_row = by_id[row['evaluation_id']]
        if score_row['status'] != 'ok' or score_row['input_sha256'] != row['sha256']:
            raise ValueError('score/input identity mismatch')
    sources = read_rows(Path(args.selection) / 'sources.jsonl'); pairs = []; controls = 0
    for source in sources:
        p = {k: source[k] for k in ('source_id', 'cohort', 'index', 'prompt_id')}
        p.update(origin=[], repair=[])
        for window in ('start', 'middle', 'end'):
            view = {v: by_id[f"{source['source_id']}:{window}:{v}"] for v in config['views']}
            base, control = view['original'], view['encoding_control']
            if (base['decoded_pixels_sha256'] != control['decoded_pixels_sha256']
                    or base['origin'] != control['origin'] or base['repair'] != control['repair']):
                raise ValueError('encoding control mismatch')
            controls += 1
            for backend in ('origin', 'repair'):
                p[backend].append([view[v][backend] for v in ('original', 'jitter1701', 'jitter2904')])
        pairs.append(p)
    output = fresh_output(args.output)
    write_rows(output / 'pairs.jsonl', pairs)
    result = {'status': 'finished', 'cohort': sources[0]['cohort'], 'sources': len(sources),
              'prompts': len({r['prompt_id'] for r in sources}), 'windows': len(sources) * 3,
              'counterfactuals': len(sources) * 6, 'scored_inputs': len(scores), 'failures': 0,
              'encoding_controls_exact': controls, 'head_sha256': config['head_sha256'],
              'config_sha256': digest(Path(args.config)), 'pairs_sha256': digest(output / 'pairs.jsonl'),
              'statistics': compute_statistics(pairs, config), 'limitations': config['limitations'],
              'human_motion_accuracy': 'NOT RUN: no independent human motion labels for this cohort'}
    write_json(output / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'build', 'score', 'summarize'))
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    for flag in ('video-dir', 'prompts', 'cohort', 'probe-sources', 'probe-root', 'selection',
                 'construction', 'trained-root', 'official-video-root', 'raft-weight', 'upstream', 'scores'):
        parser.add_argument('--' + flag)
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-aligned-forcing128-generalization-v1':
        raise ValueError('unexpected protocol')
    globals()[args.stage](args, config)


if __name__ == '__main__':
    main()
