"""Real static/moving video pilot using official LASIESTA state annotations.

Select/build before scoring; preserve every mask decision and failed item. Uses
the existing frozen scoring implementation, not a newly fitted motion model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

import cv2
import numpy as np

from . import evaluate_dynamic_generalization as general
from .local_texture_jitter import local_texture_jitter
from .official_video_jitter import encode_lossless, native_video
from .static_jitter import digest
from .validate_vjepa_motion import check_sha
from .vjepa_motion_probe import fresh_output, read_rows, write_json, write_rows


def extract_archive(archive, output):
    """Read RAR through the system libarchive; reject links and escaping paths."""
    lib = ctypes.CDLL(ctypes.util.find_library('archive'))
    specs = {
        'archive_read_new': (ctypes.c_void_p, []),
        'archive_read_support_format_all': (ctypes.c_int, [ctypes.c_void_p]),
        'archive_read_support_filter_all': (ctypes.c_int, [ctypes.c_void_p]),
        'archive_read_open_filename': (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]),
        'archive_read_next_header': (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
        'archive_entry_pathname': (ctypes.c_char_p, [ctypes.c_void_p]),
        'archive_entry_filetype': (ctypes.c_uint, [ctypes.c_void_p]),
        'archive_entry_size': (ctypes.c_int64, [ctypes.c_void_p]),
        'archive_read_data': (ctypes.c_ssize_t, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]),
        'archive_error_string': (ctypes.c_char_p, [ctypes.c_void_p]),
        'archive_read_free': (ctypes.c_int, [ctypes.c_void_p]),
    }
    for name, (result, args) in specs.items():
        func = getattr(lib, name); func.restype = result; func.argtypes = args
    reader = lib.archive_read_new(); entry = ctypes.c_void_p()
    output.mkdir(parents=True, exist_ok=False)
    files = []; buffer = ctypes.create_string_buffer(1024 * 1024)
    def ensure(code):
        if code < 0:
            raise ValueError(lib.archive_error_string(reader).decode(errors='replace'))
    try:
        ensure(lib.archive_read_support_format_all(reader)); ensure(lib.archive_read_support_filter_all(reader))
        ensure(lib.archive_read_open_filename(reader, str(archive).encode(), len(buffer)))
        while True:
            code = lib.archive_read_next_header(reader, ctypes.byref(entry))
            if code == 1:
                break
            ensure(code)
            name = lib.archive_entry_pathname(entry).decode().replace('\\', '/')
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name:
                raise ValueError('unsafe archive path')
            target = output.joinpath(*relative.parts)
            kind = lib.archive_entry_filetype(entry)
            if kind == 0o040000:
                target.mkdir(parents=True, exist_ok=True); continue
            if kind != 0o100000 or target.suffix.lower() not in ('.bmp', '.png', '.xml', '.txt'):
                raise ValueError('unexpected archive member type')
            expected = lib.archive_entry_size(entry)
            if expected < 0 or expected > 10 * 1024 * 1024:
                raise ValueError('unexpected member size')
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(); size = 0
            with target.open('xb') as handle:
                while True:
                    n = lib.archive_read_data(reader, buffer, len(buffer)); ensure(n)
                    if n == 0:
                        break
                    block = buffer.raw[:n]; handle.write(block); h.update(block); size += n
                    if size > expected:
                        raise ValueError('member exceeds declared size')
            if size != expected:
                raise ValueError('truncated member')
            files.append({'path': str(relative), 'bytes': size, 'sha256': h.hexdigest()})
    finally:
        lib.archive_read_free(reader)
    return files


def mask_state(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('RGB uint8 annotation required')
    packed = rgb[..., 0].astype(np.uint32) * 65536 + rgb[..., 1].astype(np.uint32) * 256 + rgb[..., 2]
    colors, counts = np.unique(packed, return_counts=True)
    hist = {(int(c) // 65536, (int(c) // 256) % 256, int(c) % 256): int(n) for c, n in zip(colors, counts)}
    allowed = {(0,0,0), (255,0,0), (0,255,0), (255,255,0), (255,255,255), (128,128,128)}
    if set(hist) - allowed:
        raise ValueError(f'unknown annotation colors: {set(hist) - allowed}')
    moving = sum(hist.get(c, 0) for c in ((255,0,0), (0,255,0), (255,255,0)))
    stationary = hist.get((255,255,255), 0); unknown = hist.get((128,128,128), 0)
    state = 'motion' if moving else ('ambiguous' if unknown and not stationary else 'static')
    return {'state': state, 'moving_pixels': moving, 'stationary_pixels': stationary,
            'unknown_pixels': unknown, 'height': rgb.shape[0], 'width': rgb.shape[1]}


def state_windows(states, frames=16, stride=3, step=48):
    span = (frames - 1) * stride + 1
    if step < span:
        raise ValueError('windows must not overlap')
    runs = []; windows = []; start = 0
    while start < len(states):
        stop = start + 1
        while stop < len(states) and states[stop] == states[start]:
            stop += 1
        state = states[start]; chosen = []
        if state in ('motion', 'static'):
            chosen = list(range(start, stop - span + 1, step))
            windows.extend({'condition': state, 'start': s, 'stop': s + span,
                            'indices': list(range(s, s + span, stride))} for s in chosen)
        runs.append({'state': state, 'start': start, 'stop': stop, 'selected_starts': chosen,
                     'reason': 'ambiguous' if state == 'ambiguous' else 'shorter_than_span' if not chosen else 'selected_nonoverlapping'})
        start = stop
    return windows, runs


def indexed_files(directory, extension):
    out = {}
    for p in directory.glob('*' + extension):
        match = re.search(r'[-_](\d+)$', p.stem)
        if not match or int(match[1]) in out:
            raise ValueError('invalid or duplicate frame index')
        out[int(match[1])] = p
    if not out or sorted(out) != list(range(1, len(out) + 1)):
        raise ValueError('missing frame indices')
    return out


def prepare(args, config):
    start = time.monotonic(); out = fresh_output(args.output); cv2.setNumThreads(1)
    downloads = Path(args.download)
    receipt = json.loads((downloads / 'completion.json').read_text())
    if receipt['status'] != 'finished' or {r['sequence'] for r in receipt['sequences']} != set(config['sequences']):
        raise ValueError('complete predeclared archives required')
    by_name = {r['sequence']: r for r in receipt['sequences']}
    rows = []; labels = []; segments = []; archives = []
    for sequence in config['sequences']:
        archive = downloads / (sequence + '.rar'); check_sha(archive, by_name[sequence]['sha256'])
        directory = out / 'extracted' / sequence
        files = extract_archive(archive, directory)
        archives.append({'sequence': sequence, 'archive_sha256': digest(archive), 'files': files})
        images = indexed_files(directory / sequence, '.bmp')
        masks = indexed_files(directory / (sequence + '-GT'), '.png')
        if images.keys() != masks.keys():
            raise ValueError('RGB/GT frame coverage mismatch')
        lookup = {r['path']: r['sha256'] for r in files}
        sequence_labels = []
        for i in sorted(masks):
            bgr = cv2.imread(str(masks[i]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('unreadable annotation')
            record = {'sequence': sequence, 'frame': i, **mask_state(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))}
            sequence_labels.append(record); labels.append(record)
        windows, runs = state_windows([r['state'] for r in sequence_labels],
            config['sampling']['frames'], config['sampling']['source_frame_stride'], config['sampling']['window_step'])
        segments.extend({'sequence': sequence, **r} for r in runs)
        for w in windows:
            indices = [i + 1 for i in w['indices']]
            paths = [images[i] for i in indices]
            sampled = [sequence_labels[i - 1] for i in indices]
            sid = f"{sequence}:{indices[0]:04d}"
            rows.append({'source_id': sid, 'sequence': sequence, 'condition': w['condition'],
                'cohort': 'lasiesta', 'index': len(rows), 'prompt_id': sequence,
                'windows': {'clip': indices}, 'source_frames': [str(p) for p in paths],
                'source_frame_sha256': [lookup[str(p.relative_to(directory))] for p in paths],
                'height': sampled[0]['height'], 'width': sampled[0]['width'],
                'static_subtype': ('stationary_object' if any(r['stationary_pixels'] for r in sampled) else 'empty_background') if w['condition'] == 'static' else None,
                'gt_span': [w['start'] + 1, w['stop']], 'archive_sha256': digest(archive),
                'source_fps': None, 'analysis_fps': 8, 'frames': 16})
        print(json.dumps({'sequence': sequence, 'frames': len(images), 'selected': dict(Counter(w['condition'] for w in windows))}), flush=True)
    if set(r['condition'] for r in rows) != {'static', 'motion'}:
        raise ValueError('both real static and real motion groups required')
    write_rows(out / 'sources.jsonl', rows); write_rows(out / 'frame_labels.jsonl', labels)
    write_rows(out / 'segments.jsonl', segments); write_json(out / 'archive_files.json', archives)
    result = {'status': 'frozen', 'config_sha256': digest(Path(args.config)),
              'sources_sha256': digest(out / 'sources.jsonl'), 'labels_sha256': digest(out / 'frame_labels.jsonl'),
              'segments_sha256': digest(out / 'segments.jsonl'), 'archives_sha256': digest(out / 'archive_files.json'),
              'class_counts': dict(Counter(r['condition'] for r in rows)), 'sequences': len(archives),
              'expected_scoring_inputs': len(rows) * 4, 'scores_read': False, 'source_frame_timing_verified': False,
              'source_staticized': False, 'prepare_script_sha256': digest(Path(__file__)),
              'elapsed_seconds': time.monotonic() - start}
    write_json(out / 'selection.json', result); print(json.dumps(result), flush=True)


def build(args, config):
    import imageio_ffmpeg
    sources = general.selected_sources(args); out = fresh_output(args.output)
    (out / 'videos').mkdir(); ffmpeg = imageio_ffmpeg.get_ffmpeg_exe(); failures = 0
    with (out / 'inputs.jsonl').open('x') as handle:
        for source in sources:
            frames = []
            for path, expected in zip(source['source_frames'], source['source_frame_sha256']):
                check_sha(path, expected)
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError('invalid original frame')
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            original = np.stack(frames); source_hash = hashlib.sha256(original.tobytes()).hexdigest()
            for view in config['views']:
                uid = f"{source['source_id']}:clip:{view}"
                row = {**source, 'evaluation_id': uid, 'window': 'clip', 'view': view, 'seed': 0, 'amplitude': 0}
                try:
                    intended = original
                    if view.startswith('jitter'):
                        seed = int(view[6:]); intended, info, _ = local_texture_jitter(original, 8, seed, config['intervention'])
                        row.update(seed=seed, amplitude=8, intervention=info)
                    path = out / 'videos' / (uid.replace(':', '_') + '.mp4')
                    encode_lossless(path, intended, 8, ffmpeg)
                    actual, pts, fps = native_video(path, 1e-6)
                    if fps != 8 or not np.array_equal(actual, intended):
                        raise ValueError('lossless construction verification failed')
                    row.update(video=str(path), sha256=digest(path), status='ok',
                        decoded_pixels_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                        source_window_pixels_sha256=source_hash, decoded_shape=list(actual.shape), pts=pts)
                except Exception as exc:
                    failures += 1; row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                handle.write(json.dumps(row, allow_nan=False) + '\n'); handle.flush()
            print(json.dumps({'built': source['source_id'], 'failed': failures}), flush=True)
    write_json(out / 'completion.json', {'status': 'finished' if not failures else 'failed',
        'failed': failures, 'inputs': len(sources) * 4, 'manifest_sha256': digest(out / 'inputs.jsonl'),
        'config_sha256': digest(Path(args.config)), 'selection_sha256': digest(Path(args.selection) / 'selection.json'),
        'script_sha256': digest(Path(__file__)), 'ffmpeg_sha256': digest(Path(ffmpeg))})
    if failures:
        raise RuntimeError('construction incomplete; preserve all outputs')


def load_inputs(args, config):
    receipt = json.loads((Path(args.selection) / 'selection.json').read_text())
    check_sha(args.config, receipt['config_sha256']); check_sha(Path(args.selection) / 'sources.jsonl', receipt['sources_sha256'])
    sources = read_rows(Path(args.selection) / 'sources.jsonl'); rows = []
    for manifest in sorted(Path(args.construction).glob('shard-*/inputs.jsonl')):
        complete = json.loads((manifest.parent / 'completion.json').read_text())
        check_sha(manifest, complete['manifest_sha256']); check_sha(args.config, complete['config_sha256'])
        check_sha(Path(args.selection) / 'selection.json', complete['selection_sha256'])
        if complete['status'] != 'finished' or complete['failed']:
            raise ValueError('incomplete construction')
        rows.extend(read_rows(manifest))
    expected = {f"{r['source_id']}:clip:{v}" for r in sources for v in config['views']}
    if len(rows) != len(expected) or {r['evaluation_id'] for r in rows} != expected or any(r['status'] != 'ok' for r in rows):
        raise ValueError('input coverage mismatch')
    return sorted(rows, key=lambda r: r['evaluation_id'])


def score(args, config):
    general.score(args, config, supplied_rows=load_inputs(args, config))


def auc(static, motion):
    a = np.asarray(static, float); b = np.asarray(motion, float)
    if not len(a) or not len(b):
        return None
    difference = b[:, None] - a[None, :]
    return float(np.mean((difference > 0).astype(float) + .5 * (difference == 0)))


def summarize(args, config):
    inputs = load_inputs(args, config); scores = []
    for path in sorted(Path(args.scores).glob('shard-*/scores.jsonl')):
        complete = json.loads((path.parent / 'completion.json').read_text())
        check_sha(path, complete['scores_sha256']); check_sha(args.config, complete['config_sha256'])
        if complete['status'] != 'finished' or complete['failed']:
            raise ValueError('incomplete scoring')
        scores.extend(read_rows(path))
    if len(scores) != len(inputs) or {r['evaluation_id'] for r in scores} != {r['evaluation_id'] for r in inputs}:
        raise ValueError('score coverage mismatch')
    by_id = {r['evaluation_id']: r for r in scores}; by_source = defaultdict(dict)
    for row in inputs:
        s = by_id[row['evaluation_id']]
        if s['status'] != 'ok' or s['input_sha256'] != row['sha256']:
            raise ValueError('failed or changed input')
        for backend in ('origin', 'repair'):
            if not np.isfinite(s[backend]) or not 0 <= s[backend] <= 1:
                raise ValueError('invalid score')
        by_source[row['source_id']][row['view']] = s
    sources = read_rows(Path(args.selection) / 'sources.jsonl'); pairs = []
    for source in sources:
        views = by_source[source['source_id']]; a = views['original']; c = views['encoding_control']
        if any(a[k] != c[k] for k in ('decoded_pixels_sha256', 'feature_pixels_sha256', 'origin', 'repair')):
            raise ValueError('encoding control mismatch')
        pair = {k: source[k] for k in ('source_id', 'sequence', 'condition', 'static_subtype')}
        for backend in ('origin', 'repair'):
            pair[backend] = [views[v][backend] for v in ('original', 'jitter1701', 'jitter2904')]
        pairs.append(pair)
    result = {'status': 'finished', 'config_sha256': digest(Path(args.config)),
              'source_counts': dict(Counter(r['condition'] for r in pairs)),
              'source_sequences': sorted({r['sequence'] for r in pairs}), 'inputs_per_backend': len(inputs),
              'failures': 0, 'encoding_controls_exact': len(pairs), 'classes': {}, 'discrimination': {}}
    rng = np.random.default_rng(config['analysis']['bootstrap_seed']); seqs = sorted({r['sequence'] for r in pairs})
    draws = rng.integers(0, len(seqs), (config['analysis']['bootstrap_replicates'], len(seqs)))
    for condition in ('static', 'motion'):
        chosen = [r for r in pairs if r['condition'] == condition]; result['classes'][condition] = {}
        for backend in ('origin', 'repair'):
            values = np.array([r[backend] for r in chosen]); base = values[:, 0]; cf = values[:, 1:].mean(1)
            delta = cf - base; errors = abs(values[:, 1:] - base[:, None]).mean(1)
            summary = {'n': len(chosen), 'base': float(base.mean()), 'cf': float(cf.mean()),
                       'delta': float(delta.mean()), 'mae': float(errors.mean()),
                       'max_abs_change': float(abs(values[:, 1:] - base[:, None]).max())}
            summary['by_sequence'] = {}
            for seq in seqs:
                ix = [i for i, r in enumerate(chosen) if r['sequence'] == seq]
                if ix:
                    summary['by_sequence'][seq] = {'n': len(ix), 'base': float(base[ix].mean()),
                        'cf': float(cf[ix].mean()), 'delta': float(delta[ix].mean()), 'mae': float(errors[ix].mean())}
            summary['sequence_equal_means'] = {key: float(np.mean([r[key] for r in summary['by_sequence'].values()]))
                                               for key in ('base', 'cf', 'delta', 'mae')}
            counts = np.array([sum(r['sequence'] == seq for r in chosen) for seq in seqs])
            denom = counts[draws].sum(1); valid = denom > 0
            for name, vector in [('delta', delta), ('mae', errors)]:
                totals = np.array([sum(vector[i] for i, r in enumerate(chosen) if r['sequence'] == seq) for seq in seqs])
                boot = totals[draws].sum(1)[valid] / denom[valid]
                summary[name + '_ci95'] = np.quantile(boot, [.025, .975]).tolist()
            summary['bootstrap_valid_draws'] = int(valid.sum())
            result['classes'][condition][backend] = summary
    for backend in ('origin', 'repair'):
        result['discrimination'][backend] = {}
        for j, name in enumerate(('base', 'jitter1701', 'jitter2904')):
            result['discrimination'][backend][name + '_auroc'] = auc(
                [r[backend][j] for r in pairs if r['condition'] == 'static'],
                [r[backend][j] for r in pairs if r['condition'] == 'motion'])
        matched = []
        for seq in seqs:
            s = [r[backend][0] for r in pairs if r['sequence'] == seq and r['condition'] == 'static']
            m = [r[backend][0] for r in pairs if r['sequence'] == seq and r['condition'] == 'motion']
            if s and m:
                matched.append({'sequence': seq, 'base_auroc': auc(s, m), 'mean_motion_minus_static': float(np.mean(m)-np.mean(s))})
        result['discrimination'][backend]['within_sequence'] = matched
    out = fresh_output(args.output); write_rows(out / 'pairs.jsonl', pairs)
    result['pairs_sha256'] = digest(out / 'pairs.jsonl'); write_json(out / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'build', 'score', 'summarize'))
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    for name in ('download', 'selection', 'construction', 'scores', 'probe-root', 'trained-root', 'official-video-root', 'raft-weight', 'upstream'):
        p.add_argument('--' + name)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=2)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-aligned-lasiesta-real-static-motion-v1':
        raise ValueError('unexpected protocol')
    globals()[args.stage](args, config)


if __name__ == '__main__':
    main()
