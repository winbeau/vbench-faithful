"""Frozen BMC real-video extension: timing-aware sampling, review, existing scorer.

Review and all candidates are fixed before any scoring. Ambiguous labels remain
in the primary robustness analysis. No segmentation mask is relabeled as motion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import evaluate_dynamic_generalization as general
from . import evaluate_dynamic_lasiesta as lasiesta
from .static_jitter import digest as _digest
from .validate_vjepa_motion import check_sha
from .vjepa_motion_probe import fresh_output, read_rows, write_json, write_rows


def digest(path):
    return _digest(Path(path))


def sampling_windows(count, fps, maximum=12):
    if count < 1 or not np.isfinite(fps) or fps < 8:
        raise ValueError('at least 8 real frames/s required, no frame duplication')
    duration = count / fps; n = min(maximum, int(np.floor(duration / 2)))
    if not n:
        raise ValueError('recording shorter than two seconds')
    starts = np.linspace(0, duration - 2, n) if n > 1 else np.array([0.])
    windows = []
    for i, start in enumerate(starts):
        targets = start + np.arange(16) / 8
        indices = np.floor(targets * fps + .5).astype(int)
        if len(set(indices)) != 16 or indices[-1] >= count:
            raise ValueError('invalid sampled indices')
        if windows and indices[0] <= windows[-1]['indices'][-1]:
            raise ValueError('overlapping source windows')
        windows.append({'ordinal': i, 'start_seconds': float(start), 'target_seconds': targets.tolist(),
                        'indices': indices.tolist(), 'source_seconds': (indices / fps).tolist(),
                        'max_rounding_seconds': float(abs(indices / fps - targets).max())})
    return windows


def safe_zip(archive, output):
    output.mkdir(parents=True, exist_ok=False); rows = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            relative = PurePosixPath(info.filename.replace('\\', '/'))
            mode = info.external_attr >> 16
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or stat.S_ISLNK(mode):
                raise ValueError('unsafe archive path or link')
            target = output.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True); continue
            if info.file_size > 2 * 1024 ** 3:
                raise ValueError('archive member exceeds task bound')
            target.parent.mkdir(parents=True, exist_ok=True); h = hashlib.sha256(); size = 0
            with z.open(info) as source, target.open('xb') as out:
                while block := source.read(1024 * 1024):
                    out.write(block); h.update(block); size += len(block)
            if size != info.file_size:
                raise ValueError('truncated ZIP member')
            rows.append({'path': str(relative), 'bytes': size, 'sha256': h.hexdigest()})
    return rows


def contact_sheet(rows, destination, all_frames=False):
    chosen = list(range(16)) if all_frames else [0, 5, 10, 15]
    thumb_w, thumb_h = (176, 144) if all_frames else (264, 198)
    columns = 4; per_row = (len(chosen) + columns - 1) // columns
    band = 24 + per_row * thumb_h
    canvas = Image.new('RGB', (columns * thumb_w, band * len(rows)), 'white'); draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        y = r * band
        draw.text((4, y + 5), f"{row['source_id']}  t={row['start_seconds']:.3f}s", fill='black')
        for j, idx in enumerate(chosen):
            with Image.open(row['source_frames'][idx]) as frame:
                frame.thumbnail((thumb_w, thumb_h))
                canvas.paste(frame, ((j % columns) * thumb_w, y + 24 + (j // columns) * thumb_h))
    canvas.save(destination)


def ffmpeg_timeline(video, log_path, fps, tolerance):
    """Use FFmpeg best-effort presentation times, not OpenCV's broken AVI tail PTS.

    The AVI header may count buffered packets instead of decoded frames. Never
    infer the final frame count from that header or duplicate missing frames.
    """
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, '-hide_banner', '-nostdin', '-threads', '1', '-i', str(video),
               '-an', '-vf', 'showinfo', '-f', 'null', '-']
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    count = 0; time_base = None; origin = None; max_error = 0.
    try:
        with Path(log_path).open('x') as log:
            for line in proc.stderr:
                log.write(line)
                m = re.search(r'config in time_base: (\d+)/(\d+)', line)
                if m:
                    time_base = int(m[1]) / int(m[2])
                m = re.search(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:', line)
                if m:
                    if time_base is None or int(m[1]) != count:
                        raise ValueError('invalid FFmpeg presentation-frame sequence')
                    pts = int(m[2]) * time_base
                    if origin is None:
                        origin = pts
                    error = abs(pts - origin - count / fps); max_error = max(max_error, error)
                    if error > tolerance:
                        raise ValueError('non-CFR best-effort presentation times')
                    count += 1
        if proc.wait() != 0 or not count:
            raise ValueError('FFmpeg timeline decode failed')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.communicate()
    return {'decoded_frames': count, 'pts_origin_seconds': origin, 'max_pts_error_seconds': max_error,
            'ffmpeg_sha256': digest(ffmpeg), 'timeline_log_sha256': digest(log_path),
            'timeline_source': 'FFmpeg decoded best-effort presentation timestamps; nominal FPS from AVI metadata'}


def prepare(args, config):
    out = fresh_output(args.output); cv2.setNumThreads(1); started = time.monotonic()
    downloads = Path(args.download); receipt = json.loads((downloads / 'completion.json').read_text())
    if receipt['status'] != 'finished' or {r['sequence'] for r in receipt['sequences']} != set(config['sequences']):
        raise ValueError('all nine complete author archives required')
    lookup = {r['sequence']: r for r in receipt['sequences']}; rows = []; archives = []; originals = []
    (out / 'review').mkdir(); (out / 'frames').mkdir(); (out / 'timelines').mkdir()
    for sequence in config['sequences']:
        archive = downloads / (sequence + '.zip'); check_sha(archive, lookup[sequence]['sha256'])
        directory = out / 'extracted' / sequence; files = safe_zip(archive, directory)
        archives.append({'sequence': sequence, 'archive_sha256': digest(archive), 'files': files})
        videos = list(directory.rglob('*.avi')) + list(directory.rglob('*.mp4'))
        if len(videos) != 1:
            raise ValueError(f'exactly one original recording required: {sequence}: {videos}')
        video = videos[0]; cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS); header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps < 8:
            cap.release()
            meta = {'sequence': sequence, 'source_video': str(video), 'source_sha256': digest(video),
                    'source_fps': fps, 'header_frame_count': header_count, 'status': 'NOT SCORED',
                    'reason': 'source FPS below 8; cannot supply distinct frames at the frozen 8fps cadence'}
            originals.append(meta); print(json.dumps(meta), flush=True); continue
        try:
            timeline = ffmpeg_timeline(video, out / 'timelines' / (sequence + '.log'), fps, config['sampling']['timestamp_tolerance'])
        except ValueError as exc:
            cap.release()
            if str(exc) != 'non-CFR best-effort presentation times':
                raise
            meta = {'sequence': sequence, 'source_video': str(video), 'source_sha256': digest(video),
                    'source_fps': fps, 'header_frame_count': header_count, 'status': 'NOT SCORED',
                    'reason': str(exc), 'partial_timeline_sha256': digest(out / 'timelines' / (sequence + '.log'))}
            originals.append(meta); print(json.dumps(meta), flush=True); continue
        count = timeline['decoded_frames']
        windows = sampling_windows(count, fps, config['sampling']['max_windows_per_sequence'])
        needed = {idx for w in windows for idx in w['indices']}; frames = {}; index = 0; max_error = 0.; pts_origin = None
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if pts_origin is None:
                    pts_origin = pts
                error = abs(pts - pts_origin - index / fps); max_error = max(error, max_error)
                if index in needed:
                    frame = out / 'frames' / f'{sequence}_{index:07d}.png'
                    if not cv2.imwrite(str(frame), bgr):
                        raise ValueError('failed lossless sampled-frame output')
                    frames[index] = (str(frame), digest(frame), bgr.shape[:2])
                index += 1
        finally:
            cap.release()
        if index != count or set(frames) != needed:
            raise ValueError('incomplete original decode')
        meta = {'sequence': sequence, 'source_video': str(video), 'source_sha256': digest(video),
                'source_fps': fps, 'source_frames': count, 'duration_seconds': count / fps, 'status': 'eligible',
                'opencv_pts_max_error_seconds': max_error, 'header_frame_count': header_count,
                'all_original_frames_timing_checked': True, **timeline}
        originals.append(meta); sequence_rows = []
        for w in windows:
            sampled = [frames[i] for i in w['indices']]; sid = f"{sequence}:{w['ordinal']:02d}"
            row = {'source_id': sid, 'sequence': sequence, 'condition': 'unreviewed', 'static_subtype': None,
                   'cohort': 'bmc', 'index': len(rows), 'prompt_id': sequence, 'windows': {'clip': w['indices']},
                   'source_frames': [r[0] for r in sampled], 'source_frame_sha256': [r[1] for r in sampled],
                   'height': sampled[0][2][0], 'width': sampled[0][2][1], 'source_fps': fps,
                   'analysis_fps': 8, 'frames': 16, 'source_video': str(video), 'source_sha256': meta['source_sha256'], **w}
            rows.append(row); sequence_rows.append(row)
        contact_sheet(sequence_rows, out / 'review' / (sequence + '.jpg'))
        # Full sampled-frame sheets permit stillness review without using scores.
        for start in range(0, len(sequence_rows), 3):
            contact_sheet(sequence_rows[start:start + 3], out / 'review' / f'{sequence}-full-{start//3}.jpg', True)
        print(json.dumps({'sequence': sequence, 'fps': fps, 'original_frames': count, 'windows': len(windows)}), flush=True)
    write_rows(out / 'candidates.jsonl', rows); write_json(out / 'originals.json', originals)
    write_json(out / 'archive_files.json', archives)
    write_json(out / 'preparation.json', {'status': 'awaiting_score_blind_review', 'scores_read': False,
        'config_sha256': digest(args.config), 'candidates_sha256': digest(out / 'candidates.jsonl'),
        'archives_sha256': digest(out / 'archive_files.json'), 'originals_sha256': digest(out / 'originals.json'),
        'original_recordings': len(originals), 'eligible_recordings': sum(r['status'] == 'eligible' for r in originals),
        'clips': len(rows), 'script_sha256': digest(Path(__file__)),
        'elapsed_seconds': time.monotonic() - started})


def freeze(args, config):
    directory = Path(args.selection); prepared = json.loads((directory / 'preparation.json').read_text())
    check_sha(args.config, prepared['config_sha256']); check_sha(directory / 'candidates.jsonl', prepared['candidates_sha256'])
    review = json.loads(Path(args.review).read_text())
    if review['scores_read'] or review['official_ground_truth']:
        raise ValueError('review must precede scoring and must not impersonate official labels')
    labels = review['clips']; candidates = read_rows(directory / 'candidates.jsonl')
    if set(labels) != {r['source_id'] for r in candidates}:
        raise ValueError('review must retain every candidate exactly once')
    rows = []
    for r in candidates:
        label = labels[r['source_id']]
        if label['condition'] not in config['labels']['conditions'] or not label.get('reason'):
            raise ValueError('invalid review label')
        rows.append({**r, 'condition': label['condition'], 'review_reason': label['reason'],
                     'static_subtype': 'visually_quiescent' if label['condition'] == 'static' else None})
    write_rows(directory / 'sources.jsonl', rows)
    write_json(directory / 'selection.json', {'status': 'frozen', 'config_sha256': digest(args.config),
        'sources_sha256': digest(directory / 'sources.jsonl'), 'review_sha256': digest(args.review),
        'preparation_sha256': digest(directory / 'preparation.json'), 'class_counts': dict(Counter(r['condition'] for r in rows)),
        'candidate_recordings': len(config['sequences']), 'source_sequences': len({r['sequence'] for r in rows}),
        'clips': len(rows), 'expected_scoring_inputs': len(rows) * 4,
        'scores_read': False, 'source_staticized': False, 'official_motion_ground_truth': False,
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})


def group_statistics(pairs, config):
    sequences = sorted({p['sequence'] for p in pairs}); a = config['analysis']
    rng = np.random.default_rng(a['bootstrap_seed'])
    draws = rng.integers(0, len(sequences), (a['bootstrap_replicates'], len(sequences)))
    output = {}
    for group in ('all', 'static', 'motion', 'ambiguous'):
        chosen = [p for p in pairs if group == 'all' or p['condition'] == group]
        if not chosen:
            output[group] = {'n': 0, 'status': 'NOT AVAILABLE'}; continue
        counts = np.array([sum(p['sequence'] == s for p in chosen) for s in sequences])
        denom = counts[draws].sum(1); valid = denom > 0; output[group] = {'n': len(chosen)}
        def ci(vector):
            sums = np.array([sum(v for v, p in zip(vector, chosen) if p['sequence'] == s) for s in sequences])
            return np.quantile(sums[draws].sum(1)[valid] / denom[valid], [.025, .975]).tolist()
        deltas = {}
        for backend in ('origin', 'repair'):
            values = np.array([p[backend] for p in chosen]); base = values[:, 0]; cf = values[:, 1:].mean(1)
            changes = values[:, 1:] - base[:, None]; delta = cf - base; mae = abs(changes).mean(1); deltas[backend] = delta
            output[group][backend] = {'base': float(base.mean()), 'cf': float(cf.mean()), 'delta': float(delta.mean()),
                'mae': float(mae.mean()), 'delta_ci95': ci(delta), 'mae_ci95': ci(mae),
                'max_increase': float(changes.max()), 'max_drop': float(changes.min()),
                'cf_increases_gt_0_1': int((changes > .1).sum()), 'cf_drops_lt_minus_0_1': int((changes < -.1).sum()),
                'cf_count': int(changes.size), 'by_sequence': {}}
            for sequence in sequences:
                ix = [i for i, p in enumerate(chosen) if p['sequence'] == sequence]
                if ix:
                    output[group][backend]['by_sequence'][sequence] = {'n': len(ix), 'base': float(base[ix].mean()),
                        'cf': float(cf[ix].mean()), 'delta': float(delta[ix].mean()), 'mae': float(mae[ix].mean())}
            seq = output[group][backend]['by_sequence']
            output[group][backend]['equal_sequence_mean'] = {k: float(np.mean([r[k] for r in seq.values()]))
                                                           for k in ('base', 'cf', 'delta', 'mae')}
        margin = deltas['repair'] - .1 * deltas['origin']
        output[group]['ten_percent_margin'] = {'mean': float(margin.mean()), 'ci95': ci(margin),
            'meaning': 'Repair delta minus 0.1 * Origin delta; negative favors nuisance suppression'}
        output[group]['bootstrap_valid_draws'] = int(valid.sum())
    return output


def summarize(args, config):
    inputs = lasiesta.load_inputs(args, config); scores = []; sources = general.read_rows(Path(args.selection) / 'sources.jsonl')
    for file in sorted(Path(args.scores).glob('shard-*/scores.jsonl')):
        done = json.loads((file.parent / 'completion.json').read_text()); check_sha(file, done['scores_sha256'])
        check_sha(args.config, done['config_sha256']); prov = json.loads((file.parent / 'provenance.json').read_text())
        if done['status'] != 'finished' or done['failed'] or prov['training_updates'] or prov['head_sha256'] != config['head_sha256']:
            raise ValueError('incomplete or changed frozen scorer')
        scores.extend(read_rows(file))
    by_id = {r['evaluation_id']: r for r in scores}
    if len(by_id) != len(scores) or set(by_id) != {r['evaluation_id'] for r in inputs}:
        raise ValueError('score membership mismatch')
    grouped = defaultdict(dict)
    for row in inputs:
        s = by_id[row['evaluation_id']]
        if s['status'] != 'ok' or s['input_sha256'] != row['sha256']:
            raise ValueError('failed score or changed source')
        if not all(np.isfinite(s[k]) and 0 <= s[k] <= 1 for k in ('origin', 'repair')):
            raise ValueError('invalid score')
        grouped[row['source_id']][row['view']] = s
    pairs = []
    for source in sources:
        v = grouped[source['source_id']]; original = v['original']; control = v['encoding_control']
        if any(original[k] != control[k] for k in ('decoded_pixels_sha256', 'feature_pixels_sha256', 'origin', 'repair')):
            raise ValueError('encoding control mismatch')
        p = {k: source[k] for k in ('source_id', 'sequence', 'condition', 'static_subtype', 'review_reason')}
        for backend in ('origin', 'repair'):
            p[backend] = [v[name][backend] for name in ('original', 'jitter1701', 'jitter2904')]
        pairs.append(p)
    result = {'status': 'finished', 'config_sha256': digest(args.config), 'source_clips': len(pairs),
        'source_recordings': len({r['sequence'] for r in pairs}), 'inputs_per_backend': len(inputs),
        'failures': 0, 'encoding_controls_exact': len(pairs), 'training_updates': 0,
        'selection_sha256': digest(Path(args.selection) / 'selection.json'), 'statistics': group_statistics(pairs, config),
        'discrimination': {}, 'limitations': config['limitations']}
    for backend in ('origin', 'repair'):
        result['discrimination'][backend] = {name + '_auroc': lasiesta.auc(
            [p[backend][j] for p in pairs if p['condition'] == 'static'],
            [p[backend][j] for p in pairs if p['condition'] == 'motion'])
            for j, name in enumerate(('base', 'jitter1701', 'jitter2904'))}
    out = fresh_output(args.output); write_rows(out / 'pairs.jsonl', pairs)
    result['pairs_sha256'] = digest(out / 'pairs.jsonl'); write_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'freeze', 'build', 'score', 'summarize'))
    p.add_argument('--config', required=True); p.add_argument('--output')
    for name in ('download', 'selection', 'construction', 'scores', 'review', 'probe-root', 'trained-root', 'official-video-root', 'raft-weight', 'upstream'):
        p.add_argument('--' + name)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=2)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['protocol'] != 'dynamic-aligned-bmc-real-motion-v1':
        raise ValueError('unexpected protocol')
    functions = {'prepare': prepare, 'freeze': freeze, 'build': lasiesta.build, 'score': lasiesta.score, 'summarize': summarize}
    functions[args.stage](args, config)


if __name__ == '__main__':
    main()
