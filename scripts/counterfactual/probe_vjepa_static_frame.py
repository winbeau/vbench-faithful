"""User-requested still-frame / slight-motion diagnostic, never formal CF data.

Select the first UID among existing DEV32 originals, repeat its first frame,
then compare fixed 8/32 native-pixel global pans and one 8px local-jitter view.
The frozen head and sigmoid are not adjusted to these controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import validate_vjepa_motion as frozen
from .local_texture_jitter import local_texture_jitter
from .official_video_jitter import encode_lossless, native_video
from .static_jitter import digest
from .vjepa_motion_probe import fresh_output, read_rows, write_json, write_rows


PARENT_SHA = 'aef55f4fbb6d31a334de014b2528dc01e1a373a315006bd559f6c2c9dbd38c09'
INPUT_SHA = 'fecc98496ab1617ed98fbfeb4a9c21e5802d7b326a95fbcc0f76796f256ce5ff'
SCORER_SHA = 'b2619965627e1ea554b11fe5acbd0e8713cc893ef5ac12019d9cf955b638b317'


def global_pan(frame, count, total_pixels):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or count < 2 or total_pixels < 0:
        raise ValueError('uint8 RGB frame, at least two frames and nonnegative displacement required')
    height, width = frame.shape[:2]
    return np.stack([cv2.warpAffine(frame, np.array([[1, 0, dx], [0, 1, 0]], np.float32),
                    (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                     for dx in np.linspace(0, total_pixels, count)])


def build(args, config):
    frozen.check_sha(args.previous_inputs, INPUT_SHA)
    source = min((r for r in read_rows(args.previous_inputs) if r['cohort'] == 'dev32' and r['family'] == 'original'),
                 key=lambda r: r['base_id'])
    video = Path(source['video']); frozen.check_sha(video, source['sha256'])
    frames, pts, fps = native_video(video, 1e-6)
    if len(frames) != 16 or abs(fps - 8) > 1e-6:
        raise ValueError('unexpected source cadence')
    still = np.repeat(frames[:1], 16, axis=0)
    jitter, info, _ = local_texture_jitter(still, 8, 1701, config['construction'])
    variants = [('still', still), ('pan8', global_pan(frames[0], 16, 8)),
                ('pan32', global_pan(frames[0], 16, 32)), ('local_jitter8', jitter), ('original', frames)]
    out = fresh_output(args.output); (out / 'videos').mkdir()
    rows = []
    for family, array in variants:
        path = video if family == 'original' else out / 'videos' / f'{family}.mp4'
        if family != 'original':
            encode_lossless(path, array, fps, args.ffmpeg)
        decoded, decoded_pts, decoded_fps = native_video(path, 1e-6)
        if not np.array_equal(decoded, array) or not np.allclose(decoded_pts, pts, atol=1e-6, rtol=0) or decoded_fps != fps:
            raise ValueError('lossless pixels or timestamps changed')
        row = dict(source, evaluation_id=family, candidate_id=family, family=family,
                   cohort='single_frame_diagnostic', counterfactual_primary=False, seed=1701 if family == 'local_jitter8' else 0,
                   video=str(path), sha256=digest(path), decoded_shape=list(decoded.shape),
                   decoded_pixels_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                   pts=decoded_pts, fps=decoded_fps, pixel_exact_to_intended=True, native_timeline_preserved=True,
                   status='diagnostic_not_formal_counterfactual', reason=None,
                   mean_interframe_absolute_rgb_difference=float(np.abs(np.diff(decoded.astype(float), axis=0)).mean()))
        rows.append(row)
    if not np.array_equal(still, np.repeat(still[:1], len(still), axis=0)):
        raise AssertionError('still control is not static')
    if not cv2.imwrite(str(out / 'source_frame.png'), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)):
        raise RuntimeError('failed to save source frame')
    write_rows(out / 'inputs.jsonl', rows)
    write_json(out / 'completion.json', {
        'status': 'finished', 'inputs': len(rows), 'manifest_sha256': digest(out / 'inputs.jsonl'),
        'config_sha256': digest(Path(args.config)), 'driver_sha256': digest(Path(__file__)),
        'source_uid': source['base_id'], 'source_sha256': source['sha256'], 'source_frame_index': 0,
        'selection': 'lexicographically first DEV32 original UID, not selected by score',
        'native_shape': list(frames.shape), 'fps': fps, 'duration_seconds': len(frames) / fps,
        'global_pan': {'first_to_last_native_pixels': [8, 32], 'direction': 'right',
                       'border': 'BORDER_REFLECT_101; synthetic camera-like translation, not articulated motion'},
        'jitter_geometry': info, 'source_frame_sha256': digest(out / 'source_frame.png'),
        'ffmpeg_sha256': digest(Path(args.ffmpeg)), 'training_updates': 0, 'mapping_changes': False,
        'scope': 'one-source diagnostic only, excluded from official450 and not an absolute calibration fit',
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((out / 'completion.json').read_text(), flush=True)


def score(args, config):
    receipt = json.loads((Path(args.inputs).parent / 'completion.json').read_text())
    frozen.check_sha(Path(__file__), receipt['driver_sha256'])
    if receipt['inputs'] != 5:
        raise ValueError('all five diagnostic inputs required')
    args.shard = 0; args.shards = 1
    frozen.score(args, config)
    write_json(Path(args.output) / 'diagnostic_driver.json', {
        'driver_sha256': digest(Path(__file__)), 'scorer_sha256': digest(Path(frozen.__file__)),
        'scope': receipt['scope'], 'training_updates': 0, 'mapping_changes': False})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('build'); s.add_argument('--previous-inputs', required=True); s.add_argument('--ffmpeg', required=True)
    s = sub.add_parser('score')
    for name in ('inputs', 'probe-root', 'video-root', 'raft-weight', 'upstream'):
        s.add_argument('--' + name, required=True)
    s.add_argument('--backend', choices=['vjepa', 'origin'], default='vjepa')
    for s in sub.choices.values():
        s.add_argument('--output', required=True)
    args = p.parse_args()
    frozen.check_sha(args.config, PARENT_SHA); frozen.check_sha(Path(frozen.__file__), SCORER_SHA)
    config = json.loads(Path(args.config).read_text())
    globals()[args.command](args, config)


if __name__ == '__main__':
    main()
