"""Render every frame of preselected natural DEV originals, never scores.

The contact sheets aid motion-mechanism inspection. They neither certify human
annotations nor qualify new counterfactuals. Reserved test videos are not opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .official_video_jitter import native_video
from .static_jitter import digest


def select_development(rows):
    selected = [row for row in rows if row["split"] == "dev"]
    if not selected or len({r["video_uid"] for r in selected}) != len(selected):
        raise ValueError("unique nonempty predeclared development cohort required")
    for row in selected:
        p = Path(row["relative_video_path"])
        if p.is_absolute() or ".." in p.parts or row["dimension"] != "dynamics_degree":
            raise ValueError("only safe relative Dynamic development inputs allowed")
    return selected


def contact_sheet(frames, times, title):
    # Spatial resize is DISPLAY ONLY. No frame skipping or score computation.
    height, width = frames.shape[1:3]
    cell_w = 256
    cell_h = max(1, round(height * cell_w / width))
    rows = (len(frames) + 3) // 4
    image = np.full((64 + rows * (cell_h + 28), 4 * cell_w, 3), 250, np.uint8)
    cv2.putText(image, title[:110], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(image, 'Every native frame / DISPLAY ONLY / motion classes not inferred from prompt',
                (8, 47), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
    for i, frame in enumerate(frames):
        x, y = (i % 4) * cell_w, 64 + (i // 4) * (cell_h + 28)
        stamp = f'{times[i]:.3f} s' if times[i] is not None else 'time unknown'
        cv2.putText(image, f'{i:03d} | {stamp}', (x + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, (20, 20, 20), 1, cv2.LINE_AA)
        image[y + 28:y + 28 + cell_h, x:x + cell_w] = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    return image


def display_source(path, allow_unknown_gif_timing=False):
    if path.suffix.lower() == '.gif' and allow_unknown_gif_timing:
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            frames, delays = [], []
            for frame in ImageSequence.Iterator(im):
                frames.append(np.asarray(frame.convert('RGB')))
                delay = frame.info.get('duration')
                delays.append(delay / 1000 if delay is not None and delay > 0 else None)
        if any(delay is None for delay in delays):
            # Frame order is visible, but GIF playback defaults are not native
            # physical timestamps. Never fabricate 8/100 FPS to make it pass.
            return np.stack(frames), [None] * len(frames), None, {
                'timing_status': 'unknown_missing_GIF_delays', 'frame_delays_seconds': delays,
                'decoder': 'Pillow RGB composite for display only', 'speed_calibration_eligible': False}
    frames, pts, fps = native_video(path, .001)
    return frames, pts, fps, {'timing_status': 'verified_native_CFR',
                            'decoder': 'existing native_video', 'speed_calibration_eligible': None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--anonymous-title', action='store_true',
                   help='display UID only, without generator, prompt or score')
    p.add_argument('--allow-unknown-gif-timing', action='store_true',
                   help='display frame order with null times; never qualify these for speed/scoring')
    args = p.parse_args(argv)
    sources, videos, output = Path(args.sources), Path(args.video_root), Path(args.output)
    rows = select_development([json.loads(s) for s in sources.read_text().splitlines()])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, row in enumerate(rows):
        path = videos / row['relative_video_path']
        item = {**row, 'status': 'unreviewed', 'motion_categories': None, 'human_review': 'NOT RUN',
                'scoring_input': False, 'calibration_anchor': False, 'video': str(path)}
        try:
            item['source_sha256'] = digest(path)
            frames, pts, fps, timing = display_source(path, args.allow_unknown_gif_timing)
            item.update(shape=list(frames.shape), timestamps=pts, fps=fps,
                        decoded_pixels_sha256=hashlib.sha256(frames.tobytes()).hexdigest(), **timing)
            name = f'{i:02d}_{row["video_uid"]}.png'
            title = row['video_uid'] if args.anonymous_title else f'{i:02d} | {row["generator"]} | {row["prompt_id"]}'
            rgb = contact_sheet(frames, pts, title)
            if not cv2.imwrite(str(output / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError('contact-sheet write failed')
            item.update(display=name, display_sha256=digest(output / name))
        except Exception as exc:
            item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        records.append(item)
        print(json.dumps({k: item.get(k) for k in ('video_uid','status','shape','fps','error')}), flush=True)
    identity = {'sources_sha256': digest(sources), 'script_sha256': digest(Path(__file__)),
                'decoder_source_sha256': digest(Path(__file__).with_name('official_video_jitter.py')),
                'pixel_decoder_source_sha256': digest(Path(__file__).with_name('static_jitter.py')),
                'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'allow_unknown_gif_timing': args.allow_unknown_gif_timing,
                'anonymous_title': args.anonymous_title,
                'scope': 'predeclared dev only; no reserved test input, score, intervention or automatic motion label',
                'records': records}
    (output / 'provenance.json').write_text(json.dumps(identity, indent=2))
    return 1 if any(r['status']=='failed' for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
