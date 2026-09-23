"""Score-blind natural DEV calibration, prompt-disjoint from DEV32 and holdout.

All remaining DEV prompts, one SHA256('20260923:<uid>')-ranked official MP4
per prompt/generator (LaVie, ModelScope, VideoCrafter). No CFs are constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .static_jitter import digest, rank


def select_calibration(pool, excluded):
    forbidden_prompts = {r['prompt_id'] for r in excluded}
    forbidden_uids = {r['video_uid'] for r in excluded}
    eligible = [r for r in pool if r['dimension'] == 'dynamics_degree' and r['split'] == 'dev'
                and r['prompt_id'] not in forbidden_prompts and r['video_uid'] not in forbidden_uids
                and r['relative_video_path'].endswith('.mp4')]
    selected = []
    for prompt in sorted({r['prompt_id'] for r in eligible}):
        for generator in ('lavie', 'modelscope', 'videocraft'):
            candidates = [r for r in eligible if r['prompt_id'] == prompt and r['generator'] == generator]
            if not candidates:
                raise ValueError('cannot keep prompt/generator balance; no fallback selection')
            pick = min(candidates, key=lambda r: rank(r['video_uid'], 20260923))
            selected.append(dict(pick, base_id=pick['video_uid'], selection_role='independent_natural_scale_calibration'))
    if len(selected) != 63 or len({r['video_uid'] for r in selected}) != 63:
        raise ValueError('frozen calibration design requires 21 prompts x 3 generators = 63 unique MP4s')
    return selected


def validate_calibration(rows, sources):
    if (len(rows) != 63 or len(sources) != 63 or len({r['video_uid'] for r in sources}) != 63
            or len({r['candidate_id'] for r in rows}) != 63
            or {r['base_id'] for r in rows} != {r['video_uid'] for r in sources}
            or len({r['prompt_id'] for r in sources}) != 21
            or set(Counter(r['prompt_id'] for r in sources).values()) != {3}):
        raise ValueError('complete natural calibration cohort required')
    source_by_id = {r['video_uid']: r for r in sources}
    for row in rows:
        source = source_by_id[row['base_id']]
        if (row['protocol'] != 'natural-intensity-calibration-v1' or row['split'] != 'dev'
                or row['family'] != 'original' or row['seed'] != 0 or row['amplitude'] != 0
                or source['split'] != 'dev' or not source['relative_video_path'].endswith('.mp4')
                or (row['prompt_id'], row['generator']) != (source['prompt_id'], source['generator'])):
            raise ValueError('only unmodified official natural DEV MP4s may fit the scale')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool')
    parser.add_argument('--exclude', nargs='+')
    parser.add_argument('--sources')
    parser.add_argument('--video-root', help='materialize an identity ledger only; no video copying/editing')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not args.video_root:
        if not args.pool or not args.exclude:
            parser.error('--pool and --exclude required for score-blind selection')
        excluded = [r for p in args.exclude for r in map(json.loads, Path(p).read_text().splitlines())]
        with Path(args.pool).open() as handle:
            selected = select_calibration(list(csv.DictReader(handle)), excluded)
        receipt = output.with_suffix('.selection.json')
        if output.exists() or receipt.exists():
            raise FileExistsError('fresh outputs required')
        with output.open('x') as handle:
            for row in selected:
                handle.write(json.dumps(row) + '\n')
        identity = {'protocol': 'natural-intensity-calibration-selection-v1', 'rule': __doc__,
                    'seed': 20260923, 'sources': 63, 'prompts': 21,
                    'generator_counts': dict(Counter(r['generator'] for r in selected)),
                    'pool_sha256': digest(Path(args.pool)),
                    'excluded_sha256': {p: digest(Path(p)) for p in args.exclude},
                    'sources_sha256': digest(output), 'script_sha256': digest(Path(__file__)),
                    'scores_read': False, 'holdout_media_opened': False,
                    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        receipt.write_text(json.dumps(identity, indent=2))
        print(json.dumps(identity, indent=2))
        return
    from .official_video_jitter import native_video
    sources = [json.loads(s) for s in Path(args.sources).read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (output / 'candidates.jsonl').open('x') as handle:
        for ordinal, source in enumerate(sources):
            path = (Path(args.video_root) / source['relative_video_path']).resolve(strict=True)
            if not path.is_relative_to(Path(args.video_root).resolve(strict=True)):
                raise ValueError('source outside official video root')
            frames, pts, fps = native_video(path, 1e-6)
            if len(frames) != 16 or abs(fps-8) > 1e-6 or not np.allclose(np.diff(pts), .125, atol=1e-6, rtol=0):
                raise ValueError('native calibration cadence outside fixed 16-frame/8-FPS design')
            row = dict(source, candidate_id=f'calibration_{ordinal:06d}',
                       protocol='natural-intensity-calibration-v1', family='original', seed=0, amplitude=0,
                       video=str(path), sha256=digest(path), status='unmodified_original', reason=None,
                       decoded_shape=list(frames.shape), pts=pts, fps=fps,
                       decoded_pixels_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                       pixel_exact_to_intended=True, native_timeline_preserved=True)
            rows.append(row)
            handle.write(json.dumps(row) + '\n')
    validate_calibration(rows, sources)
    completion = {'status': 'finished', 'videos': len(rows), 'media_duration_seconds': 2.,
                  'manifest_sha256': digest(output / 'candidates.jsonl'),
                  'sources_sha256': digest(Path(args.sources)), 'script_sha256': digest(Path(__file__)),
                  'holdout_media_opened': False, 'videos_generated': 0,
                  'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (output / 'completion.json').write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2))


if __name__ == '__main__':
    main()
