"""Replace untimed DEV GIFs with score-blind official DEV MP4 replicates.

Retain all 24 previously selected MP4s. In the original eight-prompt order,
replace each GIF with a second replicate from round-robin sorted MP4 generators.
Rank eligible video UIDs by SHA256('20260922:<uid>'); never read scores/media.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .static_jitter import digest, rank


def select_mp4(original, pool):
    dev = [r for r in original if r['split'] == 'dev']
    retained = [r for r in dev if r['relative_video_path'].endswith('.mp4')]
    rejected = [r for r in dev if not r['relative_video_path'].endswith('.mp4')]
    if len(dev) != 32 or len(retained) != 24 or len(rejected) != 8:
        raise ValueError('expected the original 24-MP4/8-GIF development cohort')
    generators = sorted({r['generator'] for r in retained})
    if len(generators) != 3:
        raise ValueError('expected three MP4 generators')
    forbidden = {r['video_uid'] for r in original}
    held_prompts = {r['prompt_id'] for r in original if r['split'] == 'test'}
    replacements, selected = [], []
    for row in dev:
        if row in retained:
            selected.append(dict(row, selection_role='retained_original_dev_mp4'))
            continue
        generator = generators[len(replacements) % len(generators)]
        eligible = [r for r in pool if r['dimension'] == 'dynamics_degree'
                    and r['split'] == 'dev' and r['prompt_id'] == row['prompt_id']
                    and r['prompt_id'] not in held_prompts and r['generator'] == generator
                    and r['relative_video_path'].endswith('.mp4') and r['video_uid'] not in forbidden]
        if not eligible:
            raise ValueError(f'no replacement under frozen rule: {generator}/{row["prompt_id"]}')
        pick = min(eligible, key=lambda r: rank(r['video_uid'], 20260922))
        forbidden.add(pick['video_uid'])
        selected.append(dict(pick, base_id=pick['video_uid'],
                             selection_role='score_blind_gif_replacement',
                             replaces_video_uid=row['video_uid'],
                             historical_exposure='VBench natural E0 input; not an unseen natural video'))
        replacements.append({'excluded_gif': row['video_uid'], 'replacement_mp4': pick['video_uid'],
                             'generator': generator, 'eligible_count': len(eligible),
                             'reason': 'missing native GIF frame delays; user authorized MP4 replacement'})
    if len({r['video_uid'] for r in selected}) != 32 or set(Counter(r['prompt_id'] for r in selected).values()) != {4}:
        raise ValueError('must retain 32 unique sources, eight prompts, four sources each')
    return selected, replacements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    original, pool, output = map(Path, (args.original, args.pool, args.output))
    with pool.open(newline='') as f:
        rows, replacements = select_mp4([json.loads(s) for s in original.read_text().splitlines()], list(csv.DictReader(f)))
    provenance = output.with_suffix('.selection.json')
    if output.exists() or provenance.exists():
        raise FileExistsError('fresh selection outputs required')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    identity = {'protocol': 'official-mp4-dev32-selection-v1', 'seed': 20260922,
                'original_sha256': digest(original), 'pool_sha256': digest(pool),
                'script_sha256': digest(Path(__file__)), 'output_sha256': digest(output),
                'rule': __doc__, 'sources': 32, 'prompts': 8,
                'generator_counts': dict(Counter(r['generator'] for r in rows)),
                'replacements': replacements, 'scores_read': False, 'holdout_media_opened': False}
    with provenance.open('x') as f:
        json.dump(identity, f, indent=2)
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
