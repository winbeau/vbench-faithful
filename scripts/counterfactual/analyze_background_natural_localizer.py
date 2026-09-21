"""Full-denominator background dev preference check for localization changes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .analyze_subject_natural import evaluate_population
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run/'run.json').read_text())
    protocol = json.loads((args.run/'protocol.json').read_text())
    if not run.get('completed') or sha256_file(args.run/'scores.jsonl') != run['scores_sha256']:
        raise ValueError('complete verified natural run required')
    rows = read_jsonl(args.run/'scores.jsonl'); videos = {r['video_uid']:r for r in rows}
    manifest = ROOT/'configs/background-repair/natural1720_manifest_v2.jsonl'
    if sha256_file(manifest) != protocol['manifest_sha256']:
        raise ValueError('natural population changed')
    expected = {r['video_uid']:r for r in read_jsonl(manifest) if r['split']=='dev'}
    if len(rows)!=680 or set(videos)!=set(expected):
        raise ValueError('incomplete or duplicated development population')
    for uid,row in videos.items():
        if any(row[k]!=v for k,v in expected[uid].items()):raise ValueError('natural input identity mismatch')
    labels = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(labels) != protocol['pair_labels_sha256']:
        raise ValueError('official background preference labels changed')
    pairs = [p for p in csv.DictReader(labels.open()) if p['dimension']=='background_consistency' and p['split']=='dev']
    if len(pairs)!=1020:raise ValueError('incomplete background development pairs')
    for pair in pairs:
        if any(videos[pair[k]]['prompt_id']!=pair['prompt_id'] for k in ['video_a_uid','video_b_uid']):
            raise ValueError('preference/source prompt mismatch')
    methods = ['official','frozen19','coco80','caption']
    stats = evaluate_population(pairs,videos,methods,{m:0 for m in methods})
    paired = evaluate_population(pairs,videos,['frozen19','caption'],{'frozen19':0,'caption':0})['methods']['caption']
    stats['paired_caption_vs_frozen19'] = {
        'delta_full_denominator':paired['paired_delta_vs_official_full_denominator'],
        'ci95_prompt_cluster':paired['paired_delta_ci95']}
    valid = [r for r in rows if r['status']=='completed']
    stats.update(status_counts=dict(Counter(r['status'] for r in rows)), video_count=len(rows),
        stage='All native background development inputs; no new independent test claim; no default promotion.',
        max_reference_origin_error=max((r['reference_origin_absolute_error'] for r in valid),default=None),
        max_reference_upstream_error=max((r['reference_upstream_absolute_error'] for r in valid),default=None),
        localization={'frames':sum(r['num_frames'] for r in valid),
            'empty_before_caption':sum(r['localizer_diagnostics']['num_empty_before_fallback'] for r in valid),
            'empty_after_caption':sum(r['localizer_diagnostics']['num_empty_foreground_frames'] for r in valid),
            'all_empty_before_videos':sum(r['localizer_diagnostics']['num_empty_before_fallback']==r['num_frames'] for r in valid),
            'all_empty_after_videos':sum(r['localizer_diagnostics']['num_empty_foreground_frames']==r['num_frames'] for r in valid),
            'interpretation':'Detection coverage, not annotated semantic correctness.'},
        timing={'one_gpu_wall_seconds':run['finished_unix']-run['started_unix'],
            'known_media_seconds':sum(r['media_duration_seconds'] for r in valid if r['media_duration_seconds'] is not None),
            'unknown_media_duration_videos':sum(r['media_duration_seconds'] is None for r in valid)},
        run_sha256=sha256_file(args.run/'run.json'),scores_sha256=run['scores_sha256'],
        analysis_source_sha256=sha256_file(Path(__file__)))
    output = new_output(args.output);write_json(output/'statistics.json',stats)
    print(json.dumps({'status_counts':stats['status_counts'],'methods':stats['methods'],
                      'paired_caption_vs_frozen19':stats['paired_caption_vs_frozen19']}))


if __name__ == '__main__':
    main()
