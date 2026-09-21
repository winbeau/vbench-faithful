"""Official-label analysis with dev-only ties and paired prompt-cluster CIs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_pairwise_statistics import calibrate
from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json, write_jsonl


def score(row, method):
    value = row.get('scores', {}).get(method, {})
    s = value.get('score')
    return float(s) if value.get('status') == 'succeeded' and s is not None and np.isfinite(s) else None


def evaluate_population(pairs, videos, methods, margins, *, resamples=10000, seed=20260920):
    if not pairs:
        return {'total_pairs': 0, 'methods': {}}
    correct, strict, valid = [], [], []
    for method in methods:
        gaps, present = [], []
        for pair in pairs:
            a, b = (score(videos[pair[k]], method) for k in ('video_a_uid', 'video_b_uid'))
            present.append(a is not None and b is not None)
            gaps.append(a-b if present[-1] else 0.)
        gaps, present = np.asarray(gaps), np.asarray(present)
        labels = np.asarray([float(p['human_label']) for p in pairs])
        pred = np.where(abs(gaps) <= margins[method], .5, (gaps > 0).astype(float))
        pred0 = np.where(gaps == 0, .5, (gaps > 0).astype(float))
        correct.append((pred == labels) & present)
        strict.append((pred0 == labels) & present)
        valid.append(present)
    correct, strict, valid = map(np.asarray, (correct, strict, valid))
    groups = sorted({p['prompt_id'] for p in pairs})
    counts = np.asarray([sum(p['prompt_id'] == g for p in pairs) for g in groups])
    sums = np.asarray([[row[[p['prompt_id'] == g for p in pairs]].sum() for g in groups] for row in correct])
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), (resamples, len(groups)))
    boot = sums[:, selected].sum(-1) / counts[selected].sum(-1)
    result = {}
    for i, method in enumerate(methods):
        good = int(valid[i].sum())
        common = valid[0] & valid[i]
        result[method] = {'tie_margin_dev_only': margins[method], 'total_pairs': len(pairs),
            'scored_pairs': good, 'coverage': good/len(pairs),
            'accuracy_no_ties_full_denominator': float(strict[i].mean()),
            'accuracy_full_denominator': float(correct[i].mean()),
            'accuracy_scored_pairs': float(correct[i].sum()/good) if good else None,
            'ci95_prompt_cluster': np.quantile(boot[i], [.025, .975]).tolist() if len(groups)>1 else None,
            'paired_delta_vs_official_full_denominator': float(correct[i].mean()-correct[0].mean()),
            'paired_delta_ci95': np.quantile(boot[i]-boot[0], [.025, .975]).tolist() if len(groups)>1 else None,
            'common_scored_pairs_with_official': int(common.sum()),
            'delta_on_common_pairs': float((correct[i, common].astype(float)-correct[0, common]).mean()) if common.any() else None}
    return {'total_pairs': len(pairs), 'source_prompt_clusters': len(groups), 'methods': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'configs/subject-repair/natural1440_manifest.jsonl')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/subject-repair/natural1440_protocol_v2.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if sha256_file(args.manifest) != protocol['manifest_sha256']:
        raise ValueError('manifest changed')
    manifests = {r['video_uid']: r for r in read_jsonl(args.manifest)}
    records, runs = [], []
    for directory in sorted(args.run.glob('shard*')):
        if not directory.is_dir():
            continue
        run = json.loads((directory/'run.json').read_text())
        if not run.get('completed') or run['protocol_sha256'] != sha256_file(args.protocol):
            raise ValueError('incomplete shard or protocol mismatch')
        if sha256_file(directory/'scores.jsonl') != run['scores_sha256']:
            raise ValueError('shard score hash mismatch')
        rows = read_jsonl(directory/'scores.jsonl')
        if len(rows) != run['video_count']:
            raise ValueError('shard row count mismatch')
        records.extend(rows); runs.append(run)
    videos = {r['video_uid']: r for r in records}
    if len(videos) != len(records) or set(videos) != set(manifests):
        raise ValueError('duplicate, missing or unexpected official videos')
    for uid, row in videos.items():
        if any(row[k] != manifests[uid][k] for k in ('prompt_id','split','relative_video_path','subject_en')):
            raise ValueError('manifest identity mismatch')
    pair_path = ROOT/'data/processed/pairwise_master_split.csv'
    if sha256_file(pair_path) != protocol['pair_labels_sha256']:
        raise ValueError('human pair labels changed')
    pairs = [p for p in csv.DictReader(pair_path.open()) if p['dimension']=='subject_consistency']
    if len(pairs) != 2160:
        raise ValueError('expected all 2160 official subject pairs')
    for p in pairs:
        for k in ('video_a_uid','video_b_uid'):
            if videos[p[k]]['prompt_id'] != p['prompt_id'] or videos[p[k]]['split'] != p['split']:
                raise ValueError('pair/clip prompt or split mismatch')
    methods = ['official','aggregation'] + [m+'_'+policy for m in protocol['representations'] for policy in ('zero','exclude')]
    dev = [p for p in pairs if p['split']=='dev']; test = [p for p in pairs if p['split']=='test']
    margins = {}
    for method in methods:
        valid = []
        for p in dev:
            a,b = (score(videos[p[k]], method) for k in ('video_a_uid','video_b_uid'))
            if a is not None and b is not None:
                valid.append((p,a-b))
        if not valid:
            raise ValueError(f'no scored dev pairs for {method}')
        margins[method] = calibrate(valid)
    kwargs = {'resamples': protocol['analysis']['bootstrap_resamples'], 'seed': protocol['analysis']['bootstrap_seed']}
    report = {'stage': protocol['stage'], 'n_videos': len(videos), 'n_pairs': len(pairs),
              'video_status_counts': dict(Counter(r['status'] for r in records)),
              'primary_method': protocol['primary_method'], 'all_test': evaluate_population(test,videos,methods,margins,**kwargs),
              'test_without_masked_development_prompts': evaluate_population(
                  [p for p in test if p['prompt_id'] not in protocol['known_masked_development_prompts']],videos,methods,margins,**kwargs),
              'by_subject': {}, 'by_generator': {}, 'coverage_strata': {}, 'raw_scores': {}}
    for subject in sorted(set(r['subject_en'] for r in records)):
        pp = [p for p in test if videos[p['video_a_uid']]['subject_en']==subject]
        report['by_subject'][subject] = evaluate_population(pp,videos,methods,margins,**kwargs)
    for generator in sorted(set(r['generator'] for r in records)):
        pp = [p for p in test if generator in (videos[p['video_a_uid']]['generator'],videos[p['video_b_uid']]['generator'])]
        report['by_generator'][generator] = evaluate_population(pp,videos,methods,margins,**kwargs)
    def coverage(uid):
        d=videos[uid].get('localizer_diagnostics')
        return 1-d['num_missing_frames']/d['num_frames'] if d else 0.
    for threshold in protocol['analysis']['coverage_strata']:
        pp=[p for p in test if min(coverage(p[k]) for k in ('video_a_uid','video_b_uid'))>=threshold]
        report['coverage_strata'][str(threshold)] = evaluate_population(pp,videos,methods,margins,**kwargs)
    for method in methods:
        values=[score(r,method) for r in records];values=[s for s in values if s is not None]
        report['raw_scores'][method]={'n_scored':len(values),'quantiles_0_25_50_75_100':np.quantile(values,[0,.25,.5,.75,1]).tolist(),
                                      'mean':float(np.mean(values)),'n_zero':sum(v==0 for v in values)}
    stats = [r['localizer_diagnostics'] for r in records if r.get('localizer_diagnostics')]
    total_frames = sum(d['num_frames'] for d in stats)
    report['localization'] = {'n_frames':total_frames,'n_missing_frames':sum(d['num_missing_frames'] for d in stats),
         'n_multi_instance_frames':sum(d['num_multi_instance_frames'] for d in stats),
         'n_videos_all_frames_present':sum(coverage(uid)==1 for uid in videos),
         'n_videos_no_frames_present':sum(coverage(uid)==0 for uid in videos),
         'by_subject':{s:{'n_videos':sum(r['subject_en']==s for r in records),
             'mean_frame_presence':float(np.mean([coverage(r['video_uid']) for r in records if r['subject_en']==s]))}
             for s in sorted(set(r['subject_en'] for r in records))}}
    official_path=ROOT/'results/e0/raw_official_scores/subject_consistency/results.csv'
    old_repair_path=ROOT/'output/supplementary_20260914/repair/subject_consistency/predictions.csv'
    assert sha256_file(official_path)==protocol['official_scores_sha256']
    assert sha256_file(old_repair_path)==protocol['old_aggregation_predictions_sha256']
    report['frozen_score_parity']={}
    for method,path,key in [('official',official_path,'score'),('aggregation',old_repair_path,'repair_score')]:
        prior={r['video_uid']:float(r[key]) for r in csv.DictReader(path.open())}
        errors=[abs(score(videos[uid],method)-s) for uid,s in prior.items() if score(videos[uid],method) is not None]
        report['frozen_score_parity'][method]={'n_compared':len(errors),'max_absolute_error':max(errors),
                                             'within_1e_6':sum(e<=1e-6 for e in errors)}
    report['timing']={'gpu_devices':[r['physical_gpu'] for r in runs],
        'first_started_unix':min(r['started_unix'] for r in runs), 'last_finished_unix':max(r['finished_unix'] for r in runs),
        'four_gpu_wall_seconds':max(r['finished_unix'] for r in runs)-min(r['started_unix'] for r in runs),
        'sum_media_duration_seconds':sum(r['media_duration_seconds'] for r in runs)}
    report['provenance']={'protocol_sha256':sha256_file(args.protocol), 'manifest_sha256':sha256_file(args.manifest),
                          'score_shards_sha256':[r['scores_sha256'] for r in runs]}
    output=new_output(args.output)
    write_json(output/'statistics.json',report)
    write_jsonl(output/'scores_merged.jsonl',sorted(records,key=lambda r:r['video_uid']))
    with (output/'video_scores.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['video_uid','subject_en','generator','prompt_id','split','frame_presence',*methods])
        writer.writeheader()
        for r in sorted(records,key=lambda r:r['video_uid']):
            writer.writerow({**{k:r[k] for k in ['video_uid','subject_en','generator','prompt_id','split']},
                             'frame_presence':coverage(r['video_uid']),**{m:score(r,m) for m in methods}})
    print(json.dumps({'all_test':report['all_test'],'localization':report['localization'],
                      'frozen_score_parity':report['frozen_score_parity'],'timing':report['timing']},indent=2))


if __name__=='__main__':
    main()
