#!/usr/bin/env python3
"""Objects controls on identical complete video pairs, plus detector side effects.

No proportional score drop is assumed for partial occlusion. Monotonicity is a
secondary descriptive statistic. Background contrasts use only jointly complete
target/background rows and never conflate geometric coverage with visibility.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.experiments import cluster_summary
from cache_backend_outputs import sha
from cache_matrix_transforms import evidence_id


def matched_contrasts(rows):
    by={(r['relative_path'],r['scheme'],r.get('occlusion_control'),r.get('level')):r
        for r in rows if r['task']=='objects' and r.get('occlusion_control')}
    grouped=defaultdict(list)
    for (path,scheme,control,level),target in by.items():
        if control!='target' or level==0:continue
        background=by.get((path,scheme,'background',level))
        if not background or not target['pair_complete'] or not background['pair_complete']:continue
        if target['original_score']!=background['original_score']:raise ValueError('different baselines for controls')
        grouped[(scheme,level)].append({'family':target['family'],'relative_path':path,
            'original':target['original_score'],'target':target['transformed_score'],
            'background':background['transformed_score'],
            'target_minus_background':target['transformed_score']-background['transformed_score']})
    return grouped


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--transforms',required=True);p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    rows=[r for r in J.read_jsonl(Path(args.paired)) if r['task']=='objects']
    base={r['relative_path']:r for r in J.read_jsonl(Path(args.cache))}
    status=json.loads(Path(args.transforms).with_suffix('.report.json').read_text())
    if not status.get('all_attempted') or len(rows)!=39*len(base):
        raise ValueError('incomplete Objects matrix execution')
    transforms=J.read_jsonl(Path(args.transforms));by_evidence={r['evidence_id']:r for r in transforms}
    integrity=Counter();retention=defaultdict(list)
    for record in transforms:
        source=base[record['relative_path']]
        if not record.get('frame_transform'):continue
        target,other=record['official_target'].split(' and ')
        for i,detail in enumerate(record['frame_transform']):
            integrity[detail['status']]+=1
            if detail['status']!='ok':continue
            if not detail.get('outside_unchanged'):raise ValueError('unverified outside-patch integrity')
            labels=record['frame_labels'][i]
            if labels is None:continue
            original=source['frame_labels'][i]
            for label,name in [(target,'target'),(other,'other')]:
                if label in original:
                    retention[(record['occlusion_control'],record['level'],name)].append({
                        'family':record['family'],'retained':label in labels})
            if record['level']==0 and record['frame_sha256'][i]!=source['frame_sha256'][i]:
                raise ValueError('level-zero changed pixels')
            if record['occlusion_control']=='background':
                target_record=by_evidence[evidence_id({**record,'occlusion_control':'target'})]
                target_detail=target_record['frame_transform'][i]
                if target_detail['status']=='ok' and target_detail['mask_pixels']!=detail['mask_pixels']:
                    raise ValueError('background and target masks have unequal areas')
    contrasts=[]
    for (scheme,level),values in sorted(matched_contrasts(rows).items()):
        contrasts.append({'scheme':scheme,'linear_fraction':level,'planned_videos':len(base),
            **{k:cluster_summary(values,k) for k in ['original','target','background','target_minus_background']}})
    ladders=defaultdict(dict)
    for r in rows:
        if r.get('occlusion_control'):
            ladders[(r['relative_path'],r['scheme'],r['occlusion_control'])][r['level']]=r
    monotone=defaultdict(list)
    for (path,scheme,control),levels in ladders.items():
        if set(levels)!={0.,.5,.8,1.} or not all(r['pair_complete'] for r in levels.values()):continue
        scores=[levels[v]['transformed_score'] for v in [0.,.5,.8,1.]]
        monotone[(scheme,control)].append({'family':levels[0.]['family'],
            'monotone_within_0.03':all(b<=a+.03 for a,b in zip(scores,scores[1:]))})
    report={'input_sha256':{v:sha(v) for v in [args.paired,args.cache,args.transforms]},
        'planned_videos':len(base),'frame_geometry_status':dict(integrity),
        'outside_patch_and_equal_area_checks_passed':True,'checks_apply_to':'successfully constructed frames only; failed geometry remains listed',
        'matched_background_contrasts':contrasts,
        'detector_label_retention':[{'control':k[0],'linear_fraction':k[1],'label':k[2],
            'retention_given_original_detection':cluster_summary(v,'retained')} for k,v in sorted(retention.items())],
        'complete_ladder_monotonicity':[{'scheme':k[0],'control':k[1],
            'summary':cluster_summary(v,'monotone_within_0.03')} for k,v in sorted(monotone.items())],
        'note':'Controls are compared on the same complete source videos. Retention measures the detector, not true visibility. Partial occlusion need not imply proportional degradation; monotonicity is descriptive.'}
    J.atomic_json(Path(args.out),report)
    print(json.dumps({'complete':True,'matched_conditions':len(contrasts),'geometry':dict(integrity)}));return 0


if __name__=='__main__':raise SystemExit(main())
