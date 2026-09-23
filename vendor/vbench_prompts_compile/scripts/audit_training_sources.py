#!/usr/bin/env python3
"""Audit frozen training text and visual identities without selecting new models.

Lexical similarity is a screening diagnostic, not proof of common provenance.
Public class names and independently sourced examples can legitimately overlap.
The report preserves exact matches and near matches for every matrix condition.
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.records import write_jsonl
from cache_backend_outputs import sha


def norm(text):return ' '.join(re.findall(r'\w+',text.lower()))


def video_identity(value):
    value=str(value).replace('\\','/')
    if '/videos/' in value:return 'videos/'+value.split('/videos/',1)[1]
    return value if value.startswith('videos/') else None


def near_match(a,b):
    if a==b:return 1.,'normalized_exact'
    # Fixed audit thresholds; never used to filter scores or retrain on outcomes.
    short,long=sorted([a.split(),b.split()],key=len)
    contained=len(short)>=12 and (' '+' '.join(short)+' ') in (' '+' '.join(long)+' ')
    if 2*min(len(a),len(b))/max(1,len(a)+len(b))<.9 and not contained:return 0.,None
    ratio=SequenceMatcher(None,a,b,autojunk=False).ratio()
    if ratio>=.9:return ratio,'character_similarity_at_least_0.90'
    if contained:
        return ratio,'verbatim_containment_at_least_12_words'
    return ratio,None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',required=True)
    p.add_argument('--training',action='append',required=True,help='task=directory containing train/dev.jsonl')
    p.add_argument('--cache',action='append',default=[])
    p.add_argument('--annotation-plan',action='append',default=[])
    p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    matrix=J.read_jsonl(Path(args.matrix))
    visual_ids={video_identity(r['relative_path']) for r in matrix}
    visual_hashes=set()
    for path in args.cache:
        for row in J.read_jsonl(Path(path)):
            visual_hashes.update(row.get('frame_sha256',[]))
            if row.get('video_sha256'):visual_hashes.add(row['video_sha256'])
    annotation_prompts=set()
    for path in args.annotation_plan:
        for row in J.read_jsonl(Path(path)):
            annotation_prompts.add(norm(row.get('input',row)['prompt']))
            meta=row.get('meta',{})
            if meta.get('frame_source_prompt'):annotation_prompts.add(norm(meta['frame_source_prompt']))
            if value:=video_identity(meta.get('frame_video','')):visual_ids.add(value)
    pairs=[];report={}
    paths=[args.matrix,*args.cache,*args.annotation_plan]
    for item in args.training:
        task,directory=item.split('=',1)
        splits={name:J.read_jsonl(Path(directory)/(name+'.jsonl')) for name in ['train','dev']}
        paths.extend(str(Path(directory)/(name+'.jsonl')) for name in splits)
        test=defaultdict(set)
        for r in matrix:
            if r['task']==task:test[(r['prompt'],r['transform'])].add(r['family'])
        task_report={'splits':{},'test_original_families':len({r['family'] for r in matrix if r['task']==task and r['transform']=='identity'}),
            'train_dev_group_overlap':sorted({r['group_id'] for r in splits['train']} & {r['group_id'] for r in splits['dev']})}
        for split,records in splits.items():
            unique={}
            for r in records:unique.setdefault(norm(r['input']['prompt']),r)
            exact=Counter();near=Counter();matched_families=defaultdict(set)
            for (prompt,condition),families in sorted(test.items()):
                query=norm(prompt);hits=[]
                for text,row in unique.items():
                    ratio,kind=near_match(query,text)
                    if kind:
                        hits.append((ratio,kind,row))
                if any(k=='normalized_exact' for _,k,_ in hits):exact[condition]+=1
                if hits:
                    near[condition]+=1;matched_families[condition].update(families)
                for ratio,kind,row in sorted(hits,key=lambda h:(-h[0],h[2]['sample_id'])):
                    pairs.append({'task':task,'split':split,'test_prompt':prompt,'transform':condition,
                        'test_families':sorted(families),'training_sample_id':row['sample_id'],
                        'training_group_id':row['group_id'],'training_source':row['source'],
                        'training_prompt':row['input']['prompt'],'similarity':ratio,'match_type':kind})
            video_overlap=[];hash_overlap=[];annot_overlap=[]
            for row in records:
                meta=row.get('meta',{})
                values=[meta.get(k) for k in ['frame_video','video_path','source_video'] if meta.get(k)]
                if any(video_identity(v) in visual_ids for v in values):video_overlap.append(row['sample_id'])
                if any(meta.get(k) in visual_hashes for k in ['frame_sha256','video_sha256','image_sha256']):hash_overlap.append(row['sample_id'])
                if task=='scene' and ({norm(row['input']['prompt']),norm(meta.get('frame_source_prompt',''))}&annotation_prompts):
                    annot_overlap.append(row['sample_id'])
            task_report['splits'][split]={'rows':len(records),'unique_prompts':len(unique),
                'sources':dict(Counter(r['source'] for r in records)),
                'exact_unique_test_prompts_by_transform':dict(exact),
                'near_or_exact_unique_test_prompts_by_transform':dict(near),
                'screened_families_by_transform':{k:len(v) for k,v in matched_families.items()},
                'test_video_identity_overlaps':video_overlap,'test_visual_hash_overlaps':hash_overlap,
                'annotation_plan_prompt_endpoint_overlaps':annot_overlap,
                'visual_identity_available':sum(bool(r.get('meta',{}).get('frame_video')) for r in records)}
        report[task]=task_report
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    write_jsonl(out.with_suffix('.pairs.jsonl'),pairs)
    J.atomic_json(out,{'tasks':report,'input_sha256':{v:sha(v) for v in paths},
        'method':'normalized exact plus character similarity >= 0.90 or >= 12-word verbatim containment',
        'note':'Similarity flags are diagnostic, not common-source proof. Visual identity audit only covers recorded paths/hashes. No scores, labels or model selection used.'})
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
