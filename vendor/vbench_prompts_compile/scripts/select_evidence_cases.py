#!/usr/bin/env python3
"""Select first sorted illustrative cases for known failure mechanisms.

This selection is for explanation only; no selected subset replaces any metric
denominator. The plan retains exact cache records and raw student outputs.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.experiments import text_key
from vbench_prompts_compile.official_replay import position_score,spatial_scores,object_scores
from cache_matrix_transforms import evidence_id
from cache_backend_outputs import sha


def signed_unique_frame(target,frame):
    a=[r['box'] for r in frame if r['label']==target['object_a']]
    b=[r['box'] for r in frame if r['label']==target['object_b']]
    if len(a)!=1 or len(b)!=1:return False
    a,b=a[0],b[0];dx=(b[0]+b[2]-a[0]-a[2])/2;dy=(b[1]+b[3]-a[1]-a[3])/2
    direction=target['relationship']
    right_sign={'on the left of':dx>0,'on the right of':dx<0,'on the top of':dy>0,'on the bottom of':dy<0}.get(direction,False)
    return right_sign and position_score(direction,a,b)>0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',required=True);p.add_argument('--paired',required=True);p.add_argument('--cache-dir',required=True)
    p.add_argument('--spatial-transforms',required=True);p.add_argument('--predictions',action='append',required=True)
    p.add_argument('--scene-test2',required=True);p.add_argument('--scene-synonyms',required=True)
    p.add_argument('--visibility');p.add_argument('--object-transforms');p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    matrix=J.read_jsonl(Path(args.matrix));paired=J.read_jsonl(Path(args.paired))
    caches={t:{r['relative_path']:r for r in J.read_jsonl(Path(args.cache_dir)/(t+'-test.jsonl'))} for t in ['spatial','objects','action','scene']}
    transforms={r['evidence_id']:r for r in J.read_jsonl(Path(args.spatial_transforms))}
    pred={r['request_id']:r for path in args.predictions for r in J.read_jsonl(Path(path))}
    originals={(r['task'],r['relative_path'],r['scheme']):r for r in paired if r['transform']=='identity'}
    cases=[]
    # A frame with exactly one detection of each object and the correct signed
    # relation is a detector-geometry proxy, never claimed as a human label.
    found_axes=set()
    for row in sorted((r for r in matrix if r['task']=='spatial' and r['transform']=='evidence_mirror' and r['eligible']),key=lambda r:r['relative_path']):
        if row['axis'] in found_axes:continue
        cache=caches['spatial'][row['relative_path']]
        indices=[i for i,f in enumerate(cache['frame_detections']) if signed_unique_frame(row['official_target'],f)]
        if not indices:continue
        i=indices[0];changed=transforms[evidence_id(row)]
        before=spatial_scores(row['official_target'],cache['frame_detections'])[i]
        after=spatial_scores(row['official_target'],changed['frame_detections'])[i]
        found_axes.add(row['axis'])
        cases.append({'id':'spatial-direction-'+row['axis'],'task':'spatial','title':row['axis']+': mirrored direction, unchanged official score',
            'note':'The unique detector boxes satisfy the original signed relation. Reflecting the evidence reverses it, yet the locked official formula ignores direction sign. This is a detector-geometry proxy, not human ground truth; images illustrate the exact evidence reflection.',
            'prompt':row['prompt'],'frame_index':i,'cache':cache,'transformed':changed,
            'scores':{'Origin before':before,'Origin after':after},'official_target':row['official_target'],
            'boxes_before':cache['frame_detections'][i],'boxes_after':changed['frame_detections'][i]})
        if found_axes=={'hflip','vflip'}:break
    for task in ['spatial','action']:
        for key,row in sorted(originals.items()):
            if key[0]!=task or key[2]!='Repair-model' or not row['eligible']:continue
            origin=originals[task,key[1],'Origin']
            if row['original_score']>=origin['original_score']:continue
            cache=caches[task][key[1]]
            indices=[i for i,(a,b) in enumerate(zip(origin['original_frame_scores'],row['original_frame_scores'])) if a>b]
            i=indices[0] if task=='spatial' else 8
            model=pred[text_key(task,row['prompt'])]
            cases.append({'id':task+'-interface','task':task,'title':task.title()+' text-interface failure',
                'note':'The source and visual backend are shared. Compare the raw student output, frozen canonicalization and actual backend target. These are interface differences; a higher video score is not itself proof of human correctness.',
                'prompt':row['prompt'],'frame_index':i,'cache':cache,'model_output':model,
                'origin_target':origin['original_target'],'model_backend_target':row.get('original_backend_target'),
                'video_scores':{'Origin':origin['original_score'],'Repair-model':row['original_score']}})
            break
    card=json.loads(Path(args.scene_test2).read_text())
    for truth in ['contradicted','insufficient','supported']:
        rows=sorted((r for r in card['rows'] if r['condition']!='synonym' and r['expected']==truth and r['schemes']['Repair-model']['label']!=truth),key=lambda r:r['source_id'])
        if not rows:continue
        row=rows[0]
        cases.append({'id':'scene-label-'+truth,'task':'scene','title':'Scene label disagreement: '+truth,
            'note':'The student sees only the prompt and caption. The displayed frame is contextual evidence, not student input. Expected is a model-derived caption label; ambiguity or teacher error remains possible and the label is not edited after seeing the student.',
            'prompt':row['prompt'],'caption':row['caption'],'expected_annotation':row['expected'],
            'predictions':{k:v['label'] for k,v in row['schemes'].items()},
            'annotation_image':row['image'],'frame_index':row['frame_index'],'source_id':row['source_id']})
    card=json.loads(Path(args.scene_synonyms).read_text())
    for row in sorted(card['rows'],key=lambda r:r['source_id']):
        if row['condition']!='synonym' or row['expected']!='supported':continue
        official=row['schemes']['Origin'];model=row['schemes']['Repair-model']
        if official['base_label']!='supported' or official['label']!='contradicted' or not model['correct_and_consistent']:continue
        relative='videos/'+row['video_path'].split('/videos/',1)[1];cache=caches['scene'][relative]
        if cache['frame_captions'][row['frame_index']]!=row['caption']:raise ValueError('Scene caption differs')
        cases.append({'id':'scene-synonym','task':'scene','title':'Scene synonym breaks the literal matcher',
            'note':'Both prompts use the exact same cached caption and image. The strict synonym relation was frozen before evaluation; this example is illustrative, and all pairs remain in the table.',
            'original_prompt':row['original_prompt'],'changed_prompt':row['prompt'],'caption':row['caption'],
            'expected_annotation':row['expected'],'predictions':row['schemes'],'frame_index':row['frame_index'],'cache':cache})
        break
    if args.visibility and args.object_transforms:
        reviewed=sorted(J.read_jsonl(Path(args.visibility)),key=lambda r:r['evidence_id'])
        changed={r['evidence_id']:r for r in J.read_jsonl(Path(args.object_transforms))}
        found=set()
        for review in reviewed:
            if review['status']!='annotated':continue
            record=changed[review['evidence_id']];cache=caches['objects'][review['relative_path']]
            scores=object_scores(record['official_target'],record['frame_labels'])
            kind=None
            if not review['endpoint_invisible_verified'] and 'visible' in review['labels']['after_target']:
                kind='residual';i=review['labels']['after_target'].index('visible')
            elif review['endpoint_invisible_verified'] and any(scores):
                kind='detector';i=next(i for i,v in enumerate(scores) if v)
            elif review.get('removal_verified'):
                kind='verified-control';i=8
            if kind is None or kind in found:continue
            found.add(kind)
            cases.append({'id':'objects-'+kind,'task':'objects','title':'Objects '+kind.replace('-',' ')+' example',
                'note':'Visibility is same-model multi-pass consensus, not human gold. Complete box coverage alone does not prove target invisibility. The displayed patch is reconstructed from the scored transform and hash-checked.',
                'prompt':record['prompt'],'frame_index':i,'cache':cache,'transformed':record,
                'review':{k:review.get(k) for k in ['method','human_reviewed','endpoint_invisible_verified','removal_verified','valid_passes','labels']},
                'frame_score_after':scores[i],'labels_after':record['frame_labels'][i]})
    paths=[args.matrix,args.paired,args.spatial_transforms,args.scene_test2,args.scene_synonyms,*args.predictions]
    if args.visibility:paths.extend([args.visibility,args.object_transforms])
    J.atomic_json(Path(args.out),{'selection':'first sorted case per mechanism; illustrative, not prevalence estimates',
        'script_sha256':sha(__file__),'input_sha256':{v:sha(v) for v in paths},'cases':cases})
    print(json.dumps({'cases':len(cases),'ids':[r['id'] for r in cases]}));return 0


if __name__=='__main__':raise SystemExit(main())
